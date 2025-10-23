# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import concurrent.futures
import functools
import io
import logging
import math
import os
import random
import threading
import time
import urllib.parse
import uuid
from typing import (
    Optional,
    List,
    Tuple,
    Iterator,
    Union,
    Literal,
    TypedDict,
)

from azure.core.credentials import (
    AzureSasCredential,
    TokenCredential,
)
import azure.core.exceptions
from azure.identity import DefaultAzureCredential
import azure.storage.blob
import azure.storage.blob._generated.models
from azure.storage.blob._shared.response_handlers import process_storage_error
from azure.core.pipeline import Pipeline
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.core.pipeline.transport import RequestsTransport

from azstoragetorch._version import __version__
from azstoragetorch.exceptions import ClientRequestIdMismatchError


_LOGGER = logging.getLogger(__name__)

SDK_CREDENTIAL_TYPE = Optional[
    Union[
        AzureSasCredential,
        TokenCredential,
    ]
]
AZSTORAGETORCH_CREDENTIAL_TYPE = Union[SDK_CREDENTIAL_TYPE, Literal[False]]
SUPPORTED_WRITE_BYTES_LIKE_TYPE = Union[bytes, bytearray, memoryview]
STAGE_BLOCK_FUTURE_TYPE = concurrent.futures.Future[str]


class SDKKwargsType(TypedDict, total=False):
    connection_data_block_size: int
    transport: RequestsTransport
    user_agent: str
    credential: SDK_CREDENTIAL_TYPE
    _additional_pipeline_policies: List[SansIOHTTPPolicy]
    _pipeline: Pipeline


class DownloadKwargsType(TypedDict, total=False):
    range: str
    modified_access_conditions: (
        azure.storage.blob._generated.models.ModifiedAccessConditions
    )


# Policy to ensure that request made by the client matches responses returned. This
# is an extra guard rail to ensure when using an azstoragetorch interface with processes
# customers will not successfully process a response from a different request because
# the underlying connection resource was shared as part of forking:
# https://github.com/psf/requests/issues/4323
#
# The SDK has a policy that injects the client request ID but does not validate it echoes
# back. If the SDK was to enable this in the future, we should consider removing this policy
# in favor of the SDK's.
class EchoClientRequestIdPolicy(SansIOHTTPPolicy):
    _CLIENT_REQUEST_ID_HEADER_NAME = "x-ms-client-request-id"

    def on_request(self, request):
        request.http_request.headers[self._CLIENT_REQUEST_ID_HEADER_NAME] = str(
            uuid.uuid4()
        )

    def on_response(self, request, response):
        request_client_id = request.http_request.headers[
            self._CLIENT_REQUEST_ID_HEADER_NAME
        ]
        response_client_id = response.http_response.headers[
            self._CLIENT_REQUEST_ID_HEADER_NAME
        ]
        if request_client_id != response_client_id:
            raise ClientRequestIdMismatchError(
                request_client_id=request_client_id,
                response_client_id=response_client_id,
                service_request_id=response.http_response.headers["x-ms-request-id"],
            )


class AzStorageTorchBlobClientFactory:
    # Socket timeouts set to match the default timeouts in Python SDK
    _SOCKET_CONNECTION_TIMEOUT = 20
    _SOCKET_READ_TIMEOUT = 60
    _CONNECTION_DATA_BLOCK_SIZE = 256 * 1024

    def __init__(
        self,
        credential: AZSTORAGETORCH_CREDENTIAL_TYPE = None,
    ):
        self._sdk_credential = self._get_sdk_credential(credential)
        self._transport = self._get_transport()
        self._pipeline: Optional[Pipeline] = None

    def get_blob_client_from_url(self, blob_url: str) -> "AzStorageTorchBlobClient":
        blob_sdk_client = self._get_sdk_blob_client_from_url(blob_url)
        return AzStorageTorchBlobClient(blob_sdk_client)

    def yield_blob_clients_from_container_url(
        self, container_url: str, prefix: Optional[str] = None
    ) -> Iterator["AzStorageTorchBlobClient"]:
        container_sdk_client = self._get_sdk_container_client_from_container_url(
            container_url
        )
        blob_names = container_sdk_client.list_blob_names(name_starts_with=prefix)
        for blob_name in blob_names:
            blob_client = container_sdk_client.get_blob_client(blob_name)
            # Throwaway the blob client for it's URL to ensure we are **not**
            # sharing transports as list_blob_names() may happen in the parent
            # process and calling fork() after can lead to unintentional resource
            # sharing in child processes.
            yield self.get_blob_client_from_url(blob_client.url)

    def _get_sdk_credential(
        self, credential: AZSTORAGETORCH_CREDENTIAL_TYPE
    ) -> SDK_CREDENTIAL_TYPE:
        if credential is False:
            return None
        if credential is None:
            return DefaultAzureCredential()
        if isinstance(credential, (AzureSasCredential, TokenCredential)):
            return credential
        raise TypeError(f"Unsupported credential: {type(credential)}")

    def _get_transport(self) -> RequestsTransport:
        return RequestsTransport(
            connection_timeout=self._SOCKET_CONNECTION_TIMEOUT,
            read_timeout=self._SOCKET_READ_TIMEOUT,
            connection_data_block_size=self._CONNECTION_DATA_BLOCK_SIZE,
        )

    def _get_sdk_blob_client_from_url(
        self, blob_url: str
    ) -> azure.storage.blob.BlobClient:
        client = azure.storage.blob.BlobClient.from_blob_url(
            blob_url,
            **self._get_sdk_client_kwargs(blob_url),
        )
        self._cache_pipeline_if_needed(client, blob_url)
        return client

    def _get_sdk_container_client_from_container_url(
        self, container_url: str
    ) -> azure.storage.blob.ContainerClient:
        client = azure.storage.blob.ContainerClient.from_container_url(
            container_url,
            **self._get_sdk_client_kwargs(container_url, share_transport=False),
        )
        return client

    def _get_sdk_client_kwargs(
        self, resource_url: str, share_transport: bool = True
    ) -> SDKKwargsType:
        kwargs: SDKKwargsType = {
            "user_agent": f"azstoragetorch/{__version__}",
            "_additional_pipeline_policies": [
                EchoClientRequestIdPolicy(),
            ],
        }
        if share_transport:
            kwargs["transport"] = self._transport
        credential = self._sdk_credential
        if self._url_has_sas_token(resource_url):
            # The SDK prefers the explict credential over the one in the URL. So if a SAS token is
            # in the URL, we do not want the factory to automatically inject its credential, especially
            # if it would have been the default credential.
            credential = None
        elif share_transport and self._pipeline is not None:
            # We only want to share pipelines if we previously created a shareable pipeline and if
            # the resource URL provided does not have a SAS token override, which would require a new
            # pipeline since different credential policies are used based on credential provided.
            #
            # Technically, we also no longer need to provide a credential nor transport given these
            # are not used in favor of the pipeline provided, which would already have incorporated
            # them. We provide them still to avoid relying on the assumption that these resources
            # will be shared via the pipeline, especially because the pipeline is technically not
            # a public interface so that assumption could change in the future.
            kwargs["_pipeline"] = self._pipeline
        kwargs["credential"] = credential
        return kwargs

    def _url_has_sas_token(self, resource_url: str) -> bool:
        parsed_url = urllib.parse.urlparse(resource_url)
        if parsed_url.query is None:
            return False
        parsed_qs = urllib.parse.parse_qs(parsed_url.query)
        # The signature is always required in a valid SAS token. So look for the "sig"
        # key to determine if the URL has a SAS token.
        return "sig" in parsed_qs

    def _cache_pipeline_if_needed(
        self,
        client: Union[
            azure.storage.blob.ContainerClient, azure.storage.blob.BlobClient
        ],
        resource_url: str,
    ) -> None:
        if not self._url_has_sas_token(resource_url) and self._pipeline is None:
            # Only cache the pipeline if we did not have one previously cached and the client was not
            # created using a SAS token, which would override the factory's base credential.
            self._pipeline = client._pipeline


class AzStorageTorchBlobClient:
    _PARTITIONED_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
    _PARTITION_SIZE = 16 * 1024 * 1024
    _NUM_DOWNLOAD_ATTEMPTS = 3
    _STAGE_BLOCK_SIZE = 32 * 1024 * 1024
    _RETRYABLE_READ_EXCEPTIONS = (
        azure.core.exceptions.IncompleteReadError,
        azure.core.exceptions.HttpResponseError,
        azure.core.exceptions.DecodeError,
    )
    _QS_PARAMETERS_TO_INCLUDE = [
        "snapshot",
        "versionid",
    ]

    def __init__(
        self,
        sdk_blob_client: azure.storage.blob.BlobClient,
        executor: Optional[concurrent.futures.Executor] = None,
        max_in_flight_requests: Optional[int] = None,
    ):
        self._sdk_blob_client = sdk_blob_client
        self._generated_sdk_storage_client = self._sdk_blob_client._client

        if max_in_flight_requests is None:
            max_in_flight_requests = self._get_max_in_flight_requests()
        self._max_in_flight_requests = max_in_flight_requests
        self._executor = executor
        self._blob_properties: Optional[azure.storage.blob.BlobProperties] = None

    @property
    def url(self) -> str:
        blob_sdk_url = self._sdk_blob_client.url
        parsed_url = urllib.parse.urlparse(blob_sdk_url)
        if parsed_url.query is None:
            return blob_sdk_url
        return self._get_url_without_query_string(parsed_url)

    @property
    def blob_name(self) -> str:
        return self._sdk_blob_client.blob_name  # type: ignore[attr-defined]

    @property
    def container_name(self) -> str:
        return self._sdk_blob_client.container_name  # type: ignore[attr-defined]

    def get_blob_size(self) -> int:
        return self._get_blob_properties().size

    def download(self, offset: int = 0, length: Optional[int] = None) -> bytes:
        initial_content = b""
        if self._blob_properties is None:
            initial_content = self._download_from_unknown_blob_size(offset, length)
            offset = len(initial_content) + offset
            if length is not None:
                length = length - len(initial_content)
            if not self._more_to_download(offset, length):
                return initial_content
        length = self._update_download_length_from_blob_size(offset, length)
        if length < self._PARTITIONED_DOWNLOAD_THRESHOLD:
            return b"".join(
                [initial_content, self._download_with_retries(offset, length)]
            )
        else:
            return b"".join(
                [initial_content, self._partitioned_download(offset, length)]
            )

    def stage_blocks(
        self, data: SUPPORTED_WRITE_BYTES_LIKE_TYPE
    ) -> List[STAGE_BLOCK_FUTURE_TYPE]:
        if not data:
            raise ValueError("Data must not be empty.")
        stage_block_partitions = self._get_stage_block_partitions(data)
        futures = []
        for pos, length in stage_block_partitions:
            self._max_in_flight_semaphore.acquire()
            future = self._get_executor().submit(
                self._stage_block, data[pos : pos + length]
            )
            future.add_done_callback(self._release_in_flight_semaphore)
            futures.append(future)
        return futures

    def commit_block_list(self, block_ids: List[str]) -> None:
        blob_blocks = [azure.storage.blob.BlobBlock(block_id) for block_id in block_ids]
        self._sdk_blob_client.commit_block_list(blob_blocks)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()

    def _get_max_in_flight_requests(self) -> int:
        # Ideally we would just match this value to the max workers of the executor. However
        # the executor class does not publicly expose its max worker count. So, instead we copy
        # the max worker calculation from the executor class and inject it into both the executor
        # and semaphore
        #
        # In Python 3.13, os.process_cpu_count() was added and the ThreadPoolExecutor updated to
        # use os.process_cpu_count() instead of os.cpu_count() when calculating default max workers.
        # To match ThreadPoolExecutor defaults across Python versions, we use process_cpu_count
        # if available, otherwise fall back to os.cpu_count().
        cpu_count_fn = getattr(os, "process_cpu_count", os.cpu_count)
        return min(32, (cpu_count_fn() or 1) + 4)

    def _get_executor(self) -> concurrent.futures.Executor:
        # We want executor creation to be lazy instead of instantiating immediately in
        # the constructor because the executor itself is not pickleable. This is an issue
        # when workers are used by PyTorch's DataLoader as workers are spawned as processes.
        # So we delay executor creation until it is needed for reading/writing data which will
        # happen after processes are spawned.
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                self._max_in_flight_requests
            )
        return self._executor

    @functools.cached_property
    def _max_in_flight_semaphore(self) -> threading.Semaphore:
        # The standard thread pool executor does not bound the number of tasks submitted to it.
        # This semaphore introduces bound so that the number of submitted, in-progress
        # futures are not greater than the available workers. This is important for cases where we
        # buffer data into memory for uploads as is prevents large amounts of memory from being
        # submitted to the executor when there are no workers available to upload it.
        return threading.Semaphore(self._max_in_flight_requests)

    def _get_blob_properties(self) -> azure.storage.blob.BlobProperties:
        if self._blob_properties is None:
            self._blob_properties = self._sdk_blob_client.get_blob_properties()
        return self._blob_properties

    def _update_download_length_from_blob_size(
        self, offset: int, length: Optional[int] = None
    ) -> int:
        length_from_offset = self.get_blob_size() - offset
        if length is not None:
            return min(length, length_from_offset)
        return length_from_offset

    def _partitioned_download(self, offset: int, length: int) -> bytes:
        futures = []
        for read_partition in self._get_partitions(
            offset, length, self._PARTITION_SIZE
        ):
            futures.append(
                self._get_executor().submit(
                    self._download_with_retries, *read_partition
                )
            )
        return b"".join(f.result() for f in futures)

    def _get_partitions(
        self, offset: int, length: int, partition_size: int
    ) -> List[Tuple[int, int]]:
        end = offset + length
        num_partitions = math.ceil(length / partition_size)
        partitions = []
        for i in range(num_partitions):
            start = offset + i * partition_size
            if start >= end:
                break
            size = min(partition_size, end - start)
            partitions.append((start, size))
        return partitions

    def _more_to_download(
        self, updated_offset, remaining_length: Optional[int] = None
    ) -> bool:
        if self._blob_properties and self._blob_properties.size <= updated_offset:
            return False
        if remaining_length is not None and remaining_length == 0:
            return False
        return True

    def _download_from_unknown_blob_size(
        self, offset: int, length: Optional[int] = None
    ) -> bytes:
        if length is None or length > self._PARTITIONED_DOWNLOAD_THRESHOLD:
            length = self._PARTITIONED_DOWNLOAD_THRESHOLD
        return self._download_with_retries(offset, length)

    def _download_with_retries(self, pos: int, length: int) -> bytes:
        attempt = 0
        while self._attempts_remaining(attempt):
            stream = self._get_download_stream(pos, length)
            try:
                return self._read_stream(stream)
            except self._RETRYABLE_READ_EXCEPTIONS:
                backoff_time = self._get_backoff_time(attempt)
                attempt += 1
                if not self._attempts_remaining(attempt):
                    raise
                _LOGGER.debug(
                    "Sleeping %s seconds and retrying download from caught streaming exception (attempts remaining: %s).",
                    backoff_time,
                    self._attempts_remaining(attempt),
                    exc_info=True,
                )
                time.sleep(backoff_time)
        raise RuntimeError("Exhausted all retry attempts to read blob content.")

    def _set_blob_properties_from_download(self, response) -> None:
        headers = response.response.headers
        blob_size = self._get_size_from_range(headers["Content-Range"])
        self._blob_properties = azure.storage.blob.BlobProperties(
            **{"Content-Length": blob_size, "ETag": headers.get("ETag")}
        )

    def _get_download_stream(self, pos: int, length: int) -> Iterator[bytes]:
        try:
            download_kwargs: DownloadKwargsType = {
                "range": f"bytes={pos}-{pos + length - 1}",
            }
            if self._blob_properties is not None:
                download_kwargs["modified_access_conditions"] = (
                    azure.storage.blob._generated.models.ModifiedAccessConditions(
                        if_match=self._blob_properties.etag
                    )
                )
            response = self._generated_sdk_storage_client.blob.download(
                **download_kwargs
            )
            if self._blob_properties is None:
                self._set_blob_properties_from_download(response)
            return response
        except azure.core.exceptions.HttpResponseError as e:
            if self._is_invalid_range_from_empty_blob_error(e):
                self._blob_properties = azure.storage.blob.BlobProperties(
                    **{"Content-Length": 0}
                )
                return iter([b""])
            # TODO: This is so that we properly map exceptions from the generated client to the correct
            # exception class and error code. In the future, prior to a GA, we should consider pulling
            # in this function or a derivative of it if we plan to continue to raise Azure Python SDK
            # exceptions from this library (i.e. instead of raising our own exception classes).
            process_storage_error(e)

    def _get_size_from_range(self, range_header: str) -> int:
        return int(range_header.split(" ", 1)[1].split("/", 1)[1])

    def _is_invalid_range_from_empty_blob_error(
        self, error: azure.core.exceptions.HttpResponseError
    ) -> bool:
        return (
            error.response is not None
            and error.status_code == 416
            and hasattr(error.response, "headers")
            and "Content-Range" in error.response.headers
            and self._get_size_from_range(error.response.headers["Content-Range"]) == 0
        )

    def _attempts_remaining(self, attempt_number: int) -> int:
        return max(self._NUM_DOWNLOAD_ATTEMPTS - attempt_number, 0)

    def _get_backoff_time(self, attempt_number: int) -> float:
        # Backoff time uses exponential backoff with full jitter as a starting point to have at least
        # some delay before retrying. For exceptions that we get while streaming data, it will likely be
        # because of environment's network (e.g. high network load) so the approach will give some amount
        # of backoff and randomness before attempting to stream again. In the future, we should
        # consider other approaches such as adapting/throttling stream reading speeds to reduce occurrences
        # of connection errors due to an overwhelmed network.
        return min(random.uniform(0, 2**attempt_number), 20)

    def _read_stream(self, stream: Iterator[bytes]) -> bytes:
        content = io.BytesIO()
        for chunk in stream:
            content.write(chunk)
        return content.getvalue()

    def _get_stage_block_partitions(
        self, data: SUPPORTED_WRITE_BYTES_LIKE_TYPE
    ) -> List[Tuple[int, int]]:
        return self._get_partitions(0, len(data), self._STAGE_BLOCK_SIZE)

    def _stage_block(self, data: SUPPORTED_WRITE_BYTES_LIKE_TYPE) -> str:
        block_id = str(uuid.uuid4())
        self._sdk_blob_client.stage_block(block_id, data)
        return block_id

    def _release_in_flight_semaphore(self, _: STAGE_BLOCK_FUTURE_TYPE) -> None:
        self._max_in_flight_semaphore.release()

    def _get_url_without_query_string(
        self, parsed_url: urllib.parse.ParseResult
    ) -> str:
        # Helper method to only include scheme, network location, and path for a blob URL.
        # More specifically, we do not want to return any SAS tokens in the URL as it can
        # accidentally result in leaking credentials as part of interfaces that expose the
        # URL (e.g., azstoragetorch.datasets.Blob) so we just remove all URL components past
        # the path.
        return urllib.parse.urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                None,
                None,
                None,
            )
        )
