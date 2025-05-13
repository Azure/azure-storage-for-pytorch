# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import concurrent.futures
import copy
from unittest import mock
import os
import threading
import urllib.parse
import pytest

from azure.core.credentials import AzureSasCredential, AzureNamedKeyCredential
import azure.core.exceptions
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobClient,
    BlobProperties,
    StorageErrorCode,
    BlobBlock,
    ContainerClient,
)
from azure.storage.blob._generated._azure_blob_storage import AzureBlobStorage
from azure.storage.blob._generated.operations import BlobOperations
from azure.storage.blob._generated.models import ModifiedAccessConditions
from azure.core.pipeline import Pipeline
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.transport import RequestsTransport

from azstoragetorch._client import (
    AzStorageTorchBlobClient,
    AzStorageTorchBlobClientFactory,
    EchoClientRequestIdPolicy,
)
from azstoragetorch.exceptions import ClientRequestIdMismatchError
from tests.unit.utils import random_bytes
from azstoragetorch._version import __version__
from azure.core.pipeline.transport._requests_basic import StreamDownloadGenerator

MB = 1024 * 1024
DEFAULT_PARTITION_DOWNLOAD_THRESHOLD = 16 * MB
DEFAULT_PARTITION_SIZE = 16 * MB
DEFAULT_BLOCK_SIZE = 32 * MB
EXPECTED_RETRYABLE_READ_EXCEPTIONS = [
    azure.core.exceptions.IncompleteReadError,
    azure.core.exceptions.HttpResponseError,
    azure.core.exceptions.DecodeError,
]
PROCESS_CPU_COUNT_UNAVAILABLE = object()
SAS_TOKEN = "sp=r&st=2024-10-28T20:22:30Z&se=2024-10-29T04:22:30Z&spr=https&sv=2022-11-02&sr=c&sig=signature"
SNAPSHOT = "2024-10-28T20:34:36.1724588Z"
VERSION_ID = SNAPSHOT


@pytest.fixture(autouse=True)
def sleep_patch():
    with mock.patch("time.sleep") as patched_sleep:
        yield patched_sleep


@pytest.fixture
def sas_token():
    return SAS_TOKEN


@pytest.fixture
def blob_etag():
    return "blob-etag"


@pytest.fixture
def blob_properties(blob_length, blob_etag):
    return BlobProperties(**{"Content-Length": blob_length, "ETag": blob_etag})


@pytest.fixture
def blob_names():
    return ["blob1", "blob2", "blob3"]


@pytest.fixture
def mock_pipeline():
    return mock.Mock(Pipeline)


@pytest.fixture
def mock_generated_sdk_storage_client():
    mock_generated_sdk_client = mock.Mock(AzureBlobStorage)
    mock_generated_sdk_client.blob = mock.Mock(BlobOperations)
    return mock_generated_sdk_client


@pytest.fixture
def mock_sdk_blob_client(
    mock_generated_sdk_storage_client, mock_pipeline, blob_properties, blob_url
):
    mock_sdk_client = mock.Mock(BlobClient)
    mock_sdk_client._client = mock_generated_sdk_storage_client
    mock_sdk_client._pipeline = mock_pipeline
    mock_sdk_client.get_blob_properties.return_value = blob_properties
    mock_sdk_client.url = blob_url
    return mock_sdk_client


@pytest.fixture
def mock_sdk_container_client(
    blob_names, mock_sdk_blob_clients_from_blob_names, mock_pipeline
):
    mock_container_client = mock.Mock(ContainerClient)
    mock_container_client.list_blob_names.return_value = blob_names
    mock_container_client.get_blob_client.side_effect = (
        mock_sdk_blob_clients_from_blob_names
    )
    mock_container_client._pipeline = mock_pipeline
    return mock_container_client


@pytest.fixture
def mock_sdk_blob_clients_from_blob_names(
    mock_sdk_blob_client, blob_names, container_url
):
    mock_clients = []
    for blob_name in blob_names:
        mock_client = copy.copy(mock_sdk_blob_client)
        mock_client.url = f"{container_url}/{blob_name}"
        mock_clients.append(mock_client)
    return mock_clients


@pytest.fixture
def single_threaded_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def azstoragetorch_blob_client(mock_sdk_blob_client, single_threaded_executor):
    return AzStorageTorchBlobClient(
        mock_sdk_blob_client, executor=single_threaded_executor
    )


@pytest.fixture
def http_response_error(blob_url):
    mock_http_response = mock.Mock(HttpResponse)
    mock_http_response.reason = "message"
    mock_http_response.status_code = 400
    mock_http_response.headers = {}
    mock_http_response.content_type = "application/xml"
    mock_http_response.text.return_value = ""
    return azure.core.exceptions.HttpResponseError(response=mock_http_response)


@pytest.fixture
def mock_uuid4():
    with mock.patch("uuid.uuid4") as mock_uuid:
        yield mock_uuid


@pytest.fixture
def echo_client_request_id_policy():
    return EchoClientRequestIdPolicy()


@pytest.fixture
def mock_pipeline_request():
    pipeline_request = mock.Mock(PipelineRequest)
    pipeline_request.http_request = mock.Mock(HttpRequest)
    pipeline_request.http_request.headers = {}
    return pipeline_request


@pytest.fixture
def mock_pipeline_response():
    pipeline_response = mock.Mock(PipelineResponse)
    pipeline_response.http_response = mock.Mock(HttpResponse)
    pipeline_response.http_response.headers = {
        "x-ms-request-id": "request-id",
    }
    return pipeline_response


def slice_bytes(content, range_value):
    start, end = range_value.split("-", 1)
    return content[int(start) : int(end) + 1]


def to_bytes_iterator(content, chunk_size=64 * 1024, exception_to_raise=None):
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]
        if exception_to_raise is not None:
            raise exception_to_raise


def mock_download_response(
    expected_range, blob_length, content, exception=None, etag=None
):
    response = mock.MagicMock(spec=StreamDownloadGenerator)
    response.response = mock.Mock()
    response.response.headers = {
        "Content-Range": f"bytes {expected_range}/{blob_length}",
        "ETag": etag,
    }
    response.iter_content_func = to_bytes_iterator(
        content=slice_bytes(content, expected_range), exception_to_raise=exception
    )
    response.__iter__.return_value = response.iter_content_func
    return response


def preset_blob_size_on_clients(
    azstoragetorch_blob_client,
    mock_sdk_blob_client,
    blob_properties,
):
    mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
    azstoragetorch_blob_client.get_blob_size()


class NonRetryableException(Exception):
    pass


class AtomicCounter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value

    def decrement(self):
        with self._lock:
            self.value -= 1
            return self.value


class SpySubmitExcecutor(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = AtomicCounter()

    def submit(self, fn, *args, **kwargs):
        self.counter.increment()
        return super().submit(fn, *args, **kwargs)


class TestEchoClientRequestIdPolicy:
    def test_adds_client_request_id(
        self, echo_client_request_id_policy, mock_pipeline_request, mock_uuid4
    ):
        mock_uuid4.return_value = "unique-id"
        echo_client_request_id_policy.on_request(mock_pipeline_request)
        mock_uuid4.assert_called_once()
        assert (
            mock_pipeline_request.http_request.headers["x-ms-client-request-id"]
            == "unique-id"
        )

    def test_throws_if_client_request_id_mismatch(
        self,
        echo_client_request_id_policy,
        mock_pipeline_request,
        mock_pipeline_response,
    ):
        mock_pipeline_request.http_request.headers["x-ms-client-request-id"] = (
            "unique-id"
        )
        mock_pipeline_response.http_response.headers["x-ms-client-request-id"] = (
            "does-not-match"
        )
        with pytest.raises(ClientRequestIdMismatchError):
            echo_client_request_id_policy.on_response(
                mock_pipeline_request, mock_pipeline_response
            )

    def test_does_not_raise_if_client_request_id_matches(
        self,
        echo_client_request_id_policy,
        mock_pipeline_request,
        mock_pipeline_response,
    ):
        mock_pipeline_request.http_request.headers["x-ms-client-request-id"] = (
            "unique-id"
        )
        mock_pipeline_response.http_response.headers["x-ms-client-request-id"] = (
            "unique-id"
        )
        try:
            echo_client_request_id_policy.on_response(
                mock_pipeline_request, mock_pipeline_response
            )
        except ClientRequestIdMismatchError as e:
            pytest.fail(f"Client request ID should match but received: {e}")


class TestAzStorageTorchBlobClientFactory:
    @pytest.fixture(autouse=True)
    def sdk_blob_client_patch(self, mock_sdk_blob_client):
        with mock.patch("azure.storage.blob.BlobClient", mock_sdk_blob_client):
            mock_sdk_blob_client.from_blob_url.return_value = mock_sdk_blob_client
            yield mock_sdk_blob_client

    @pytest.fixture(autouse=True)
    def sdk_container_client_patch(self, mock_sdk_container_client):
        with mock.patch(
            "azure.storage.blob.ContainerClient", mock_sdk_container_client
        ):
            mock_sdk_container_client.from_container_url.return_value = (
                mock_sdk_container_client
            )
            yield mock_sdk_container_client

    @pytest.fixture
    def azstoragetorch_blob_client_cls_patch(self):
        with mock.patch(
            "azstoragetorch._client.AzStorageTorchBlobClient", spec=True
        ) as mock_azstorage_blob_client_cls:
            yield mock_azstorage_blob_client_cls

    def assert_expected_from_blob_url_call(self, mock_sdk_blob_client, **kwargs):
        self.assert_expected_from_url_call(mock_sdk_blob_client.from_blob_url, **kwargs)

    def assert_expected_from_container_url_call(
        self, mock_sdk_container_client, **kwargs
    ):
        self.assert_expected_from_url_call(
            mock_sdk_container_client.from_container_url,
            expected_transport=None,
            **kwargs,
        )

    def assert_expected_from_url_call(
        self,
        mock_from_url_method,
        expected_url,
        **kwargs,
    ):
        mock_from_url_method.assert_called_once_with(
            expected_url, **self.get_expected_from_url_kwargs(**kwargs)
        )

    def assert_expected_sdk_blob_clients_from_container_url(
        self,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        expected_url,
        expected_blob_names_used,
        expected_prefix=None,
        expected_credential=mock.ANY,
        expected_credentials_cls=None,
        expected_sdk_blob_client_pipeline=mock.ANY,
    ):
        self.assert_expected_from_container_url_call(
            mock_sdk_container_client,
            expected_url=expected_url,
            expected_credential=expected_credential,
        )
        if expected_credentials_cls is not None:
            assert isinstance(
                self.get_container_credential_used(mock_sdk_container_client),
                expected_credentials_cls,
            )
        mock_sdk_container_client.list_blob_names.assert_called_once_with(
            name_starts_with=expected_prefix
        )
        assert mock_sdk_container_client.get_blob_client.call_args_list == [
            mock.call(blob_name) for blob_name in expected_blob_names_used
        ]
        self.assert_expected_from_blob_url_calls(
            mock_sdk_blob_client,
            expected_container_url=expected_url,
            expected_blob_names_used=expected_blob_names_used,
            expected_credential=expected_credential,
            expected_pipeline=expected_sdk_blob_client_pipeline,
        )

    def assert_expected_underlying_sdk_blob_clients(
        self,
        mock_azstorage_blob_client_cls,
        expected_blob_sdk_clients,
    ):
        assert mock_azstorage_blob_client_cls.call_args_list == [
            mock.call(sdk_blob_client) for sdk_blob_client in expected_blob_sdk_clients
        ]

    def assert_expected_from_blob_url_calls(
        self,
        mock_sdk_blob_client,
        expected_container_url,
        expected_blob_names_used,
        expected_credential=mock.ANY,
        expected_pipeline=mock.ANY,
    ):
        expected_calls = []
        container_url_parts = urllib.parse.urlparse(expected_container_url)
        for i, expected_blob_name in enumerate(expected_blob_names_used):
            expected_blob_url = urllib.parse.urlunparse(
                (
                    container_url_parts.scheme,
                    container_url_parts.netloc,
                    f"{container_url_parts.path}/{expected_blob_name}",
                    container_url_parts.params,
                    container_url_parts.query,
                    container_url_parts.fragment,
                )
            )
            expected_from_url_kwargs = self.get_expected_from_url_kwargs(
                expected_credential=expected_credential,
            )
            if expected_pipeline and i != 0:
                # For the blob clients, pipelines should be shared. So the first blob
                # client created will have its pipeline cached and reused for
                # subsequent blob clients.
                expected_from_url_kwargs["_pipeline"] = expected_pipeline
            expected_calls.append(
                mock.call(expected_blob_url, **expected_from_url_kwargs)
            )
        assert mock_sdk_blob_client.from_blob_url.call_args_list == expected_calls

    def get_expected_from_url_kwargs(
        self,
        expected_credential=mock.ANY,
        expected_transport=mock.ANY,
        expected_pipeline=None,
    ):
        expected_from_url_kwargs = {
            "credential": expected_credential,
            "user_agent": f"azstoragetorch/{__version__}",
            "_additional_pipeline_policies": [
                mock.ANY,
            ],
        }
        if expected_transport is not None:
            expected_from_url_kwargs["transport"] = expected_transport
        if expected_pipeline is not None:
            expected_from_url_kwargs["_pipeline"] = expected_pipeline
        return expected_from_url_kwargs

    def get_credential_used(self, mock_sdk_blob_client):
        return self.get_sdk_kwarg_used(mock_sdk_blob_client, "credential")

    def get_container_credential_used(self, mock_sdk_container_client):
        return self.get_sdk_kwarg_used(
            mock_sdk_container_client,
            "credential",
            from_url_method_name="from_container_url",
        )

    def get_transport_used(self, mock_sdk_blob_client):
        return self.get_sdk_kwarg_used(mock_sdk_blob_client, "transport")

    def get_additional_pipeline_policies_used(self, mock_sdk_blob_client):
        return self.get_sdk_kwarg_used(
            mock_sdk_blob_client, "_additional_pipeline_policies"
        )

    def get_sdk_kwarg_used(
        self, mock_sdk_client, kwarg_name, from_url_method_name="from_blob_url"
    ):
        mock_from_url_method = getattr(mock_sdk_client, from_url_method_name)
        return mock_from_url_method.call_args[1][kwarg_name]

    def get_mock_azstoragetorch_blob_clients(self, blob_names):
        return [mock.Mock(AzStorageTorchBlobClient) for _ in blob_names]

    def test_get_blob_client_from_url(
        self, blob_url, mock_sdk_blob_client, azstoragetorch_blob_client_cls_patch
    ):
        factory = AzStorageTorchBlobClientFactory()
        returned_client = factory.get_blob_client_from_url(blob_url)
        assert returned_client is azstoragetorch_blob_client_cls_patch.return_value
        azstoragetorch_blob_client_cls_patch.assert_called_once_with(
            mock_sdk_blob_client.from_blob_url.return_value
        )
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client, expected_url=blob_url
        )

    def test_credential_defaults_to_azure_default_credential(
        self, blob_url, mock_sdk_blob_client
    ):
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(blob_url)
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client, expected_url=blob_url
        )
        assert isinstance(
            self.get_credential_used(mock_sdk_blob_client), DefaultAzureCredential
        )

    @pytest.mark.parametrize(
        "credential",
        [
            DefaultAzureCredential(),
            AzureSasCredential("sas"),
        ],
    )
    def test_respects_user_provided_credential(
        self, blob_url, mock_sdk_blob_client, credential
    ):
        factory = AzStorageTorchBlobClientFactory(credential=credential)
        factory.get_blob_client_from_url(blob_url)
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client, expected_url=blob_url, expected_credential=credential
        )

    def test_anonymous_credential(self, blob_url, mock_sdk_blob_client):
        factory = AzStorageTorchBlobClientFactory(credential=False)
        factory.get_blob_client_from_url(blob_url)
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client, expected_url=blob_url, expected_credential=None
        )

    def test_detects_sas_token_in_blob_url(
        self, blob_url, mock_sdk_blob_client, sas_token
    ):
        blob_url_with_sas = f"{blob_url}?{sas_token}"
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(blob_url_with_sas)
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client,
            expected_url=blob_url_with_sas,
            expected_credential=None,
        )

    def test_sas_token_in_blob_url_overrides_credential(
        self, blob_url, mock_sdk_blob_client, sas_token
    ):
        blob_url_with_sas = f"{blob_url}?{sas_token}"
        factory = AzStorageTorchBlobClientFactory(credential=DefaultAzureCredential())
        factory.get_blob_client_from_url(blob_url_with_sas)
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client,
            expected_url=blob_url_with_sas,
            expected_credential=None,
        )

    def test_credential_defaults_to_azure_default_credential_for_snapshot_url(
        self, blob_url, mock_sdk_blob_client
    ):
        snapshot_url = f"{blob_url}?snapshot={SNAPSHOT}"
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(snapshot_url)
        self.assert_expected_from_blob_url_call(
            mock_sdk_blob_client, expected_url=snapshot_url
        )
        assert isinstance(
            self.get_credential_used(mock_sdk_blob_client), DefaultAzureCredential
        )

    @pytest.mark.parametrize(
        "credential",
        [
            "key",
            {"account_name": "name", "account_key": "key"},
            AzureNamedKeyCredential("name", "key"),
        ],
    )
    def test_raises_for_unsupported_credential(self, credential):
        with pytest.raises(TypeError, match="Unsupported credential"):
            AzStorageTorchBlobClientFactory(credential=credential)

    def test_reuses_credential(self, blob_url, mock_sdk_blob_client):
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(blob_url)
        first_credential_used = self.get_credential_used(mock_sdk_blob_client)
        assert isinstance(first_credential_used, DefaultAzureCredential)

        factory.get_blob_client_from_url(blob_url)
        assert mock_sdk_blob_client.from_blob_url.call_count == 2
        second_credential_used = self.get_credential_used(mock_sdk_blob_client)
        assert first_credential_used is second_credential_used

    def test_transport_defaults(self, blob_url, mock_sdk_blob_client):
        with mock.patch(
            "azstoragetorch._client.RequestsTransport", spec=True
        ) as mock_requests_transport_cls:
            factory = AzStorageTorchBlobClientFactory()
            factory.get_blob_client_from_url(blob_url)
            self.assert_expected_from_blob_url_call(
                mock_sdk_blob_client,
                expected_url=blob_url,
                expected_transport=mock_requests_transport_cls.return_value,
            )
            mock_requests_transport_cls.assert_called_once_with(
                connection_timeout=20,
                read_timeout=60,
                connection_data_block_size=256 * 1024,
            )

    def test_reuses_transport(self, blob_url, mock_sdk_blob_client):
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(blob_url)
        first_transport_used = self.get_transport_used(mock_sdk_blob_client)
        assert isinstance(first_transport_used, RequestsTransport)

        factory.get_blob_client_from_url(blob_url)
        assert mock_sdk_blob_client.from_blob_url.call_count == 2
        second_transport_used = self.get_transport_used(mock_sdk_blob_client)
        assert first_transport_used is second_transport_used

    def test_reuses_pipeline(self, blob_url, mock_sdk_blob_client, mock_pipeline):
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(blob_url)
        factory.get_blob_client_from_url(blob_url)

        assert mock_sdk_blob_client.from_blob_url.call_args_list == [
            mock.call(blob_url, **self.get_expected_from_url_kwargs()),
            mock.call(
                blob_url,
                **self.get_expected_from_url_kwargs(expected_pipeline=mock_pipeline),
            ),
        ]

    def test_does_not_reuse_pipeline_if_sas_used(
        self, blob_url, sas_token, mock_sdk_blob_client, mock_pipeline
    ):
        factory = AzStorageTorchBlobClientFactory()
        blob_url_with_sas = f"{blob_url}?{sas_token}"
        factory.get_blob_client_from_url(blob_url_with_sas)
        factory.get_blob_client_from_url(blob_url)
        factory.get_blob_client_from_url(blob_url)
        factory.get_blob_client_from_url(blob_url_with_sas)

        assert mock_sdk_blob_client.from_blob_url.call_args_list == [
            # No pipeline provided or cached for first client since it uses SAS.
            mock.call(blob_url_with_sas, **self.get_expected_from_url_kwargs()),
            # This client's pipeline should be cached since it uses the factory's credentials
            mock.call(blob_url, **self.get_expected_from_url_kwargs()),
            # Next non-SAS token client should now use cached pipeline
            mock.call(
                blob_url,
                **self.get_expected_from_url_kwargs(expected_pipeline=mock_pipeline),
            ),
            # Should not use the cached pipeline since it uses a SAS token
            mock.call(blob_url_with_sas, **self.get_expected_from_url_kwargs()),
        ]

    def test_does_not_reuse_pipeline_for_container_clients(
        self,
        container_url,
        blob_url,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        mock_pipeline,
        blob_name,
        blob_names,
        mock_sdk_blob_clients_from_blob_names,
    ):
        factory = AzStorageTorchBlobClientFactory()
        mock_sdk_container_client.get_blob_client.side_effect = (
            mock_sdk_blob_clients_from_blob_names * 2
        )

        list(factory.yield_blob_clients_from_container_url(container_url))
        factory.get_blob_client_from_url(blob_url)
        list(factory.yield_blob_clients_from_container_url(container_url))

        # None of the container clients created should use a shared pipeline
        # nor a shared transport.
        expected_from_container_url_kwargs = self.get_expected_from_url_kwargs(
            expected_transport=None,
            expected_pipeline=None,
        )
        assert mock_sdk_container_client.from_container_url.call_args_list == [
            mock.call(container_url, **expected_from_container_url_kwargs),
            mock.call(container_url, **expected_from_container_url_kwargs),
        ]

        # For the blob clients, we still want to share pipelines so we will cache the
        # pipeline of the first client and reuse for the rest of them including the clients
        # created from a container client
        expected_blob_names = blob_names + [blob_name] + blob_names
        self.assert_expected_from_blob_url_calls(
            mock_sdk_blob_client,
            expected_container_url=container_url,
            expected_blob_names_used=expected_blob_names,
            expected_pipeline=mock_pipeline,
        )

    def test_injects_echo_client_request_id_policy(
        self, blob_url, mock_sdk_blob_client
    ):
        factory = AzStorageTorchBlobClientFactory()
        factory.get_blob_client_from_url(blob_url)
        additional_policies = self.get_additional_pipeline_policies_used(
            mock_sdk_blob_client
        )
        assert len(additional_policies) == 1
        assert isinstance(additional_policies[0], EchoClientRequestIdPolicy)

    def test_yield_blob_clients_from_container_url(
        self,
        container_url,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        azstoragetorch_blob_client_cls_patch,
        blob_names,
        mock_sdk_blob_clients_from_blob_names,
    ):
        factory = AzStorageTorchBlobClientFactory()
        mock_sdk_blob_client.from_blob_url.side_effect = (
            mock_sdk_blob_clients_from_blob_names
        )
        mock_azstorage_blob_clients = self.get_mock_azstoragetorch_blob_clients(
            blob_names
        )
        azstoragetorch_blob_client_cls_patch.side_effect = mock_azstorage_blob_clients
        blob_clients = list(
            factory.yield_blob_clients_from_container_url(container_url)
        )
        assert blob_clients == mock_azstorage_blob_clients
        self.assert_expected_sdk_blob_clients_from_container_url(
            mock_sdk_container_client,
            mock_sdk_blob_client,
            expected_url=container_url,
            expected_blob_names_used=blob_names,
            expected_prefix=None,
            expected_credentials_cls=DefaultAzureCredential,
        )
        self.assert_expected_underlying_sdk_blob_clients(
            azstoragetorch_blob_client_cls_patch,
            expected_blob_sdk_clients=mock_sdk_blob_clients_from_blob_names,
        )

    def test_yield_blob_clients_from_container_url_with_prefix(
        self,
        container_url,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        blob_names,
    ):
        factory = AzStorageTorchBlobClientFactory()
        list(
            factory.yield_blob_clients_from_container_url(
                container_url, prefix="prefix"
            )
        )
        self.assert_expected_sdk_blob_clients_from_container_url(
            mock_sdk_container_client,
            mock_sdk_blob_client,
            expected_url=container_url,
            expected_blob_names_used=blob_names,
            expected_prefix="prefix",
        )

    def test_yield_blob_clients_from_container_url_with_credential(
        self,
        container_url,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        blob_names,
    ):
        credential = AzureSasCredential("sas")
        factory = AzStorageTorchBlobClientFactory(credential=credential)
        list(factory.yield_blob_clients_from_container_url(container_url))
        self.assert_expected_sdk_blob_clients_from_container_url(
            mock_sdk_container_client,
            mock_sdk_blob_client,
            expected_url=container_url,
            expected_blob_names_used=blob_names,
            expected_credential=credential,
        )

    def test_yield_blob_clients_from_container_url_with_anonymous_credential(
        self,
        container_url,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        blob_names,
    ):
        factory = AzStorageTorchBlobClientFactory(credential=False)
        list(factory.yield_blob_clients_from_container_url(container_url))
        self.assert_expected_sdk_blob_clients_from_container_url(
            mock_sdk_container_client,
            mock_sdk_blob_client,
            expected_url=container_url,
            expected_blob_names_used=blob_names,
            expected_credential=None,
        )

    def test_yield_blob_clients_from_container_url_with_sas_in_url(
        self,
        container_url,
        mock_sdk_container_client,
        mock_sdk_blob_client,
        blob_names,
        mock_sdk_blob_clients_from_blob_names,
        sas_token,
    ):
        container_url_with_sas = f"{container_url}?{sas_token}"
        for mock_blob_client, blob_name in zip(
            mock_sdk_blob_clients_from_blob_names, blob_names
        ):
            mock_blob_client.url = f"{container_url}/{blob_name}?{sas_token}"
        factory = AzStorageTorchBlobClientFactory()
        list(factory.yield_blob_clients_from_container_url(container_url_with_sas))
        self.assert_expected_sdk_blob_clients_from_container_url(
            mock_sdk_container_client,
            mock_sdk_blob_client,
            expected_url=container_url_with_sas,
            expected_blob_names_used=blob_names,
            expected_credential=None,
            expected_sdk_blob_client_pipeline=None,
        )


class TestAzStorageTorchBlobClient:
    def assert_expected_download_calls(
        self,
        mock_generated_sdk_storage_client,
        expected_ranges,
        expected_etag,
        known_blob_size=False,
    ):
        expected_download_calls = []
        for i, expected_range in enumerate(expected_ranges):
            download_kwargs = {
                "range": f"bytes={expected_range}",
            }
            if i != 0 or known_blob_size:
                download_kwargs["modified_access_conditions"] = (
                    ModifiedAccessConditions(if_match=expected_etag)
                )
            expected_download_calls.append(mock.call(**download_kwargs))
        assert (
            mock_generated_sdk_storage_client.blob.download.call_args_list
            == expected_download_calls
        )

    def assert_stage_block_ids(self, stage_block_futures, expected_block_ids):
        actual_block_ids = [future.result() for future in stage_block_futures]
        assert actual_block_ids == expected_block_ids

    @pytest.mark.parametrize(
        "cpu_count,process_cpu_count,expected_max_workers",
        [
            (1, PROCESS_CPU_COUNT_UNAVAILABLE, 5),
            (4, PROCESS_CPU_COUNT_UNAVAILABLE, 8),
            (16, PROCESS_CPU_COUNT_UNAVAILABLE, 20),
            # cpu_count reaches max_workers ceiling
            (64, PROCESS_CPU_COUNT_UNAVAILABLE, 32),
            # max_workers are still set even if cpu_count is None or 0
            (None, PROCESS_CPU_COUNT_UNAVAILABLE, 5),
            (0, PROCESS_CPU_COUNT_UNAVAILABLE, 5),
            # proccess_cpu_count overrides cpu_count
            (64, 1, 5),
            # process_cpu_count reaches max_workers ceiling
            (1, 64, 32),
            # max_workers are set when both cpu_count and process_cpu_count are None or 0
            (None, None, 5),
            (0, 0, 5),
        ],
    )
    def test_default_worker_count(
        self,
        mock_sdk_blob_client,
        monkeypatch,
        cpu_count,
        process_cpu_count,
        expected_max_workers,
    ):
        if process_cpu_count is PROCESS_CPU_COUNT_UNAVAILABLE:
            monkeypatch.delattr(os, "process_cpu_count", raising=False)
        else:
            monkeypatch.setattr(
                os, "process_cpu_count", lambda: process_cpu_count, raising=False
            )
        monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
        with mock.patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            client = AzStorageTorchBlobClient(mock_sdk_blob_client)
            # Executor instantiation is lazy. Stage some content to instantiate it
            # and determine what the max_workers value is.
            client.stage_blocks(b"content")
            mock_executor.assert_called_once_with(expected_max_workers)

    @pytest.mark.parametrize(
        "sdk_blob_client_url, expected_url",
        [
            (
                "https://account.blob.core.windows.net/container/blob",
                "https://account.blob.core.windows.net/container/blob",
            ),
            # Cases to make sure query string is not included in returned URL.
            (
                f"https://account.blob.core.windows.net/container/blob?{SAS_TOKEN}",
                "https://account.blob.core.windows.net/container/blob",
            ),
            (
                f"https://account.blob.core.windows.net/container/blob?snapshot={SNAPSHOT}&versionid={VERSION_ID}&{SAS_TOKEN}",
                "https://account.blob.core.windows.net/container/blob",
            ),
            (
                "https://account.blob.core.windows.net/container/blob?unknown1=val1&unknown2=val2",
                "https://account.blob.core.windows.net/container/blob",
            ),
        ],
    )
    def test_url(
        self,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        sdk_blob_client_url,
        expected_url,
    ):
        mock_sdk_blob_client.url = sdk_blob_client_url
        assert azstoragetorch_blob_client.url == expected_url

    def test_blob_name(self, azstoragetorch_blob_client, mock_sdk_blob_client):
        mock_sdk_blob_client.blob_name = "blob-name"
        assert azstoragetorch_blob_client.blob_name == "blob-name"

    def test_container_name(self, azstoragetorch_blob_client, mock_sdk_blob_client):
        mock_sdk_blob_client.container_name = "container-name"
        assert azstoragetorch_blob_client.container_name == "container-name"

    def test_get_blob_size(
        self, azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
    ):
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        assert azstoragetorch_blob_client.get_blob_size() == blob_properties.size
        mock_sdk_blob_client.get_blob_properties.assert_called_once_with()

    def test_get_blob_size_caches_result(
        self, azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
    ):
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        assert azstoragetorch_blob_client.get_blob_size() == blob_properties.size
        assert azstoragetorch_blob_client.get_blob_size() == blob_properties.size
        mock_sdk_blob_client.get_blob_properties.assert_called_once_with()

    def test_close(self, mock_sdk_blob_client):
        mock_executor = mock.Mock(concurrent.futures.Executor)
        client = AzStorageTorchBlobClient(mock_sdk_blob_client, mock_executor)
        client.close()
        mock_executor.shutdown.assert_called_once_with()

    def test_no_executor_used_when_no_transfers(self, mock_sdk_blob_client):
        with mock.patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            client = AzStorageTorchBlobClient(mock_sdk_blob_client)
            client.close()
            mock_executor.assert_not_called()
            mock_executor.return_value.shutdown.assert_not_called()

    @pytest.mark.parametrize(
        "blob_size, download_offset, download_length, expected_ranges, known_blob_size",
        [
            # Download an empty blob
            (0, None, None, ["0--1"], True),
            (0, None, None, [f"0-{DEFAULT_PARTITION_SIZE - 1}"], False),
            # Small single GET full, download
            (10, None, None, ["0-9"], True),
            (10, None, None, [f"0-{DEFAULT_PARTITION_SIZE - 1}"], False),
            # Small download with offset
            (10, 5, None, ["5-9"], True),
            (10, 5, None, [f"5-{DEFAULT_PARTITION_SIZE + 5 - 1}"], False),
            # Small download with length
            (10, 0, 5, ["0-4"], True),
            (10, 0, 5, ["0-4"], False),
            # Small download with offset and length
            (10, 3, 4, ["3-6"], True),
            (10, 3, 4, ["3-6"], False),
            # Small download with length past blob size
            (10, 5, 10, ["5-9"], True),
            (10, 5, 10, ["5-14"], False),
            # Small download of portion of large blob
            (32 * MB, 10, 10, ["10-19"], True),
            (32 * MB, 10, 10, ["10-19"], False),
            # Download just below partitioned threshold
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1,
                None,
                None,
                [f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 2}"],
                True,
            ),
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1,
                None,
                None,
                [f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}"],
                False,
            ),
            # Download at partitioned threshold
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD,
                None,
                None,
                [f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}"],
                True,
            ),
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD,
                None,
                None,
                [f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}"],
                False,
            ),
            # Download just above partitioned threshold
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD + 1,
                None,
                None,
                [
                    f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}",
                    f"{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD}-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD}",
                ],
                True,
            ),
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD + 1,
                None,
                None,
                [
                    f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}",
                    f"{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD}-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD}",
                ],
                False,
            ),
            # Large download with multiple partitions
            (
                4 * DEFAULT_PARTITION_SIZE,
                None,
                None,
                [
                    f"0-{DEFAULT_PARTITION_SIZE - 1}",
                    f"{DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{2 * DEFAULT_PARTITION_SIZE}-{3 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{3 * DEFAULT_PARTITION_SIZE}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
                True,
            ),
            (
                4 * DEFAULT_PARTITION_SIZE,
                None,
                None,
                [
                    f"0-{DEFAULT_PARTITION_SIZE - 1}",
                    f"{DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{2 * DEFAULT_PARTITION_SIZE}-{3 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{3 * DEFAULT_PARTITION_SIZE}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
                False,
            ),
            # Large download with offset
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                None,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (3 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (3 * DEFAULT_PARTITION_SIZE)}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
                True,
            ),
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                None,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (3 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (3 * DEFAULT_PARTITION_SIZE)}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
                False,
            ),
            # Large download with length
            (
                4 * DEFAULT_PARTITION_SIZE,
                None,
                2 * DEFAULT_PARTITION_SIZE + 5,
                [
                    f"0-{DEFAULT_PARTITION_SIZE - 1}",
                    f"{DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{2 * DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE + 4}",
                ],
                True,
            ),
            (
                4 * DEFAULT_PARTITION_SIZE,
                None,
                2 * DEFAULT_PARTITION_SIZE + 5,
                [
                    f"0-{DEFAULT_PARTITION_SIZE - 1}",
                    f"{DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{2 * DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE + 4}",
                ],
                False,
            ),
            # Large download with offset and length
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                2 * DEFAULT_PARTITION_SIZE + 5,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (2 * DEFAULT_PARTITION_SIZE + 4)}",
                ],
                True,
            ),
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                2 * DEFAULT_PARTITION_SIZE + 5,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (2 * DEFAULT_PARTITION_SIZE + 4)}",
                ],
                False,
            ),
            # Large download with length past blob size
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                5 * DEFAULT_PARTITION_SIZE,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (3 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (3 * DEFAULT_PARTITION_SIZE)}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
                True,
            ),
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                5 * DEFAULT_PARTITION_SIZE,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (3 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (3 * DEFAULT_PARTITION_SIZE)}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
                False,
            ),
        ],
    )
    def test_download(
        self,
        blob_size,
        download_offset,
        download_length,
        expected_ranges,
        known_blob_size,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
    ):
        blob_properties.size = blob_size
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        if known_blob_size:
            azstoragetorch_blob_client.get_blob_size()
        content = random_bytes(blob_size)
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            mock_download_response(
                expected_range, blob_size, content, etag=blob_properties.etag
            )
            for expected_range in expected_ranges
        ]
        download_kwargs = {}
        expected_download_content = content
        if download_offset is not None:
            download_kwargs["offset"] = download_offset
            expected_download_content = expected_download_content[download_offset:]
        if download_length is not None:
            download_kwargs["length"] = download_length
            expected_download_content = expected_download_content[:download_length]
        assert (
            azstoragetorch_blob_client.download(**download_kwargs)
            == expected_download_content
        )
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            expected_ranges=expected_ranges,
            expected_etag=blob_properties.etag,
            known_blob_size=known_blob_size,
        )

    @pytest.mark.parametrize(
        "response_error_code,expected_sdk_exception,expected_storage_error_code,headers",
        [
            (
                "BlobNotFound",
                azure.core.exceptions.ResourceNotFoundError,
                StorageErrorCode.BLOB_NOT_FOUND,
                {"Content-Range": "bytes 1-4/12"},
            ),
            (
                "ConditionNotMet",
                azure.core.exceptions.ResourceModifiedError,
                StorageErrorCode.CONDITION_NOT_MET,
                {"Content-Range": "bytes 1-4/12"},
            ),
            (
                "InvalidRange",
                azure.core.exceptions.HttpResponseError,
                StorageErrorCode.INVALID_RANGE,
                {"Content-Range": "bytes */12"},
            ),
        ],
    )
    def test_maps_download_exceptions(
        self,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        http_response_error,
        response_error_code,
        expected_sdk_exception,
        expected_storage_error_code,
        headers,
    ):
        http_response_error.response.text.return_value = (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f" <Error><Code>{response_error_code}</Code>"
            f" <Message>message</Message>"
            f"</Error>"
        )

        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        http_response_error.response.headers = headers
        mock_generated_sdk_storage_client.blob.download.side_effect = (
            http_response_error
        )
        with pytest.raises(expected_sdk_exception) as exc_info:
            azstoragetorch_blob_client.download()
        assert exc_info.value.error_code == expected_storage_error_code

    def test_invalid_range_exception_size_zero(
        self,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        http_response_error,
    ):
        http_response_error.response.text.return_value = (
            '<?xml version="1.0" encoding="utf-8"?>'
            " <Error><Code>InvalidRange</Code>"
            " <Message>message</Message>"
            "</Error>"
        )
        blob_properties.size = 0
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        http_response_error.status_code = 416
        http_response_error.response.headers = {"Content-Range": "bytes */0"}
        mock_generated_sdk_storage_client.blob.download.side_effect = (
            http_response_error
        )
        assert azstoragetorch_blob_client.download() == b""
        assert azstoragetorch_blob_client.get_blob_size() == 0

    @pytest.mark.parametrize(
        "retryable_exception_cls", EXPECTED_RETRYABLE_READ_EXCEPTIONS
    )
    def test_retries_reads(
        self,
        retryable_exception_cls,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        sleep_patch,
    ):
        content = random_bytes(10)
        blob_properties.size = len(content)
        download_kwargs = {}
        download_kwargs["length"] = len(content)
        preset_blob_size_on_clients(
            azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
        )
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
            to_bytes_iterator(content),
        ]
        assert azstoragetorch_blob_client.download(**download_kwargs) == content
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            ["0-9", "0-9"],
            blob_properties.etag,
            True,
        )
        assert sleep_patch.call_count == 1

    @pytest.mark.parametrize(
        "retryable_exception_cls", EXPECTED_RETRYABLE_READ_EXCEPTIONS
    )
    def test_raises_after_retries_exhausted(
        self,
        retryable_exception_cls,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        sleep_patch,
    ):
        content = random_bytes(10)
        blob_properties.size = len(content)
        download_kwargs = {}
        download_kwargs["length"] = len(content)
        preset_blob_size_on_clients(
            azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
        )
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
        ]
        with pytest.raises(retryable_exception_cls):
            azstoragetorch_blob_client.download(**download_kwargs)
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            ["0-9", "0-9", "0-9"],
            blob_properties.etag,
            True,
        )
        assert sleep_patch.call_count == 2

    def test_does_not_retry_on_non_retryable_exceptions(
        self,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
    ):
        content = random_bytes(10)
        blob_properties.size = len(content)
        download_kwargs = {}
        download_kwargs["length"] = len(content)
        preset_blob_size_on_clients(
            azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
        )
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(content, exception_to_raise=NonRetryableException()),
        ]
        with pytest.raises(NonRetryableException):
            azstoragetorch_blob_client.download(**download_kwargs)
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            ["0-9"],
            blob_properties.etag,
            True,
        )

    @pytest.mark.parametrize(
        "bytes_like_type",
        [
            bytes,
            bytearray,
            memoryview,
        ],
    )
    @pytest.mark.parametrize(
        "content_length,expected_block_start_ends",
        [
            (1, [(0, 1)]),
            (DEFAULT_BLOCK_SIZE - 1, [(0, DEFAULT_BLOCK_SIZE - 1)]),
            (DEFAULT_BLOCK_SIZE, [(0, DEFAULT_BLOCK_SIZE)]),
            (
                DEFAULT_BLOCK_SIZE + 1,
                [(0, DEFAULT_BLOCK_SIZE), (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE + 1)],
            ),
            (
                DEFAULT_BLOCK_SIZE * 2,
                [(0, DEFAULT_BLOCK_SIZE), (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE * 2)],
            ),
            (
                DEFAULT_BLOCK_SIZE * 2 + 1,
                [
                    (0, DEFAULT_BLOCK_SIZE),
                    (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE * 2),
                    (DEFAULT_BLOCK_SIZE * 2, DEFAULT_BLOCK_SIZE * 2 + 1),
                ],
            ),
            (
                DEFAULT_BLOCK_SIZE * 4,
                [
                    (0, DEFAULT_BLOCK_SIZE),
                    (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE * 2),
                    (DEFAULT_BLOCK_SIZE * 2, DEFAULT_BLOCK_SIZE * 3),
                    (DEFAULT_BLOCK_SIZE * 3, DEFAULT_BLOCK_SIZE * 4),
                ],
            ),
        ],
    )
    def test_stage_blocks(
        self,
        bytes_like_type,
        content_length,
        expected_block_start_ends,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_uuid4,
    ):
        expected_block_ids = [str(i) for i in range(len(expected_block_start_ends))]
        mock_uuid4.side_effect = expected_block_ids
        content = bytes_like_type(random_bytes(content_length))
        expected_stage_block_calls = [
            mock.call(str(i), content[start:end])
            for i, (start, end) in enumerate(expected_block_start_ends)
        ]
        stage_block_futures = azstoragetorch_blob_client.stage_blocks(content)
        self.assert_stage_block_ids(stage_block_futures, expected_block_ids)
        assert (
            mock_sdk_blob_client.stage_block.call_args_list
            == expected_stage_block_calls
        )
        assert mock_uuid4.call_count == len(expected_block_start_ends)

    def test_stage_blocks_returns_error_in_future(
        self, azstoragetorch_blob_client, mock_sdk_blob_client
    ):
        mock_sdk_blob_client.stage_block.side_effect = azure.core.exceptions.AzureError(
            "message"
        )
        stage_block_futures = azstoragetorch_blob_client.stage_blocks(random_bytes(8))
        with pytest.raises(azure.core.exceptions.AzureError):
            stage_block_futures[0].result()

    @pytest.mark.parametrize(
        "empty_content",
        [
            b"",
            bytearray(),
            memoryview(b""),
            memoryview(bytearray()),
        ],
    )
    def test_stage_blocks_raises_on_empty_data(
        self, empty_content, azstoragetorch_blob_client
    ):
        with pytest.raises(ValueError, match="must not be empty"):
            azstoragetorch_blob_client.stage_blocks(empty_content)

    def test_stage_blocks_bounds_submitted_and_in_progress_futures(
        self, mock_sdk_blob_client
    ):
        # To test the in-flight request limit, we flood the client with stage block requests and update a counter
        # as part of the Executor.submit() and stage_block() client call to ensure that th number of queued and
        # in-progress futures is never greater than the specified limit.
        max_in_flight_requests = 5
        spy_submit_executor = SpySubmitExcecutor(max_in_flight_requests)
        client = AzStorageTorchBlobClient(
            mock_sdk_blob_client, spy_submit_executor, max_in_flight_requests
        )

        in_flight_counts = []

        def stage_block_side_effect(*args):
            in_flight_counts.append(spy_submit_executor.counter.decrement())

        mock_sdk_blob_client.stage_block.side_effect = stage_block_side_effect

        content = random_bytes(10)
        for _ in range(2000):
            client.stage_blocks(content)
        client.close()

        assert spy_submit_executor.counter.value == 0
        assert max(in_flight_counts) <= max_in_flight_requests

    @pytest.mark.parametrize(
        "block_list",
        [
            [],
            ["block-1"],
            ["block-1", "block-2"],
        ],
    )
    def test_commit_block_list(
        self, block_list, azstoragetorch_blob_client, mock_sdk_blob_client
    ):
        azstoragetorch_blob_client.commit_block_list(block_list)
        expected_blob_blocks = [BlobBlock(block_id) for block_id in block_list]
        mock_sdk_blob_client.commit_block_list.assert_called_once_with(
            expected_blob_blocks
        )
