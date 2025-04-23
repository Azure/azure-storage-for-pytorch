# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

from collections.abc import Callable, Iterable, Iterator
from typing import Optional, Union, TypedDict
from typing_extensions import Self, TypeVar

import torch.utils.data

from azstoragetorch.io import BlobIO
from azstoragetorch import _client


_TransformOutputType_co = TypeVar(
    "_TransformOutputType_co", covariant=True, default="_DefaultTransformOutput"
)


class _DefaultTransformOutput(TypedDict):
    url: str
    data: bytes


def _default_transform(blob: "Blob") -> _DefaultTransformOutput:
    with blob.reader() as f:
        content = f.read()
    ret: _DefaultTransformOutput = {
        "url": blob.url,
        "data": content,
    }
    return ret


class Blob:
    def __init__(self, blob_client: _client.AzStorageTorchBlobClient):
        self._blob_client = blob_client

    @property
    def url(self) -> str:
        return self._blob_client.url

    @property
    def blob_name(self) -> str:
        return self._blob_client.blob_name

    @property
    def container_name(self) -> str:
        return self._blob_client.container_name

    def reader(self) -> BlobIO:
        return BlobIO(
            self._blob_client.url, "rb", _azstoragetorch_blob_client=self._blob_client
        )


class BlobDataset(torch.utils.data.Dataset[_TransformOutputType_co]):
    def __init__(
        self,
        blobs: Iterable[Blob],
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ):
        self._blobs = list(blobs)
        if transform is None:
            transform = _default_transform
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ) -> Self:
        blobs = _BlobUrlsBlobIterable(blob_urls, credential=credential)
        return cls(blobs, transform=transform)

    @classmethod
    def from_container_url(
        cls,
        container_url: str,
        *,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ) -> Self:
        blobs = _ContainerUrlBlobIterable(
            container_url, prefix=prefix, credential=credential
        )
        return cls(blobs, transform=transform)

    def __getitem__(self, index: int) -> _TransformOutputType_co:
        blob = self._blobs[index]
        return self._transform(blob)

    def __len__(self) -> int:
        return len(self._blobs)


class IterableBlobDataset(torch.utils.data.IterableDataset[_TransformOutputType_co]):
    def __init__(
        self,
        blobs: Iterable[Blob],
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ):
        self._blobs = blobs
        if transform is None:
            transform = _default_transform
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ) -> Self:
        blobs = _BlobUrlsBlobIterable(blob_urls, credential=credential)
        return cls(blobs, transform=transform)

    @classmethod
    def from_container_url(
        cls,
        container_url: str,
        *,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ) -> Self:
        blobs = _ContainerUrlBlobIterable(
            container_url, prefix=prefix, credential=credential
        )
        return cls(blobs, transform=transform)

    def __iter__(self) -> Iterator[_TransformOutputType_co]:
        worker_info = torch.utils.data.get_worker_info()
        for i, blob in enumerate(self._blobs):
            if self._should_yield_from_worker_shard(worker_info, i):
                yield self._transform(blob)

    def _should_yield_from_worker_shard(self, worker_info, blob_index: int) -> bool:
        if worker_info is None:
            return True
        return blob_index % worker_info.num_workers == worker_info.id


class _BaseBlobIterable(Iterable[Blob]):
    def __init__(self, credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None):
        self._credential = credential
        self._blob_client_factory = _client.AzStorageTorchBlobClientFactory(
            credential=self._credential
        )

    def __iter__(self) -> Iterator[Blob]:
        raise NotImplementedError("__iter__")


class _ContainerUrlBlobIterable(_BaseBlobIterable):
    def __init__(
        self,
        container_url: str,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
    ):
        super().__init__(credential)
        self._container_url = container_url
        self._prefix = prefix

    def __iter__(self) -> Iterator[Blob]:
        blob_clients = self._blob_client_factory.yield_blob_clients_from_container_url(
            self._container_url, prefix=self._prefix
        )
        for blob_client in blob_clients:
            yield Blob(blob_client)


class _BlobUrlsBlobIterable(_BaseBlobIterable):
    def __init__(
        self,
        blob_urls: Union[str, Iterable[str]],
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
    ):
        super().__init__(credential)
        if isinstance(blob_urls, str):
            blob_urls = [blob_urls]
        self._blob_urls = blob_urls

    def __iter__(self) -> Iterator[Blob]:
        for blob_url in self._blob_urls:
            yield Blob(self._blob_client_factory.get_blob_client_from_url(blob_url))
