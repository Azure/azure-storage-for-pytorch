# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

from collections.abc import Callable, Iterable
from typing import Optional, Union, TypeVar, TypedDict

import torch.utils.data

from azstoragetorch.io import BlobIO
from azstoragetorch import _client


_TransformOutputType_co = TypeVar("_TransformOutputType_co", covariant=True)
_TRANSFORM_TYPE = Callable[["Blob"], _TransformOutputType_co]


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
        self, blobs: list[Blob], transform: _TRANSFORM_TYPE = _default_transform
    ):
        self._blobs = blobs
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: _TRANSFORM_TYPE = _default_transform,
    ) -> "BlobDataset":
        if isinstance(blob_urls, str):
            blob_urls = [blob_urls]
        blob_client_factory = _client.AzStorageTorchBlobClientFactory(
            credential=credential
        )
        blobs = [
            Blob(blob_client_factory.get_blob_client_from_url(blob_url))
            for blob_url in blob_urls
        ]
        return cls(blobs, transform=transform)

    @classmethod
    def from_container_url(
        cls,
        container_url: str,
        *,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: _TRANSFORM_TYPE = _default_transform,
    ) -> "BlobDataset":
        blob_client_factory = _client.AzStorageTorchBlobClientFactory(
            credential=credential
        )
        blobs = [
            Blob(blob_client)
            for blob_client in blob_client_factory.yield_blob_clients_from_container_url(
                container_url, prefix=prefix
            )
        ]
        return cls(blobs, transform=transform)

    def __getitem__(self, index: int) -> _TransformOutputType_co:
        blob = self._blobs[index]
        return self._transform(blob)

    def __len__(self) -> int:
        return len(self._blobs)
