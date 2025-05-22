# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

from collections.abc import Callable, Iterable, Iterator
from typing import Optional, Union, TypedDict, cast
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
    """Object representing a single blob in a dataset.

    Datasets instantiate :py:class:`Blob` objects and pass them directly to a dataset's
    ``transform`` callable. Within the ``transform`` callable, use properties and methods
    to access a blob's properties and content. For example::

        from azstoragetorch.datasets import Blob, BlobDataset

        def to_bytes(blob: Blob) -> bytes:
            with blob.reader() as f:
                return f.read()

        dataset = BlobDataset.from_blob_urls(
            "https://<storage-account-name>.blob.core.windows.net/<container-name>/<blob-name>",
            transform=to_bytes
        )
        print(type(dataset[0]))  # Type should be: <class 'bytes'>


    Instantiating class directly using ``__init__()`` is **not** supported.
    """

    def __init__(self, blob_client: _client.AzStorageTorchBlobClient):
        self._blob_client = blob_client

    @property
    def url(self) -> str:
        """The full endpoint URL of the blob.

        The query string is **not** included in the returned URL.
        """
        return self._blob_client.url

    @property
    def blob_name(self) -> str:
        """The name of the blob."""
        return self._blob_client.blob_name

    @property
    def container_name(self) -> str:
        """The name of the blob's container."""
        return self._blob_client.container_name

    def reader(self) -> BlobIO:
        """Open file-like object for reading the blob's content.

        :returns: A file-like object for reading the blob's content.
        """
        return BlobIO(
            self._blob_client.url, "rb", _azstoragetorch_blob_client=self._blob_client
        )


class BlobDataset(torch.utils.data.Dataset[_TransformOutputType_co]):
    """Map-style dataset for blobs in Azure Blob Storage.

    Data samples returned from dataset map directly one-to-one to blobs in Azure Blob Storage.
    Use :py:meth:`from_blob_urls` or :py:meth:`from_container_url` to create an instance of
    this dataset. For example::

        from azstoragetorch.datasets import BlobDataset

        dataset = BlobDataset.from_container_url(
            "https://<storage-account-name>.blob.core.windows.net/<container-name>"
        )
        print(dataset[0])  # Print first blob in the dataset

    Instantiating dataset class directly using ``__init__()`` is **not** supported.

    **Usage with PyTorch DataLoader**

    The dataset can be provided directly to a PyTorch :py:class:`~torch.utils.data.DataLoader`::

        import torch.utils.data

        loader = torch.utils.data.DataLoader(dataset)

    **Dataset output**

    The default output format of the dataset is a dictionary with the keys:

    * ``url``: The full endpoint URL of the blob.
    * ``data``: The content of the blob as :py:class:`bytes`.

    For example::

        {
            "url": "https://<account-name>.blob.core.windows.net/<container-name>/<blob-name>",
            "data": b"<blob-content>"
        }

    To override the output format, provide a ``transform`` callable to either :py:meth:`from_blob_urls`
    or :py:meth:`from_container_url` when creating the dataset.
    """

    def __init__(
        self,
        blobs: Iterable[Blob],
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ):
        self._blobs = list(blobs)
        if transform is None:
            transform = cast(
                Callable[[Blob], _TransformOutputType_co], _default_transform
            )
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ) -> Self:
        """Instantiate dataset from provided blob URLs.

        **Sample usage**::

            container_url = "https://<storage-account-name>.blob.core.windows.net/<container-name>"
            dataset = BlobDataset.from_blob_urls([
                f"{container_url}/<blob-name-1>",
                f"{container_url}/<blob-name-2>",
                f"{container_url}/<blob-name-3>",
            ])

        :param blob_urls: The full endpoint URLs to the blobs to be used for dataset.
            Can be a single URL or an iterable of URLs. URLs respect SAS tokens,
            snapshots, and version IDs in their query strings.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            See :py:class:`Blob` class for more information on writing a ``transform`` callable to
            override the default dataset output format.

        :returns: Dataset formed from the provided blob URLs.
        """
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
        """Instantiate dataset by listing blobs from provided container URL.

        **Sample usage**::

            dataset = BlobDataset.from_container_url(
                "https://<storage-account-name>.blob.core.windows.net/<container-name>",
            )

        :param container_url: The full endpoint URL to the container to be used for dataset.
            The URL respects SAS tokens in its query string.
        :param prefix: The prefix to filter blobs by. Only blobs whose names begin with
            ``prefix`` will be included in the dataset. If not specified, all blobs
            in the container will be included in the dataset.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            See :py:class:`Blob` class for more information on writing a ``transform`` callable to
            override the default dataset output format.

        :returns: Dataset formed from the blobs in the provided container URL.
        """
        blobs = _ContainerUrlBlobIterable(
            container_url, prefix=prefix, credential=credential
        )
        return cls(blobs, transform=transform)

    def __getitem__(self, index: int) -> _TransformOutputType_co:
        """Retrieve the blob at the specified index in the dataset.

        :param index: The index of the blob to retrieve.
        :returns: The blob, with ``transform`` applied, at the specified index.
        """
        blob = self._blobs[index]
        return self._transform(blob)

    def __len__(self) -> int:
        """Return the number of blobs in the dataset.

        :returns: The number of blobs in the dataset.
        """
        return len(self._blobs)


class IterableBlobDataset(torch.utils.data.IterableDataset[_TransformOutputType_co]):
    """Iterable-style dataset for blobs in Azure Blob Storage.

    Data samples returned from dataset map directly one-to-one to blobs in Azure Blob Storage.
    Use :py:meth:`from_blob_urls` or :py:meth:`from_container_url` to create an instance of
    this dataset. For example::

        from azstoragetorch.datasets import IterableBlobDataset

        dataset = IterableBlobDataset.from_container_url(
            "https://<storage-account-name>.blob.core.windows.net/<container-name>"
        )
        print(next(iter(dataset)))  # Print first blob in the dataset

    Instantiating dataset class directly  using ``__init__()`` is **not** supported.

    **Usage with PyTorch DataLoader**

    The dataset can be provided directly to a PyTorch :py:class:`~torch.utils.data.DataLoader`::

        import torch.utils.data

        loader = torch.utils.data.DataLoader(dataset)

    When setting ``num_workers`` for the :py:class:`~torch.utils.data.DataLoader`,
    the dataset automatically shards data samples returned across workers to avoid the
    ``DataLoader`` returning duplicate data samples from its workers.

    **Dataset output**

    The default output format of the dataset is a dictionary with the keys:

    * ``url``: The full endpoint URL of the blob.
    * ``data``: The content of the blob as :py:class:`bytes`.

    For example::

        {
            "url": "https://<account-name>.blob.core.windows.net/<container-name>/<blob-name>",
            "data": b"<blob-content>"
        }

    To override the output format, provide a ``transform`` callable to either :py:meth:`from_blob_urls`
    or :py:meth:`from_container_url` when creating the dataset.
    """

    def __init__(
        self,
        blobs: Iterable[Blob],
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ):
        self._blobs = blobs
        if transform is None:
            transform = cast(
                Callable[[Blob], _TransformOutputType_co], _default_transform
            )
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: Optional[Callable[[Blob], _TransformOutputType_co]] = None,
    ) -> Self:
        """Instantiate dataset from provided blob URLs.

        **Sample usage**::

            container_url = "https://<storage-account-name>.blob.core.windows.net/<container-name>"
            dataset = IterableBlobDataset.from_blob_urls([
                f"{container_url}/<blob-name-1>",
                f"{container_url}/<blob-name-2>",
                f"{container_url}/<blob-name-3>",
            ])

        :param blob_urls: The full endpoint URLs to the blobs to be used for dataset.
            Can be a single URL or an iterable of URLs. URLs respect SAS tokens,
            snapshots, and version IDs in their query strings.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            See :py:class:`Blob` class for more information on writing a ``transform`` callable to
            override the default dataset output format.

        :returns: Dataset formed from the provided blob URLs.
        """
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
        """Instantiate dataset by listing blobs from provided container URL.

        **Sample usage**::

            dataset = IterableBlobDataset.from_container_url(
                "https://<storage-account-name>.blob.core.windows.net/<container-name>",
            )

        :param container_url: The full endpoint URL to the container to be used for dataset.
            The URL respects SAS tokens in its query string.
        :param prefix: The prefix to filter blobs by. Only blobs whose names begin with
            ``prefix`` will be included in the dataset. If not specified, all blobs
            in the container will be included in the dataset.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            See :py:class:`Blob` class for more information on writing a ``transform`` callable to
            override the default dataset output format.

        :returns: Dataset formed from the blobs in the provided container URL.
        """
        blobs = _ContainerUrlBlobIterable(
            container_url, prefix=prefix, credential=credential
        )
        return cls(blobs, transform=transform)

    def __iter__(self) -> Iterator[_TransformOutputType_co]:
        """Iterate over the blobs in the dataset.

        :returns: An iterator over the blobs, with ``transform`` applied, in the dataset.
            The ``transform`` is applied lazily to each blob as it is yielded.
        """
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
