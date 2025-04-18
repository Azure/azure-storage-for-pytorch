# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
from unittest import mock
import pytest

from azure.core.credentials import AzureSasCredential

from azstoragetorch.datasets import BlobDataset, Blob
from azstoragetorch._client import (
    AzStorageTorchBlobClient,
    AzStorageTorchBlobClientFactory,
)


@pytest.fixture
def create_mock_azstoragetorch_blob_client(blob_url, blob_content):
    def _create_mock_azstoragetorch_blob_client(url=None, data=None):
        if url is None:
            url = blob_url
        if data is None:
            data = blob_content
        client = mock.Mock(AzStorageTorchBlobClient)
        client.url = url
        client.get_blob_size.return_value = len(data)
        client.download.return_value = data
        return client

    return _create_mock_azstoragetorch_blob_client


@pytest.fixture
def mock_azstoragetorch_blob_client(create_mock_azstoragetorch_blob_client):
    return create_mock_azstoragetorch_blob_client()


@pytest.fixture
def blob(mock_azstoragetorch_blob_client):
    return Blob(mock_azstoragetorch_blob_client)


@pytest.fixture
def mock_azstoragetorch_blob_client_factory():
    return mock.Mock(AzStorageTorchBlobClientFactory)


@pytest.fixture(autouse=True)
def azstoragetorch_blob_factory_patch(mock_azstoragetorch_blob_client_factory):
    with mock.patch(
        "azstoragetorch._client.AzStorageTorchBlobClientFactory",
        mock_azstoragetorch_blob_client_factory,
    ):
        mock_azstoragetorch_blob_client_factory.return_value = (
            mock_azstoragetorch_blob_client_factory
        )
        yield mock_azstoragetorch_blob_client_factory


@pytest.fixture
def data_samples(container_url):
    return [
        {"url": f"{container_url}/blob{i}", "data": f"sample data {i}".encode("utf-8")}
        for i in range(10)
    ]


@pytest.fixture
def data_sample_blob_urls(data_samples):
    return [sample["url"] for sample in data_samples]


@pytest.fixture
def data_sample_blob_clients(data_samples, create_mock_azstoragetorch_blob_client):
    return [
        create_mock_azstoragetorch_blob_client(**data_sample)
        for data_sample in data_samples
    ]


class TestBlob:
    def test_url(self, blob, mock_azstoragetorch_blob_client, blob_url):
        mock_azstoragetorch_blob_client.url = blob_url
        assert blob.url == blob_url

    def test_blob_name(self, blob, mock_azstoragetorch_blob_client, blob_name):
        mock_azstoragetorch_blob_client.blob_name = blob_name
        assert blob.blob_name == blob_name

    def test_container_name(
        self, blob, mock_azstoragetorch_blob_client, container_name
    ):
        mock_azstoragetorch_blob_client.container_name = container_name
        assert blob.container_name == container_name

    def test_reader(self, blob, mock_azstoragetorch_blob_client):
        with mock.patch(
            "azstoragetorch.datasets.BlobIO", spec=True
        ) as mock_blob_io_cls:
            reader = blob.reader()
            assert reader is mock_blob_io_cls.return_value
            mock_blob_io_cls.assert_called_once_with(
                blob.url,
                "rb",
                _azstoragetorch_blob_client=mock_azstoragetorch_blob_client,
            )


class TestBlobDataset:
    def assert_expected_dataset(self, dataset, expected_data_samples):
        assert isinstance(dataset, BlobDataset)
        assert len(dataset) == len(expected_data_samples)
        for i in range(len(dataset)):
            assert dataset[i] == expected_data_samples[i]

    def assert_factory_calls_from_container_url(
        self,
        mock_azstoragetorch_blob_client_factory,
        expected_container_url,
        expected_prefix=None,
        expected_credential=None,
    ):
        mock_azstoragetorch_blob_client_factory.assert_called_once_with(
            credential=expected_credential
        )
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.assert_called_once_with(
            expected_container_url, prefix=expected_prefix
        )

    def assert_factory_calls_from_blob_urls(
        self,
        mock_azstoragetorch_blob_client_factory,
        expected_blob_urls,
        expected_credential=None,
    ):
        mock_azstoragetorch_blob_client_factory.assert_called_once_with(
            credential=expected_credential
        )
        assert (
            mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.call_args_list
            == [mock.call(url) for url in expected_blob_urls]
        )

    def test_from_container_url(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(container_url)
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
        )

    def test_from_container_url_with_prefix(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(container_url, prefix="prefix/")
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
            expected_prefix="prefix/",
        )

    def test_from_container_url_with_credential(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        credential = AzureSasCredential("sas_token")
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(container_url, credential=credential)
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
            expected_credential=credential,
        )

    def test_from_container_url_with_transform(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(
            container_url, transform=lambda x: x.url
        )
        self.assert_expected_dataset(
            dataset, expected_data_samples=data_sample_blob_urls
        )
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
        )

    def test_from_blob_urls(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = BlobDataset.from_blob_urls(data_sample_blob_urls)
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=data_sample_blob_urls,
        )

    def test_from_blob_urls_with_single_blob_url(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.return_value = data_sample_blob_clients[
            0
        ]
        dataset = BlobDataset.from_blob_urls(data_sample_blob_urls[0])
        self.assert_expected_dataset(dataset, expected_data_samples=[data_samples[0]])
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=[data_sample_blob_urls[0]],
        )

    def test_from_blob_urls_with_credential(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        credential = AzureSasCredential("sas_token")
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = BlobDataset.from_blob_urls(
            data_sample_blob_urls, credential=credential
        )
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=data_sample_blob_urls,
            expected_credential=credential,
        )

    def test_from_blob_urls_with_transform(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = BlobDataset.from_blob_urls(
            data_sample_blob_urls, transform=lambda x: x.url
        )
        self.assert_expected_dataset(
            dataset, expected_data_samples=data_sample_blob_urls
        )
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=data_sample_blob_urls,
        )
