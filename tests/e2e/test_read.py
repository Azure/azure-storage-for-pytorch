# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import string

import pytest
from azstoragetorch.io import BlobIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


_PARTITIONED_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024


@pytest.fixture
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            f'"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture
def sample_data(request):
    return os.urandom(request.param)


@pytest.fixture
def container_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_name = random_resource_name()
    container = blob_service_client.create_container(name=container_name)
    yield container
    container.delete_container()


@pytest.fixture
def blob_client(container_client, sample_data):
    blob_name = random_resource_name()
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(sample_data)
    return blob_client


@pytest.fixture
def blob_url(account_url, container_client, blob_client):
    return f"{account_url}/{container_client.container_name}/{blob_client.blob_name}"


@pytest.fixture
def blob_io(blob_url):
    yield BlobIO(blob_url=blob_url, mode="rb")


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


class TestRead:
    @pytest.mark.parametrize("sample_data", [20], indirect=True)
    def test_reads_all_data(self, blob_io, sample_data):
        with blob_io as f:
            assert f.read() == sample_data

    @pytest.mark.parametrize("n", [1, 2, 4, 5, 9])
    @pytest.mark.parametrize("sample_data", [20], indirect=True)
    def test_read_n_bytes(self, blob_io, sample_data, n):
        with blob_io as f:
            for i in range(0, 10, n):
                assert f.read(n) == sample_data[i : i + n]

    @pytest.mark.parametrize("n", [1, 2, 4, 5, 9])
    @pytest.mark.parametrize("sample_data", [20], indirect=True)
    def test_random_seeks_and_reads(self, blob_io, sample_data, n):
        with blob_io as f:
            f.seek(n)
            assert f.read() == sample_data[n:]

    @pytest.mark.parametrize("sample_data", [20], indirect=True)
    def test_read_using_iter(self, blob_io, sample_data):
        with blob_io as f:
            data = b""
            for i in iter(f):
                data += i
            assert data == sample_data

    @pytest.mark.parametrize(
        "sample_data",
        [
            20,
            1000,
            _PARTITIONED_DOWNLOAD_THRESHOLD,
            _PARTITIONED_DOWNLOAD_THRESHOLD * 2,
        ],
        indirect=True,
    )
    def test_read_different_blob_sizes(self, blob_io, sample_data):
        with blob_io as f:
            assert f.read() == sample_data
