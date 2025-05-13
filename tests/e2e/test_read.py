# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import io
import pytest

from azstoragetorch.io import BlobIO
from dataclasses import dataclass
from utils import sample_data, random_resource_name


_PARTITIONED_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024


@dataclass
class Blob:
    data: bytes
    url: str


@pytest.fixture(scope="module")
def small_blob(account_url, container_client):
    return upload_blob(account_url, container_client, sample_data(20))


@pytest.fixture(scope="module")
def large_blob(account_url, container_client):
    return upload_blob(
        account_url, container_client, sample_data(_PARTITIONED_DOWNLOAD_THRESHOLD * 2)
    )


@pytest.fixture(scope="module")
def empty_blob(account_url, container_client):
    return upload_blob(account_url, container_client, b"")


@pytest.fixture(scope="module")
def small_with_newlines_blob(account_url, container_client):
    return upload_blob(account_url, container_client, sample_data_with_newlines(20, 2))


@pytest.fixture
def blob(request):
    return request.getfixturevalue(f"{request.param}_blob")


def sample_data_with_newlines(data_length=20, num_lines=1):
    lines = []
    for i in range(num_lines):
        lines.append(sample_data(int(data_length / num_lines)))
    return b"\n".join(lines)


def upload_blob(account_url, container_client, data):
    blob_name = random_resource_name()
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data)
    url = f"{account_url}/{container_client.container_name}/{blob_name}"
    return Blob(data=data, url=url)


class TestRead:
    @pytest.mark.parametrize(
        "blob",
        [
            "empty",
            "small",
            "large",
        ],
        indirect=True,
    )
    def test_reads_all_data(self, blob):
        with BlobIO(blob.url, "rb") as f:
            assert f.read() == blob.data
            assert f.tell() == len(blob.data)

    @pytest.mark.parametrize("n", [1, 5, 20, 21])
    def test_read_n_bytes(self, small_blob, n):
        with BlobIO(small_blob.url, "rb") as f:
            for i in range(0, len(small_blob.data), n):
                assert f.read(n) == small_blob.data[i : i + n]
                expected_position = min(i + n, len(small_blob.data))
                assert f.tell() == expected_position

    @pytest.mark.parametrize("n", [1, 5, 20, 21])
    def test_random_seeks_and_reads(self, small_blob, n):
        with BlobIO(small_blob.url, "rb") as f:
            f.seek(n)
            assert f.read() == small_blob.data[n:]
            expected_position = max(n, len(small_blob.data))
            assert f.tell() == expected_position

    def test_read_using_iter(self, small_with_newlines_blob):
        with BlobIO(small_with_newlines_blob.url, "rb") as f:
            lines = [line for line in f]
            expected_lines = io.BytesIO(small_with_newlines_blob.data).readlines()
            assert lines == expected_lines
