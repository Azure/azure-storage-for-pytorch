# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import io
import pytest

from azstoragetorch.io import BlobIO
from azstoragetorch.exceptions import FatalBlobIOWriteError
from tests.e2e.utils import sample_data, random_resource_name

_SMALL_BLOB_SIZE = 20
_LARGE_BLOB_SIZE = 32 * 1024 * 1024 * 2
_STAGE_BLOCK_SIZE = 32 * 1024 * 1024


@pytest.fixture
def blob_url(account_url, container_client):
    blob_name = random_resource_name()
    return f"{account_url}/{container_client.container_name}/{blob_name}"


def downloaded_blob(container_client, blob_url):
    blob_client = container_client.get_blob_client(blob=blob_url.split("/")[-1])
    stream = io.BytesIO()
    blob_client.download_blob().readinto(stream)
    return stream.getvalue()


class TestWrite:
    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_write_all_content(self, blob_url, blob_size, container_client):
        blob_data = sample_data(blob_size)
        with BlobIO(blob_url, "wb") as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)

        assert downloaded_blob(container_client, blob_url) == blob_data

    @pytest.mark.parametrize(
        "blob_size, n",
        [
            (_SMALL_BLOB_SIZE, 1),
            (_SMALL_BLOB_SIZE, 5),
            (_SMALL_BLOB_SIZE, 20),
            (_SMALL_BLOB_SIZE, 21),
            (_LARGE_BLOB_SIZE, _STAGE_BLOCK_SIZE * 2),
            (_LARGE_BLOB_SIZE, _STAGE_BLOCK_SIZE * 3),
        ],
    )
    def test_write_content_in_chunks(self, blob_size, blob_url, container_client, n):
        blob_data = sample_data(blob_size)
        written = 0
        with BlobIO(blob_url, "wb") as f:
            for i in range(0, len(blob_data), n):
                chunk = blob_data[i : i + n]
                assert f.write(chunk) == len(chunk)
                written += len(chunk)
                assert f.tell() == written

        assert downloaded_blob(container_client, blob_url) == blob_data

    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_write_error(self, blob_size, blob_url):
        blob_data = sample_data(blob_size)
        # Forcing an error by using anonymous credentials assuming the account does not allow anonymous write permissions
        with pytest.raises(FatalBlobIOWriteError):
            with BlobIO(blob_url, "wb", credential=False) as f:
                f.write(blob_data)
                f.flush()

        assert f.closed

    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_overwrite_blob(self, blob_size, blob_url, container_client):
        blob_data = sample_data(blob_size)
        with BlobIO(blob_url, "wb") as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)

        assert downloaded_blob(container_client, blob_url) == blob_data

        blob_data = sample_data(10)
        with BlobIO(blob_url, "wb") as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)

        assert downloaded_blob(container_client, blob_url) == blob_data
