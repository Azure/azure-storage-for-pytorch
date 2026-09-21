# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import contextlib
import io
import time

import pytest
from azure.core.exceptions import ResourceNotFoundError

from azstoragetorch.exceptions import FatalBlobIOWriteError
from azstoragetorch.io import BlobIO

from tests.e2e.utils import random_resource_name, sample_data

_SMALL_BLOB_SIZE = 20
_LARGE_BLOB_SIZE = 32 * 1024 * 1024 * 2
# Expected size of each block staged by BlobIO.
_STAGE_BLOCK_SIZE = 32 * 1024 * 1024
_STAGE_BLOCK_POLL_ATTEMPTS = 10
_STAGE_BLOCK_POLL_INTERVAL = 1
_NO_ADDITIONAL_BLOCKS_WAIT = 5


@pytest.fixture
def blob_client(container_client):
    blob_name = random_resource_name()
    return container_client.get_blob_client(blob=blob_name)


@pytest.fixture
def unauthorized_blob_io(blob_client):
    # Assume the target container does not permit anonymous writes.
    f = BlobIO(blob_client.url, "wb", credential=False)
    yield f
    # Suppress the repeated fatal error while cleaning up the file object.
    with contextlib.suppress(FatalBlobIOWriteError):
        f.close()
    assert f.closed


def downloaded_blob(blob_client):
    stream = io.BytesIO()
    blob_client.download_blob().readinto(stream)
    return stream.getvalue()


def assert_staged_blocks(blob_client, expected_sizes):
    # Block staging happens asynchronously, so poll until all expected blocks appear.
    for attempt in range(_STAGE_BLOCK_POLL_ATTEMPTS):
        try:
            _assert_staged_blocks(blob_client, expected_sizes)
            return
        except AssertionError:
            if attempt == _STAGE_BLOCK_POLL_ATTEMPTS - 1:
                raise
            time.sleep(_STAGE_BLOCK_POLL_INTERVAL)


def assert_no_additional_blocks_staged(blob_client, existing_sizes=None):
    # Give delayed asynchronous staging time to change the existing block list.
    if existing_sizes is None:
        existing_sizes = []
    time.sleep(_NO_ADDITIONAL_BLOCKS_WAIT)
    _assert_staged_blocks(blob_client, existing_sizes)


def _assert_staged_blocks(blob_client, expected_sizes):
    try:
        _, uncommitted_blocks = blob_client.get_block_list(
            block_list_type="uncommitted"
        )
        assert sorted(block.size for block in uncommitted_blocks) == sorted(
            expected_sizes
        )
    except ResourceNotFoundError:
        assert expected_sizes == []


class TestWrite:
    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_write_all_content(self, blob_client, blob_size, credential):
        blob_data = sample_data(blob_size)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)
        assert downloaded_blob(blob_client) == blob_data

    @pytest.mark.parametrize("bytes_like_type", [bytes, bytearray, memoryview])
    def test_write_bytes_like_types(self, blob_client, credential, bytes_like_type):
        blob_data = sample_data(_SMALL_BLOB_SIZE)
        content = bytes_like_type(blob_data)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(content) == len(content)
            assert f.tell() == len(content)
        assert downloaded_blob(blob_client) == blob_data

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
    def test_write_content_in_chunks(self, blob_size, blob_client, credential, n):
        blob_data = sample_data(blob_size)
        written = 0
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            for i in range(0, len(blob_data), n):
                chunk = blob_data[i : i + n]
                assert f.write(chunk) == len(chunk)
                written += len(chunk)
                assert f.tell() == written
        assert downloaded_blob(blob_client) == blob_data

    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_overwrite_blob(self, blob_size, blob_client, credential):
        blob_data = sample_data(blob_size)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)
        assert downloaded_blob(blob_client) == blob_data

        blob_data = sample_data(10)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)
        assert downloaded_blob(blob_client) == blob_data

    def test_no_writes_result_in_empty_blob(self, blob_client, credential):
        with BlobIO(blob_client.url, "wb", credential=credential):
            pass
        assert downloaded_blob(blob_client) == b""

    def test_empty_writes_result_in_empty_blob(self, blob_client, credential):
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(b"") == 0
            assert f.tell() == 0
        assert downloaded_blob(blob_client) == b""

    def test_writelines(self, blob_client, credential):
        lines = [b"line1\n", b"line2\n", b"line3\n"]
        expected_content = b"".join(lines)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            f.writelines(lines)
            assert f.tell() == len(expected_content)
        assert downloaded_blob(blob_client) == expected_content

    def test_write_caches_until_reaches_threshold(self, blob_client, credential):
        blob_data = sample_data(_STAGE_BLOCK_SIZE)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            # First write a portion of the blob data, but not enough to reach the stage
            # block size threshold
            assert f.write(blob_data[:-1]) == len(blob_data) - 1
            assert f.tell() == len(blob_data) - 1
            assert_no_additional_blocks_staged(blob_client)

            # Write the last byte to reach the stage block size threshold
            assert f.write(blob_data[-1:]) == 1
            assert f.tell() == len(blob_data)
            assert_staged_blocks(blob_client, [_STAGE_BLOCK_SIZE])
        assert downloaded_blob(blob_client) == blob_data

    def test_write_caches_small_writes_and_uploads_with_large_writes(
        self, blob_client, credential
    ):
        first_small_write = b"a"
        first_large_write = sample_data(_STAGE_BLOCK_SIZE)
        second_small_write = b"b"
        second_large_write = sample_data(_STAGE_BLOCK_SIZE)
        expected_content = b"".join(
            [
                first_small_write,
                first_large_write,
                second_small_write,
                second_large_write,
            ]
        )
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(first_small_write) == len(first_small_write)
            assert f.tell() == len(first_small_write)
            assert_no_additional_blocks_staged(blob_client)

            # Large write triggers a flush(), causing one full staged block and
            # one small staged block of the remaining data
            assert f.write(first_large_write) == len(first_large_write)
            assert f.tell() == len(first_small_write) + len(first_large_write)
            assert_staged_blocks(blob_client, [_STAGE_BLOCK_SIZE, 1])

            # Small write does not trigger a flush, so the staged blocks remain the same
            assert f.write(second_small_write) == len(second_small_write)
            assert f.tell() == (
                len(first_small_write)
                + len(first_large_write)
                + len(second_small_write)
            )
            assert_no_additional_blocks_staged(
                blob_client, existing_sizes=[_STAGE_BLOCK_SIZE, 1]
            )

            # Large write triggers a flush, adding another full staged block and
            # one small staged block of the remaining data
            assert f.write(second_large_write) == len(second_large_write)
            assert f.tell() == len(expected_content)
            assert_staged_blocks(
                blob_client,
                [_STAGE_BLOCK_SIZE, 1, _STAGE_BLOCK_SIZE, 1],
            )
        assert downloaded_blob(blob_client) == expected_content


class TestFlush:
    def test_flush_uploads_cached_write(self, blob_client, credential):
        blob_data = sample_data(_SMALL_BLOB_SIZE)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(blob_data) == len(blob_data)
            assert f.tell() == len(blob_data)
            assert_no_additional_blocks_staged(blob_client)

            f.flush()
            assert f.tell() == len(blob_data)
            assert_staged_blocks(blob_client, [len(blob_data)])
        assert downloaded_blob(blob_client) == blob_data

    def test_flush_is_noop_when_no_writes_cached(self, blob_client, credential):
        blob_data = sample_data(_SMALL_BLOB_SIZE)
        with BlobIO(blob_client.url, "wb", credential=credential) as f:
            assert f.write(blob_data) == len(blob_data)
            f.flush()
            assert_staged_blocks(blob_client, [len(blob_data)])

            f.flush()
            assert f.tell() == len(blob_data)
            assert_no_additional_blocks_staged(
                blob_client, existing_sizes=[len(blob_data)]
            )
        assert downloaded_blob(blob_client) == blob_data


class TestClose:
    def test_close_commits_staged_blocks(self, blob_client, credential):
        blob_data = sample_data(_STAGE_BLOCK_SIZE)
        f = BlobIO(blob_client.url, "wb", credential=credential)
        assert f.write(blob_data) == len(blob_data)
        assert_staged_blocks(blob_client, [_STAGE_BLOCK_SIZE])

        f.close()
        assert f.closed
        assert downloaded_blob(blob_client) == blob_data

    def test_close_flushes_buffered_tail(self, blob_client, credential):
        staged_data = sample_data(_STAGE_BLOCK_SIZE)
        buffered_tail = b"ab"
        f = BlobIO(blob_client.url, "wb", credential=credential)
        assert f.write(staged_data) == len(staged_data)
        assert f.write(buffered_tail) == len(buffered_tail)
        assert_staged_blocks(blob_client, [_STAGE_BLOCK_SIZE])

        f.close()
        assert f.closed
        assert downloaded_blob(blob_client) == staged_data + buffered_tail

    def test_close_is_idempotent(self, blob_client, credential):
        blob_data = sample_data(_SMALL_BLOB_SIZE)
        f = BlobIO(blob_client.url, "wb", credential=credential)
        assert f.write(blob_data) == len(blob_data)

        f.close()
        f.close()
        assert f.closed
        assert downloaded_blob(blob_client) == blob_data

    def test_deletion_commits_staged_blocks(self, blob_client, credential):
        blob_data = sample_data(_STAGE_BLOCK_SIZE)
        f = BlobIO(blob_client.url, "wb", credential=credential)
        assert f.write(blob_data) == len(blob_data)
        assert_staged_blocks(blob_client, [_STAGE_BLOCK_SIZE])

        del f
        assert downloaded_blob(blob_client) == blob_data


class TestFatalWriteErrors:
    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_context_manager_propagates_write_error(
        self, blob_size, unauthorized_blob_io
    ):
        blob_data = sample_data(blob_size)
        with pytest.raises(FatalBlobIOWriteError):
            with unauthorized_blob_io as f:
                f.write(blob_data)
        assert f.closed

    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_flush_propagates_write_error(self, blob_size, unauthorized_blob_io):
        blob_data = sample_data(blob_size)
        unauthorized_blob_io.write(blob_data)

        with pytest.raises(FatalBlobIOWriteError):
            unauthorized_blob_io.flush()

    @pytest.mark.parametrize(
        "blob_size",
        [_SMALL_BLOB_SIZE, _LARGE_BLOB_SIZE],
    )
    def test_close_propagates_write_error(self, blob_size, unauthorized_blob_io):
        blob_data = sample_data(blob_size)
        unauthorized_blob_io.write(blob_data)

        with pytest.raises(FatalBlobIOWriteError):
            unauthorized_blob_io.close()
        assert unauthorized_blob_io.closed

    def test_write_propagates_previous_stage_block_error(
        self, blob_client, unauthorized_blob_io
    ):
        unauthorized_blob_io.write(sample_data(_STAGE_BLOCK_SIZE))

        # Wait for the asynchronous unauthorized staging attempt to finish so the
        # next write observes its stored failure instead of racing the upload.
        assert_no_additional_blocks_staged(blob_client)
        with pytest.raises(FatalBlobIOWriteError):
            unauthorized_blob_io.write(b"more-content")

    @pytest.mark.parametrize(
        "method,args",
        [
            ("write", [b"more-content"]),
            ("flush", []),
            ("close", []),
        ],
    )
    def test_methods_continue_to_fail_after_fatal_error(
        self, method, args, unauthorized_blob_io
    ):
        unauthorized_blob_io.write(sample_data(_SMALL_BLOB_SIZE))

        with pytest.raises(FatalBlobIOWriteError):
            unauthorized_blob_io.flush()
        with pytest.raises(FatalBlobIOWriteError):
            getattr(unauthorized_blob_io, method)(*args)
