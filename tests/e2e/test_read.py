# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import io
from dataclasses import dataclass

import pytest

from azstoragetorch.io import BlobIO

from tests.e2e.utils import random_resource_name, sample_data


_PARTITIONED_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
# Expected readahead GET request size used by BlobIO.readline().
_READLINE_GET_SIZE = 4 * 1024 * 1024


@dataclass
class Blob:
    data: bytes
    url: str


@pytest.fixture(scope="module")
def small_blob(container_client):
    return upload_blob(container_client, sample_data(20))


@pytest.fixture(scope="module")
def large_blob(container_client):
    return upload_blob(
        container_client,
        sample_data(_PARTITIONED_DOWNLOAD_THRESHOLD * 2),
    )


@pytest.fixture(scope="module")
def empty_blob(container_client):
    return upload_blob(container_client, b"")


@pytest.fixture(scope="module")
def small_with_newlines_blob(container_client):
    return upload_blob(
        container_client,
        sample_data_with_newlines(20, 2),
    )


@pytest.fixture(scope="module")
def three_line_blob(container_client):
    return upload_blob(container_client, b"line1\nline2\nline3\n")


@pytest.fixture(scope="module")
def large_line_blob(container_client):
    data = b"a" * (_READLINE_GET_SIZE + 6) + b"\n" + b"b" * (_READLINE_GET_SIZE + 100)
    return upload_blob(container_client, data)


@pytest.fixture
def blob(request):
    return request.getfixturevalue(f"{request.param}_blob")


def sample_data_with_newlines(data_length=20, num_lines=1):
    lines = []
    for i in range(num_lines):
        lines.append(sample_data(int(data_length / num_lines)))
    return b"\n".join(lines)


def upload_blob(container_client, data):
    blob_name = random_resource_name()
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data)
    return Blob(data=data, url=blob_client.url)


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
    def test_reads_all_data(self, blob, credential):
        with BlobIO(blob.url, "rb", credential=credential) as f:
            assert f.read() == blob.data
            assert f.tell() == len(blob.data)

    @pytest.mark.parametrize("n", [1, 5, 20, 21])
    def test_read_n_bytes(self, small_blob, n, credential):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            for i in range(0, len(small_blob.data), n):
                assert f.read(n) == small_blob.data[i : i + n]
                expected_position = min(i + n, len(small_blob.data))
                assert f.tell() == expected_position

    def test_read_beyond_end(self, small_blob, credential):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.read() == small_blob.data
            assert f.tell() == len(small_blob.data)
            assert f.read() == b""
            assert f.tell() == len(small_blob.data)
            assert f.read() == b""
            assert f.tell() == len(small_blob.data)

    def test_read_size_zero(self, small_blob, credential):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.read(0) == b""
            assert f.tell() == 0
            assert f.read() == small_blob.data

    @pytest.mark.parametrize("size", [None, -1])
    def test_read_size_synonyms_for_read_all(self, small_blob, credential, size):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.read(size) == small_blob.data
            assert f.tell() == len(small_blob.data)


class TestSeek:
    @pytest.mark.parametrize("n", [1, 5, 20, 21])
    def test_random_seeks_and_reads(self, small_blob, n, credential):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.seek(n) == n
            assert f.read() == small_blob.data[n:]
            expected_position = max(n, len(small_blob.data))
            assert f.tell() == expected_position

    def test_seek_beyond_end(self, small_blob, credential):
        expected_position = len(small_blob.data) + 1
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.seek(expected_position) == expected_position
            assert f.tell() == expected_position
            assert f.read(1) == b""
            assert f.tell() == expected_position

    @pytest.mark.parametrize("offset", [0, -1])
    def test_seek_from_end(self, small_blob, credential, offset):
        expected_position = len(small_blob.data) + offset
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.seek(offset, io.SEEK_END) == expected_position
            assert f.tell() == expected_position
            assert f.read() == small_blob.data[expected_position:]
            assert f.tell() == len(small_blob.data)

    def test_seek_multiple_times(self, small_blob, credential):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.seek(1) == 1
            assert f.tell() == 1
            assert f.seek(2) == 2
            assert f.tell() == 2
            assert f.seek(0) == 0
            assert f.tell() == 0
            assert f.read() == small_blob.data

    def test_seek_cur(self, small_blob, credential):
        with BlobIO(small_blob.url, "rb", credential=credential) as f:
            assert f.seek(1, io.SEEK_CUR) == 1
            assert f.tell() == 1
            assert f.seek(1, io.SEEK_CUR) == 2
            assert f.tell() == 2
            assert f.read() == small_blob.data[2:]


class TestReadline:
    @pytest.mark.parametrize(
        "lines",
        [
            [b"line1\n", b"line2\n"],
            [b"line1-no-new-line"],
            [b"line1\n", b"line2-no-new-line"],
            [b"line1\n", b"\n", b"\n", b"line2\n"],
            [b"line1 \t\r\f\v\n", b"line2\n"],
        ],
    )
    def test_readline(self, container_client, credential, lines):
        blob = upload_blob(container_client, b"".join(lines))
        with BlobIO(blob.url, "rb", credential=credential) as f:
            expected_position = 0
            for line in lines:
                assert f.readline() == line
                expected_position += len(line)
                assert f.tell() == expected_position

    @pytest.mark.parametrize(
        "size, expected",
        [
            (2, b"li"),
            (8, b"line1\n"),
            (100, b"line1\n"),
            (None, b"line1\n"),
            (-1, b"line1\n"),
            # Size less than -1 is synonymous with size not being set. Note that is different behavior than read()
            # which throws validation errors for sizes < -1. This behavior was chosen to stay consistent with
            # file-like objects from open().
            (-2, b"line1\n"),
        ],
    )
    def test_readline_with_size(self, three_line_blob, credential, size, expected):
        with BlobIO(three_line_blob.url, "rb", credential=credential) as f:
            assert f.readline(size) == expected
            assert f.tell() == len(expected)

    def test_readline_size_zero(self, three_line_blob, credential):
        with BlobIO(three_line_blob.url, "rb", credential=credential) as f:
            assert f.readline(0) == b""
            assert f.tell() == 0
            assert f.readline() == b"line1\n"

    def test_readline_mixed_with_read(self, three_line_blob, credential):
        with BlobIO(three_line_blob.url, "rb", credential=credential) as f:
            assert f.readline() == b"line1\n"
            assert f.tell() == 6
            assert f.read(5) == b"line2"
            assert f.tell() == 11
            assert f.readline() == b"\n"
            assert f.tell() == 12
            assert f.readline() == b"line3\n"
            assert f.tell() == len(three_line_blob.data)

    def test_readline_mixed_with_seek(self, three_line_blob, credential):
        with BlobIO(three_line_blob.url, "rb", credential=credential) as f:
            assert f.readline() == b"line1\n"
            assert f.tell() == 6
            assert f.seek(11) == 11
            assert f.tell() == 11
            assert f.readline() == b"\n"
            assert f.tell() == 12
            assert f.readline() == b"line3\n"
            assert f.tell() == len(three_line_blob.data)

    def test_readline_beyond_end(self, three_line_blob, credential):
        with BlobIO(three_line_blob.url, "rb", credential=credential) as f:
            assert f.seek(0, io.SEEK_END) == len(three_line_blob.data)
            assert f.readline() == b""
            assert f.tell() == len(three_line_blob.data)
            assert f.readline() == b""
            assert f.tell() == len(three_line_blob.data)

    def test_readline_stops_at_newline_across_multiple_gets(
        self, large_line_blob, credential
    ):
        newline_position = large_line_blob.data.index(b"\n")
        first_line = large_line_blob.data[: newline_position + 1]
        remaining_content = large_line_blob.data[newline_position + 1 :]
        with BlobIO(large_line_blob.url, "rb", credential=credential) as f:
            assert f.readline() == first_line
            assert f.tell() == len(first_line)
            assert f.readline() == remaining_content
            assert f.tell() == len(large_line_blob.data)

    def test_readline_with_size_across_multiple_gets(self, large_line_blob, credential):
        size = _READLINE_GET_SIZE + 2
        newline_position = large_line_blob.data.index(b"\n")
        remaining_content = large_line_blob.data[newline_position + 1 :]
        with BlobIO(large_line_blob.url, "rb", credential=credential) as f:
            assert f.readline(size) == large_line_blob.data[:size]
            assert f.tell() == size
            assert f.readline(size) == large_line_blob.data[size : newline_position + 1]
            assert f.tell() == newline_position + 1
            assert f.readline(size) == remaining_content[:size]
            assert f.tell() == newline_position + 1 + size
            assert f.readline(size) == remaining_content[size:]
            assert f.tell() == len(large_line_blob.data)

    def test_readlines(self, three_line_blob, credential):
        expected_lines = [b"line1\n", b"line2\n", b"line3\n"]
        with BlobIO(three_line_blob.url, "rb", credential=credential) as f:
            assert f.readlines() == expected_lines
            assert f.tell() == len(three_line_blob.data)

    def test_read_using_iter(self, small_with_newlines_blob, credential):
        with BlobIO(small_with_newlines_blob.url, "rb", credential=credential) as f:
            lines = [line for line in f]
            expected_lines = io.BytesIO(small_with_newlines_blob.data).readlines()
            assert lines == expected_lines
