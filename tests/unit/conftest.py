import pytest


@pytest.fixture
def container_name():
    return "mycontainer"


@pytest.fixture
def container_url(container_name):
    return f"https://myaccount.blob.core.windows.net/{container_name}"


@pytest.fixture
def blob_name():
    return "myblob"


@pytest.fixture
def blob_url(container_url, blob_name):
    return f"{container_url}/{blob_name}"


@pytest.fixture
def blob_content():
    return b"blob content"


@pytest.fixture
def blob_length(blob_content):
    return len(blob_content)
