# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import pytest

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from tests.e2e.utils import random_resource_name


@pytest.fixture(scope="package")
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            '"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture(scope="package")
def blob_service_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    return blob_service_client


@pytest.fixture(scope="package")
def create_container(blob_service_client):
    def _create_container(container_name=None):
        if container_name is None:
            container_name = random_resource_name()
        container_client = blob_service_client.create_container(name=container_name)
        return container_client

    return _create_container


@pytest.fixture(scope="package")
def container_client(create_container):
    container = create_container()
    yield container
    container.delete_container()
