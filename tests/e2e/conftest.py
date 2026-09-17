# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import socket
import time
from datetime import datetime, timedelta, timezone

import pytest
from azure.core.credentials import AzureSasCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    AccountSasPermissions,
    BlobServiceClient,
    ResourceTypes,
    generate_account_sas,
)

from azstoragetorch._client import ALLOW_MISSING_CLIENT_REQUEST_ID_ENV_VAR
from tests.e2e.utils import random_resource_name

_AZURITE_ACCOUNT_NAME = "devstoreaccount1"
_AZURITE_ACCOUNT_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq"
    "/K1SZFPTOtr/KBHBeksoGMGw=="
)
_AZURITE_HOST = "127.0.0.1"
_AZURITE_BLOB_PORT = "10000"


def pytest_addoption(parser):
    parser.addoption(
        "--azurite",
        action="store_true",
        help="Run end-to-end tests against Azurite using its default local endpoint.",
    )


@pytest.fixture(scope="session")
def use_azurite(request):
    enabled = request.config.getoption("--azurite")
    if enabled:
        os.environ[ALLOW_MISSING_CLIENT_REQUEST_ID_ENV_VAR] = "true"
        _wait_for_azurite()
    yield enabled
    if enabled:
        os.environ.pop(ALLOW_MISSING_CLIENT_REQUEST_ID_ENV_VAR)


@pytest.fixture(scope="session")
def account_url(use_azurite):
    if use_azurite:
        return f"http://{_AZURITE_HOST}:{_AZURITE_BLOB_PORT}/{_AZURITE_ACCOUNT_NAME}"

    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            '"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture(scope="session")
def credential(use_azurite):
    if use_azurite:
        # Use SAS because azstoragetorch does not support connection strings or
        # account-key credentials.
        sas_token = generate_account_sas(
            account_name=_AZURITE_ACCOUNT_NAME,
            account_key=_AZURITE_ACCOUNT_KEY,
            resource_types=ResourceTypes(container=True, object=True),
            permission=AccountSasPermissions(
                read=True,
                write=True,
                delete=True,
                list=True,
            ),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        return AzureSasCredential(sas_token)
    return None


@pytest.fixture(scope="package")
def blob_service_client(account_url, credential):
    if credential is None:
        credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)
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


def _wait_for_azurite():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                (_AZURITE_HOST, int(_AZURITE_BLOB_PORT)), timeout=0.2
            ):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Timed out waiting for Azurite to start.")
