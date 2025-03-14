import os
import random
import string

import pytest
import torch
import ssl
import functools
import certifi
from azstoragetorch.io import BlobIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


@pytest.fixture(scope="module")
def model():
    ssl._create_default_https_context = functools.partial(
        ssl.create_default_context, cafile=certifi.where()
    )
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet101", pretrained=False)
    return model


@pytest.fixture(scope="module")
def upload_model(model, container_client):
    blob_name = "model.pth"
    torch.save(model.state_dict(), blob_name)
    blob_client = container_client.get_blob_client(blob=blob_name)
    with open(blob_name, "rb") as f:
        blob_client.upload_blob(f)
    return blob_name


@pytest.fixture(scope="module")
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            f'"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture(scope="module")
def container_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_name = random_resource_name()
    container = blob_service_client.create_container(name=container_name)
    yield container
    container.delete_container()


@pytest.fixture()
def blob_url(account_url, container_client, upload_model):
    return f"{account_url}/{container_client.container_name}/{upload_model}"


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


class TestLoad:
    def test_load_existing_model(self, blob_url, model):
        test_model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet101", pretrained=False
        )

        with BlobIO(blob_url, "rb") as f:
            state_dict = torch.load(f)
            test_model.load_state_dict(state_dict)

        for key in model.state_dict():
            assert torch.equal(test_model.state_dict()[key], model.state_dict()[key])
