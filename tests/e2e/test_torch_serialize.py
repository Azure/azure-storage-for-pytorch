# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import pytest
import torch
import torchvision.models

from azstoragetorch.io import BlobIO
from utils import random_resource_name


@pytest.fixture(scope="module", autouse=True)
def torch_hub_cache(tmp_path_factory):
    current_dir = torch.hub.get_dir()
    torch.hub.set_dir(tmp_path_factory.mktemp("torch_hub"))
    yield
    torch.hub.set_dir(current_dir)


@pytest.fixture(scope="module")
def model():
    return torchvision.models.resnet101()


@pytest.fixture(scope="module")
def model_path_name(tmp_path_factory):
    return tmp_path_factory.mktemp("model") / f"{random_resource_name()}.pth"


@pytest.fixture(scope="module")
def container_url(account_url, container_client):
    return f"{account_url}/{container_client.container_name}"


@pytest.fixture(scope="module", autouse=True)
def upload_model(model, container_client, model_path_name):
    torch.save(model.state_dict(), model_path_name)
    blob_client = container_client.get_blob_client(blob=model_path_name.name)
    with open(model_path_name, "rb") as f:
        blob_client.upload_blob(f)


@pytest.fixture()
def state_dict_blob_url(container_url, model_path_name):
    return f"{container_url}/{model_path_name.name}"


def assert_state_dict(expected_state_dict, actual_state_dict):
    assert expected_state_dict.keys() == actual_state_dict.keys()
    for key in expected_state_dict.keys():
        assert torch.equal(expected_state_dict[key], actual_state_dict[key])


class TestTorchSerialize:
    def test_load_existing_model(self, state_dict_blob_url, model):
        with BlobIO(state_dict_blob_url, "rb") as f:
            state_dict = torch.load(f)

        assert_state_dict(model.state_dict(), state_dict)

    def test_save_model(self, model, container_url):
        save_blob_url = f"{container_url}/{random_resource_name()}.pth"
        with BlobIO(save_blob_url, "wb") as f:
            torch.save(model.state_dict(), f)

        with BlobIO(save_blob_url, "rb") as f:
            state_dict = torch.load(f)

        assert_state_dict(model.state_dict(), state_dict)
