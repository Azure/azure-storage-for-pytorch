# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: saving_and_loading_model_sample.py
DESCRIPTION:
    This sample demonstrates the way to save and load a PyTorch model using Azure Blob Storage.
USAGE: python saving_and_loading_model_sample.py
    Set the environment variables with your own values before running the sample.
"""

import os
import torch
import torchvision.models  # Install separately: ``pip install torchvision``
from azstoragetorch.io import BlobIO
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError


def setup(my_storage_account_name, my_container_name):
    account_url = f"https://{my_storage_account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_client = blob_service_client.get_container_client(my_container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        print(f"Container {my_container_name} already exists.")

def save_model(CONTAINER_URL):
    # Sample model to save and load. Replace with your own model.
    model = torchvision.models.resnet18(weights="DEFAULT")

    # Save trained model to Azure Blob Storage. This saves the model weights
    # to a blob named "model_weights.pth" in the container specified by CONTAINER_URL.
    with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "wb") as f:
        torch.save(model.state_dict(), f)


def load_model(CONTAINER_URL):
    # Sample model to save and load. Replace with your own model.
    model = torchvision.models.resnet18()

    # Load trained model from Azure Blob Storage.  This loads the model weights
    # from the blob named "model_weights.pth" in the container specified by CONTAINER_URL.
    with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "rb") as f:
        model.load_state_dict(torch.load(f))


if __name__ == "__main__":
    my_storage_account_name = os.environ["AZSTORAGETORCH_STORAGE_ACCOUNT_NAME"]
    my_container_name = "model-sample-container"
    setup(my_storage_account_name, my_container_name)
    
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = (
        f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"
    )

    save_model(CONTAINER_URL)
    load_model(CONTAINER_URL)
