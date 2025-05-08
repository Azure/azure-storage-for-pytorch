# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: iterable_blob_dataset_sample.py
DESCRIPTION:
    This sample demonstrates the ways to create an iterable-style dataset from Azure Blob Storage using the IterableBlobDataset class.
USAGE: python iterable_blob_dataset_sample.py
    Set the environment variables with your own values before running the sample.
"""

import os
from azstoragetorch.datasets import IterableBlobDataset
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
    for i in range(3):
        blob_name = f"<blob-name-{i}>"
        blob_client = container_client.get_blob_client(blob=blob_name)
        blob_client.upload_blob(os.urandom(20))


def create_dataset_from_container_url(my_storage_account_name, my_container_name):
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = (
        f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"
    )

    # Create an iterable-style dataset by listing blobs in the container specified by CONTAINER_URL.
    iterable_dataset = IterableBlobDataset.from_container_url(CONTAINER_URL)


def create_dataset_from_container_url_with_prefix(
    my_storage_account_name, my_container_name, prefix
):
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = (
        f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"
    )

    # Create an iterable-style dataset only including blobs whose name starts with the prefix "images/"
    iterable_dataset = IterableBlobDataset.from_container_url(
        CONTAINER_URL, prefix="images/"
    )


def create_dataset_from_blob_urls(my_storage_account_name, my_container_name):
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = (
        f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"
    )

    # List of blob URLs to create dataset from. Update with your own blob names.
    blob_urls = [
        f"{CONTAINER_URL}/<blob-name-1>",
        f"{CONTAINER_URL}/<blob-name-2>",
        f"{CONTAINER_URL}/<blob-name-3>",
    ]

    # Create an iterable-style dataset from the list of blob URLs
    iterable_dataset = IterableBlobDataset.from_blob_urls(blob_urls)


if __name__ == "__main__":
    account_name = os.environ["AZSTORAGETORCH_STORAGE_ACCOUNT_NAME"]
    container_name = "iterable-blob-dataset-sample-container"
    setup(account_name, container_name)
    create_dataset_from_container_url(account_name, container_name)
    create_dataset_from_container_url_with_prefix(
        account_name, container_name, "images/"
    )
    create_dataset_from_blob_urls(account_name, container_name)
