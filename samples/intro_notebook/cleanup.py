# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse
import os
import shutil

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

CONTAINER_NAME = "azstoragetorchintro"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_DIR = os.path.join(ROOT_DIR, "local-models")


def cleanup(account_url):
    delete_local_models_dir()
    delete_container(account_url)


def delete_local_models_dir():
    if os.path.exists(LOCAL_MODELS_DIR):
        print(f"Deleting local models directory: {LOCAL_MODELS_DIR}")
        shutil.rmtree(LOCAL_MODELS_DIR)
    else:
        print(
            f"Skipping deletion of local models directory: {LOCAL_MODELS_DIR}. It does not exist."
        )


def delete_container(account_url):
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=DefaultAzureCredential()
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    if container_client.exists():
        print(f"Deleting container: {container_client.url}")
        container_client.delete_container()
    else:
        print(
            f"Skipping deletion of container: {container_client.url}. It does not exist."
        )


def create_parser():
    parser = argparse.ArgumentParser(
        description="Clean up resources from azstoragetorch-intro notebook"
    )
    parser.add_argument(
        "account_url",
        help="The URL of the Azure Storage account notebook resources were created in.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    cleanup(args.account_url)


if __name__ == "__main__":
    main()
