# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse
import concurrent.futures
import io
import hashlib
import os
import shutil
import tempfile
import zipfile

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import requests
import torch
import torchvision

CONTAINER_NAME = "azstoragetorchintro"
DATASET_PREFIX = "datasets/caltech101"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_DIR = os.path.join(ROOT_DIR, "local-models")

DATASET_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
DATASET_MD5 = "3138e1922a9193bfa496528edbbc45d0"


def bootstrap(account_url):
    container_client = create_container(account_url)
    init_model(container_client)
    init_dataset(container_client)
    print(f"Set CONTAINER_URL variable in next cell to: {container_client.url}")


def create_container(account_url):
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=DefaultAzureCredential()
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    print(f"Creating container: {container_client.url}")
    container_client.create_container()
    return container_client


def init_model(container_client):
    model = torchvision.models.resnet18(weights="DEFAULT")
    save_local_model(model, "resnet18_weights.pth")
    save_model_to_blob(container_client, "resnet18_weights.pth")


def init_dataset(container_client):
    with tempfile.TemporaryDirectory() as temp_dir:
        path_to_samples = download_dataset(temp_dir)
        print(
            f"Uploading Caltech 101 dataset to container path: {container_client.url}/{DATASET_PREFIX}/. "
            "This may take several minutes."
        )
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for root, _, filenames in os.walk(path_to_samples):
                for filename in filenames:
                    sample_path = os.path.join(root, filename)
                    futures.append(
                        executor.submit(
                            upload_data_sample, sample_path, container_client
                        )
                    )
            for future in futures:
                future.result()


def upload_data_sample(sample_path, container_client):
    category = get_sample_category(sample_path)
    name = os.path.basename(sample_path)
    blob_name = f"{DATASET_PREFIX}/{category}/{name}"
    blob_client = container_client.get_blob_client(blob_name)
    with open(sample_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)


def get_sample_category(sample_path):
    return os.path.basename(os.path.dirname(sample_path))


def download_dataset(temp_dir):
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    content = io.BytesIO(response.content)
    md5_hash = hashlib.md5(content.getvalue()).hexdigest()
    assert md5_hash == DATASET_MD5, (
        f"MD5 checksum does not match expected value: {DATASET_MD5}, got {md5_hash}"
    )
    with zipfile.ZipFile(content, "r") as zf:
        zf.extractall(temp_dir)
    dataset_tar_path = os.path.join(
        temp_dir, "caltech-101", "101_ObjectCategories.tar.gz"
    )
    shutil.unpack_archive(dataset_tar_path, temp_dir)
    dataset_dir = os.path.join(temp_dir, "101_ObjectCategories")
    return dataset_dir


def save_local_model(model, model_filename):
    print(f"Saving ResNet-18 model to local directory: {LOCAL_MODELS_DIR}")
    if not os.path.exists(LOCAL_MODELS_DIR):
        os.makedirs(LOCAL_MODELS_DIR)
    local_model_file = os.path.join(LOCAL_MODELS_DIR, model_filename)
    torch.save(model.state_dict(), local_model_file)


def save_model_to_blob(container_client, model_filename):
    blob_client = container_client.get_blob_client(f"models/{model_filename}")
    print(f"Uploading ResNet-18 model to blob storage: {blob_client.url}")
    local_model_filename = os.path.join(LOCAL_MODELS_DIR, model_filename)
    with open(local_model_filename, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Bootstrap resources for azstoragetorch-intro notebook"
    )
    parser.add_argument(
        "account_url",
        help="The URL of the Azure Storage account to create resources for notebook.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    bootstrap(args.account_url)


if __name__ == "__main__":
    main()
