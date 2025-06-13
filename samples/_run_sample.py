# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""For internal development purposes only. Serves as a way to automate running of samples."""

import os
import sys
import re
import subprocess
import tempfile
import argparse

STORAGE_ACCOUNT_NAME = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = os.environ.get("AZSTORAGETORCH_CONTAINER_NAME")
BLOB_IMAGE_NAME = os.environ.get("AZSTORAGETORCH_BLOB_IMAGE_NAME")


def update_placeholders(path):
    if not STORAGE_ACCOUNT_NAME or not CONTAINER_NAME:
        raise ValueError(
            "Please set environment variables AZSTORAGETORCH_STORAGE_ACCOUNT_NAME and AZSTORAGETORCH_CONTAINER_NAME"
        )

    with open(path, "r") as f:
        content = f.read()

    account_pattern = r"<my-storage-account-name>"
    container_pattern = r"<my-container-name>"
    blob_image_pattern = r"<blob-image-name>"
    modified_content = re.sub(account_pattern, STORAGE_ACCOUNT_NAME, content)
    modified_content = re.sub(blob_image_pattern, BLOB_IMAGE_NAME, modified_content)
    return re.sub(container_pattern, CONTAINER_NAME, modified_content)


def run_sample(path):
    modified_content = update_placeholders(path)
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "temp_sample.py"), "w") as temp_file:
            temp_file.write(modified_content)
            temp_path = temp_file.name
        subprocess.run([sys.executable, temp_path], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the sample script")
    args = parser.parse_args()
    run_sample(args.path)
