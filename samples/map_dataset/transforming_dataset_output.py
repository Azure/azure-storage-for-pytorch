# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Override the default output format using a transform callable"""

from azstoragetorch.datasets import BlobDataset, Blob
import PIL.Image  # Install separately: ``pip install pillow``
import torch
import torchvision.transforms  # Install separately: ``pip install torchvision``

# Update URL with your own Azure Storage account, container, and blob containing an image
IMAGE_BLOB_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>/<blob-image-name>"


# Define transform to convert blob to a tuple of (image_name, image_tensor)
def to_img_name_and_tensor(blob: Blob) -> tuple[str, torch.Tensor]:
    # Use blob reader to retrieve blob contents and then transform to an image tensor.
    with blob.reader() as f:
        image = PIL.Image.open(f)
        image_tensor = torchvision.transforms.ToTensor()(image)
    return blob.blob_name, image_tensor


# Provide transform to dataset constructor
dataset = BlobDataset.from_blob_urls(
    IMAGE_BLOB_URL,
    transform=to_img_name_and_tensor,
)

print(dataset[0])  # Prints tuple of (image_name, image_tensor) for blob in dataset
