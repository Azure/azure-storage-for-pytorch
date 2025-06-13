# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Loads data samples using multiple workers"""

from azstoragetorch.datasets import IterableBlobDataset
from torch.utils.data import DataLoader

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)


def load_with_workers():
    dataset = IterableBlobDataset.from_container_url(CONTAINER_URL)

    # Iterate over the dataset to get the number of samples in it
    num_samples_from_dataset = len([blob["url"] for blob in dataset])

    # Create a DataLoader to load data samples from the dataset in batches of 32 using 4 workers
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    # Iterate over the DataLoader to get the number of samples returned from it
    num_samples_from_dataloader = 0
    for batch in dataloader:
        num_samples_from_dataloader += len(batch["url"])

    # The number of samples returned from the dataset should be equal to the number of samples
    # returned from the DataLoader. If the dataset did not handle sharding, the number of samples
    # returned from the DataLoader would be ``num_workers`` times (i.e., four times) the number
    # of samples in the dataset.
    assert num_samples_from_dataset == num_samples_from_dataloader


if __name__ == "__main__":
    # Because the DataLoader uses processes for its workers, this if statement protects the script so that
    # spawned processes can safely import the module without risk of calling `load_with_workers()` again.
    load_with_workers()
