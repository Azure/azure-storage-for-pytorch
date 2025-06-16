# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Uses a DataLoader to load data samples in batches"""

from azstoragetorch.datasets import BlobDataset
from torch.utils.data import DataLoader

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

dataset = BlobDataset.from_container_url(CONTAINER_URL)

# Create a DataLoader to load data samples from the dataset in batches of 32
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    print(batch["url"])  # Prints blob URLs for each 32 sample batch
