# Azure Storage Connector for PyTorch (`azstoragetorch`) (Preview)

The Azure Storage Connector for PyTorch (`azstoragetorch`) is a library that provides
seamless, performance-optimized integrations between [Azure Storage] and  [PyTorch].
Use this library to easily access and store data in Azure Storage while using PyTorch. The
library currently offers:

- [File-like object for saving and loading PyTorch models (i.e., checkpointing) with Azure Blob Storage][user guide checkpointing]
- [PyTorch datasets for loading data samples from Azure Blob Storage][user guide datasets]

## Documentation

For detailed documentation on `azstoragetorch`, we recommend visiting its
[official documentation]. It includes both a [user guide] and [API references]
for the project. Content in this README is scoped to a high-level overview of the
project and its GitHub repository policies.

## Backwards compatibility
While the project is major version `0` (i.e., version is `0.x.y`), public interfaces are not stable.

Backwards incompatible changes may be introduced between minor version bumps (e.g., upgrading from
`0.1.0` to `0.2.0`). If backwards compatibility is needed while using the library,
we recommend pinning to a minor version of the library (e.g., `azstoragetorch~=0.1.0`).

## Getting started

### Prerequisites

- Python 3.9 or later installed
- Have an [Azure subscription] and an [Azure storage account]

### Installation

Install the library with [pip]:

```shell
pip install azstoragetorch
```

### Configuration

`azstoragetorch` should work without any explicit credential configuration.

`azstoragetorch` interfaces default to [`DefaultAzureCredential`][defaultazurecredential guide]
for credentials which automatically retrieves [Microsoft Entra ID tokens] based on
your current environment. For more information on using credentials with
`azstoragetorch`, see the [user guide][user guide configuration].


## Features
This section highlights core features of `azstoragetorch`. For more details, see the [user guide].

### Saving and loading PyTorch models (Checkpointing)
PyTorch [supports saving and loading trained models][pytorch checkpoint tutorial]
(i.e., checkpointing). The core PyTorch interfaces for saving and loading models are
[`torch.save()`][pytorch save] and [`torch.load()`][pytorch load] respectively.
Both of these functions accept a file-like object to be written to or read from.

`azstoragetorch` offers the [`azstoragetorch.io.BlobIO`][blobio reference] file-like
object class to save and load models directly to and from Azure Blob Storage when
using `torch.save()` and `torch.load()`:

```python
import torch
import torchvision.models  # Install separately: ``pip install torchvision``
from azstoragetorch.io import BlobIO

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

# Model to save. Replace with your own model.
model = torchvision.models.resnet18(weights="DEFAULT")

# Save trained model to Azure Blob Storage. This saves the model weights
# to a blob named "model_weights.pth" in the container specified by CONTAINER_URL.
with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "wb") as f:
    torch.save(model.state_dict(), f)

# Load trained model from Azure Blob Storage.  This loads the model weights
# from the blob named "model_weights.pth" in the container specified by CONTAINER_URL.
with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "rb") as f:
    model.load_state_dict(torch.load(f))
```

### PyTorch Datasets

PyTorch offers the [Dataset and DataLoader primitives][pytorch dataset tutorial] for
loading data samples. `azstoragetorch` provides implementations for both types
of PyTorch datasets, [map-style and iterable-style datasets][pytorch dataset types],
to load data samples from Azure Blob Storage:

- [`azstoragetorch.datasets.BlobDataset`][blobdataset reference] - [Map-style dataset][pytorch dataset map-style]
- [`azstoragetorch.datasets.IterableBlobDataset`][iterableblobdataset reference] - [Iterable-style dataset][pytorch dataset iterable-style]

Data samples returned from both datasets map directly one-to-one to blobs in Azure Blob
Storage. When instantiating these dataset classes, use one of their class methods:

- `from_container_url()` - Instantiate dataset by listing blobs from an Azure Storage container.
- `from_blob_urls()` - Instantiate dataset from provided blob URLs


```python
from azstoragetorch.datasets import BlobDataset, IterableBlobDataset

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

# Create an iterable-style dataset by listing blobs in the container specified by CONTAINER_URL.
dataset = IterableBlobDataset.from_container_url(CONTAINER_URL)

# Print the first blob in the dataset. Default output is a dictionary with
# the blob URL and the blob data. Use `transform` keyword argument when
# creating dataset to customize output format.
print(next(iter(dataset)))

# List of blob URLs to create dataset from. Update with your own blob names.
blob_urls = [
    f"{CONTAINER_URL}/<blob-name-1>",
    f"{CONTAINER_URL}/<blob-name-2>",
    f"{CONTAINER_URL}/<blob-name-3>",
]

# Create a map-style dataset from the list of blob URLs
blob_list_dataset = BlobDataset.from_blob_urls(blob_urls)

print(blob_list_dataset[0])  # Print the first blob in the dataset
```
Once instantiated, `azstoragetorch` datasets can be provided directly to a PyTorch
[`DataLoader`][pytorch dataloader] for loading samples:

```python
from torch.utils.data import DataLoader

# Create a DataLoader to load data samples from the dataset in batches of 32
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    print(batch["url"])  # Prints blob URLs for each 32 sample batch
```


## Additional resources

For more information on using the Azure Storage Connector for PyTorch, see the following resources:

* [Official documentation][official documentation]
* [Code samples](samples/)
* [Microsoft Build 2025 presentation][2025 build presentation] - Watch this presentation on the Azure Storage
Connector for PyTorch to learn more about the library and how to use its features with PyTorch.
* [Introductory Jupyter notebook](samples/intro_notebook/azstoragetorch-intro.ipynb) - Run this notebook to learn
how to use the library to save and load PyTorch models and datasets from Azure Blob Storage. This is the same notebook
used in the [Microsoft Build 2025 presentation][2025 build presentation].



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a
CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided
by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

[azure storage]: https://learn.microsoft.com/azure/storage/common/storage-introduction
[azure storage account]: https://learn.microsoft.com/azure/storage/common/storage-account-overview
[azure subscription]: https://azure.microsoft.com/free/
[microsoft entra id tokens]: https://learn.microsoft.com/azure/storage/blobs/authorize-access-azure-active-directory
[defaultazurecredential guide]: https://learn.microsoft.com/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview

[pip]: https://pypi.org/project/pip/

[pytorch]: https://pytorch.org/
[pytorch checkpoint tutorial]: https://pytorch.org/tutorials/beginner/saving_loading_models.html
[pytorch save]: https://pytorch.org/docs/stable/generated/torch.save.html
[pytorch load]: https://pytorch.org/docs/stable/generated/torch.load.html
[pytorch dataloader]: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
[pytorch dataset iterable-style]: https://pytorch.org/docs/stable/data.html#iterable-style-datasets
[pytorch dataset map-style]: https://pytorch.org/docs/stable/data.html#map-style-datasets
[pytorch dataset tutorial]: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#datasets-dataloaders
[pytorch dataset types]: https://pytorch.org/docs/stable/data.html#dataset-types

[official documentation]: https://azure.github.io/azure-storage-for-pytorch/
[user guide]: https://azure.github.io/azure-storage-for-pytorch/user-guide.html
[user guide configuration]: https://azure.github.io/azure-storage-for-pytorch/user-guide.html#configuration
[user guide checkpointing]: https://azure.github.io/azure-storage-for-pytorch/user-guide.html#saving-and-loading-pytorch-models-checkpointing
[user guide datasets]: https://azure.github.io/azure-storage-for-pytorch/user-guide.html#pytorch-datasets
[api references]: https://azure.github.io/azure-storage-for-pytorch/api.html
[blobio reference]: https://azure.github.io/azure-storage-for-pytorch/api.html#azstoragetorch.io.BlobIO
[blobdataset reference]: https://azure.github.io/azure-storage-for-pytorch/api.html#azstoragetorch.datasets.BlobDataset
[iterableblobdataset reference]: https://azure.github.io/azure-storage-for-pytorch/api.html#azstoragetorch.datasets.IterableBlobDataset

[2025 build presentation]: https://youtu.be/lJ9ZiiVP1-w?si=zowVmXemFK4w9HKc&t=1437
