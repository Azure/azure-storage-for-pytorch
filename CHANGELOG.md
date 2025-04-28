# Release History

## 0.1.0 (Unreleased)

### Features Added
- Added `BlobIO` file-like object for reading and writing data to and from Azure Blob Storage.
Instances of this class can be provided directly to `torch.save()` and `torch.load()` to
respectively save and load PyTorch models with Azure Blob Storage.
- Added `datasets` module. It provides `BlobDataset`, a map-style PyTorch dataset, and
`IterableBlobDataset`, an iterable-style PyTorch dataset, for loading data samples from
Azure Blob Storage. Dataset implementations can be instantiated using class methods
`from_containter_url()` to list data samples from an Azure Storage container or
`from_blob_urls()` to list data samples from a pre-defined list of blobs.

## 0.0.1 (2024-08-23)

### Features Added
- Initialized `azstoragetorch` package. Initial package contained no features.
