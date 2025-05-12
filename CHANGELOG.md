# Release History

## 0.1.1 (2025-05-12)

### Bug Fixes
- Plumb `connection_data_block_size` to shared transport instead of providing directly to SDK
client constructor. This reenabled faster download speeds of larger blobs using `BlobIO`.
- Fix issue with `BlobDataset.from_container_url()` datasets and multi-worker PyTorch dataloaders
where underlying `requests` transport is not safe when used prior to process forking; it can
result in child processes sharing connections: https://github.com/psf/requests/issues/4323.
To fix it, List Blob API calls (e.g., List blobs), which occur in the parent process, no
longer share transport with blob clients, which are used in the workers.
Pipeline policy was also added to assert responses returned from `requests` match their
expected request for cases where a dataset is accessed before and outside of a multi-worker
dataloader in the main process.

### Other Changes
- Reuse `azure.core.pipeline.Pipeline` for clients built as part of `from_blob_urls()`-based
datasets. This improves speed to load data samples as it enables caching OAuth tokens across
clients instead of retrieving a new OAuth token for each blob URL.
- Add `BlobIO.read()` optimization to immediately call GET blob on first `read()` and use
response headers to retrieve blob metadata (e.g, size and ETag) for subsequent downloads
instead of immediately calling GetBlobProperties. This improves speed in downloading smaller
blobs during data loading as it results in one less API call (i.e., GetBlobProperties) per blob.

## 0.1.0 (2025-05-01)

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
