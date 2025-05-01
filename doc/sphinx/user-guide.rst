User Guide
==========

.. _getting-started:

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~
* Python 3.9 or later installed
* Have an `Azure subscription`_ and an `Azure storage account`_

Installation
~~~~~~~~~~~~
Install the Azure Storage Connector for PyTorch (``azstoragetorch``) with `pip`_:

.. code-block:: shell

    pip install azstoragetorch


Configuration
~~~~~~~~~~~~~

``azstoragetorch`` should work without any explicit credential configuration.

``azstoragetorch`` interfaces default to :py:class:`~azure.identity.DefaultAzureCredential`
for  credentials. ``DefaultAzureCredential`` automatically retrieves
`Microsoft Entra ID tokens`_ based on your current environment. For more information
on ``DefaultAzureCredential``, see its `documentation <DefaultAzureCredential guide_>`_.

To override credentials, ``azstoragetorch`` interfaces accept a ``credential``
keyword argument override and accept `SAS`_ tokens in query strings of
provided Azure Storage URLs. See the :doc:`API Reference <api>` for more details.


.. _checkpoint-guide:

Saving and Loading PyTorch Models (Checkpointing)
-------------------------------------------------

PyTorch `supports saving and loading trained models <PyTorch checkpoint tutorial_>`_
(i.e., checkpointing). The core PyTorch interfaces for saving and loading models are
:py:func:`torch.save` and :py:func:`torch.load` respectively. Both of these functions
accept a file-like object to be written to or read from.

``azstoragetorch`` offers the :py:class:`azstoragetorch.io.BlobIO` file-like object class
to save and load models directly to and from Azure Blob Storage when using :py:func:`torch.save`
and :py:func:`torch.load`.

Saving a Model
~~~~~~~~~~~~~~
To save a model to Azure Blob Storage, pass a :py:class:`azstoragetorch.io.BlobIO`
directly to :py:func:`torch.save`. When creating the :py:class:`~azstoragetorch.io.BlobIO`,
specify the URL to the blob you'd like to save the model to and use write mode (i.e., ``wb``)::

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


Loading a Model
~~~~~~~~~~~~~~~
To load a model from Azure Blob Storage, pass a :py:class:`azstoragetorch.io.BlobIO`
directly to :py:func:`torch.load`. When creating the :py:class:`~azstoragetorch.io.BlobIO`,
specify the URL to the blob storing the model weights and use read mode (i.e., ``rb``)::

    import torch
    import torchvision.models  # Install separately: ``pip install torchvision``
    from azstoragetorch.io import BlobIO

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # Model to load weights for. Replace with your own model.
    model = torchvision.models.resnet18()

    # Load trained model from Azure Blob Storage.  This loads the model weights
    # from the blob named "model_weights.pth" in the container specified by CONTAINER_URL.
    with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "rb") as f:
        model.load_state_dict(torch.load(f))


.. _datasets-guide:

PyTorch Datasets
----------------

PyTorch offers the `Dataset and DataLoader primitives <PyTorch dataset tutorial_>`_ for
loading data samples. ``azstoragetorch`` provides implementations for both types
of PyTorch datasets, `map-style and iterable-style datasets <PyTorch dataset types_>`_,
to load data samples from Azure Blob Storage:

* :py:class:`azstoragetorch.datasets.BlobDataset` - `Map-style dataset <PyTorch dataset map-style_>`_.
  Use this class for random access to data samples. The class eagerly lists samples in
  dataset on instantiation.

* :py:class:`azstoragetorch.datasets.IterableBlobDataset` - `Iterable-style dataset <PyTorch dataset iterable-style_>`_.
  Use this class when working with large datasets that may not fit in memory. The class
  lazily lists samples as dataset is iterated over.

Data samples returned from both datasets map directly one-to-one to blobs in Azure Blob Storage.
Both classes can be directly provided to a PyTorch :py:class:`~torch.utils.data.DataLoader`
(read more :ref:`here <datasets-guide-with-dataloader>`). When instantiating these dataset
classes, use one of their class methods:

* ``from_container_url()`` - Instantiate dataset by listing blobs from an Azure Storage container.
* ``from_blob_urls()`` - Instantiate dataset from provided blob URLs

Instantiation directly using ``__init__()`` is **not** supported. Read sections below on
how to use these class methods to create datasets.


Create Dataset from Azure Storage Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create an ``azstoragetorch`` dataset by listing blobs in a single Azure Storage container,
use the dataset class's corresponding ``from_container_url()`` method:

* :py:meth:`azstoragetorch.datasets.BlobDataset.from_container_url()` for map-style dataset
* :py:meth:`azstoragetorch.datasets.IterableBlobDataset.from_container_url()` for iterable-style dataset

The methods accept the URL to the Azure Storage container to list blobs from. Listing
is performed using the `List Blobs API <List Blobs API_>`_. For example::

    from azstoragetorch.datasets import BlobDataset, IterableBlobDataset

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # Create a map-style dataset by listing blobs in the container specified by CONTAINER_URL.
    map_dataset = BlobDataset.from_container_url(CONTAINER_URL)

    # Create an iterable-style dataset by listing blobs in the container specified by CONTAINER_URL.
    iterable_dataset = IterableBlobDataset.from_container_url(CONTAINER_URL)

The above examples lists all blobs in the container. To only include blobs whose name starts with
a specific prefix, provide the ``prefix`` keyword argument::

    from azstoragetorch.datasets import BlobDataset, IterableBlobDataset

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # Create a map-style dataset only including blobs whose name starts with the prefix "images/"
    map_dataset = BlobDataset.from_container_url(CONTAINER_URL, prefix="images/")

    # Create an iterable-style dataset only including blobs whose name starts with the prefix "images/"
    iterable_dataset = IterableBlobDataset.from_container_url(CONTAINER_URL, prefix="images/")


Create Dataset from List of Blobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create an ``azstoragetorch`` dataset from a pre-defined list of blobs, use the dataset class's
corresponding ``from_blob_urls()`` method:

* :py:meth:`azstoragetorch.datasets.BlobDataset.from_blob_urls()` for map-style dataset
* :py:meth:`azstoragetorch.datasets.IterableBlobDataset.from_blob_urls()` for iterable-style dataset

The method accepts a list of blob URLs to create the dataset from. For example::

    from azstoragetorch.datasets import BlobDataset, IterableBlobDataset

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # List of blob URLs to create dataset from. Update with your own blob names.
    blob_urls = [
        f"{CONTAINER_URL}/<blob-name-1>",
        f"{CONTAINER_URL}/<blob-name-2>",
        f"{CONTAINER_URL}/<blob-name-3>",
    ]

    # Create a map-style dataset from the list of blob URLs
    map_dataset = BlobDataset.from_blob_urls(blob_urls)

    # Create an iterable-style dataset from the list of blob URLs
    iterable_dataset = IterableBlobDataset.from_blob_urls(blob_urls)


Transforming Dataset Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default output format of dataset samples are dictionaries representing a blob
in the dataset. Each dictionary has the keys:

* ``url``: The full endpoint URL of the blob.
* ``data``: The content of the blob as :py:class:`bytes`.

For example, when accessing a dataset sample::

    print(map_dataset[0])


It will have the following return format::

    {
        "url": "https://<account-name>.blob.core.windows.net/<container-name>/<blob-name>",
        "data": b"<blob-content>"
    }


To override the output format, provide a ``transform`` callable to either ``from_blob_urls``
or ``from_container_url`` when creating the dataset. The ``transform`` callable accepts a
single positional argument of type :py:class:`azstoragetorch.datasets.Blob` representing
a blob in the dataset. This :py:class:`~azstoragetorch.datasets.Blob` object can be used to
retrieve properties and content of the blob as part of the ``transform`` callable.

Emulating the `PyTorch transform tutorial <PyTorch transform tutorial_>`_, the example below shows
how to transform a :py:class:`~azstoragetorch.datasets.Blob` object to a :py:class:`torch.Tensor` of
a :py:mod:`PIL.Image`::

    from azstoragetorch.datasets import BlobDataset, Blob
    import PIL.Image  # Install separately: ``pip install pillow``
    import torch
    import torchvision.transforms  # Install separately: ``pip install torchvision``

    # Update URL with your own Azure Storage account, container, and blob containing an image
    IMAGE_BLOB_URL = "https://<storage-account-name>.blob.core.windows.net/<container-name>/<blob-image-name>"

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

The output should include the blob name and :py:class:`~torch.Tensor` of the image::

    ("<blob-image-name>", tensor([...]))


.. _datasets-guide-with-dataloader:

Using Dataset with PyTorch DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once instantiated, ``azstoragetorch`` datasets can be provided directly to a PyTorch
:py:class:`~torch.utils.data.DataLoader` for loading samples::

    from azstoragetorch.datasets import BlobDataset
    from torch.utils.data import DataLoader

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    dataset = BlobDataset.from_container_url(CONTAINER_URL)

    # Create a DataLoader to load data samples from the dataset in batches of 32
    dataloader = DataLoader(dataset, batch_size=32)

    for batch in dataloader:
        print(batch["url"])  # Prints blob URLs for each 32 sample batch


Iterable-style Datasets with Multiple Workers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a :py:class:`~azstoragetorch.datasets.IterableBlobDataset` and
:py:class:`~torch.utils.data.DataLoader` with multiple workers (i.e., ``num_workers > 1``), the
:py:class:`~azstoragetorch.datasets.IterableBlobDataset` automatically shards data samples
returned across workers to avoid a :py:class:`~torch.utils.data.DataLoader` from returning
duplicate samples from its workers::

    from azstoragetorch.datasets import IterableBlobDataset
    from torch.utils.data import DataLoader

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

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


.. _Azure subscription: https://azure.microsoft.com/free/
.. _Azure storage account: https://learn.microsoft.com/azure/storage/common/storage-account-overview
.. _pip: https://pypi.org/project/pip/
.. _Microsoft Entra ID tokens: https://learn.microsoft.com/azure/storage/blobs/authorize-access-azure-active-directory
.. _DefaultAzureCredential guide: https://learn.microsoft.com/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview
.. _SAS: https://learn.microsoft.com/azure/storage/common/storage-sas-overview
.. _PyTorch checkpoint tutorial: https://pytorch.org/tutorials/beginner/saving_loading_models.html
.. _PyTorch dataset tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#datasets-dataloaders
.. _PyTorch dataset types: https://pytorch.org/docs/stable/data.html#dataset-types
.. _PyTorch dataset map-style: https://pytorch.org/docs/stable/data.html#map-style-datasets
.. _PyTorch dataset iterable-style: https://pytorch.org/docs/stable/data.html#iterable-style-datasets
.. _List Blobs API: https://learn.microsoft.com/rest/api/storageservices/list-blobs?tabs=microsoft-entra-id
.. _PyTorch transform tutorial: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
