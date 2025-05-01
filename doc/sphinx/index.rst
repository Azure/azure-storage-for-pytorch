Azure Storage Connector for PyTorch Documentation
=================================================

The Azure Storage Connector for PyTorch (``azstoragetorch``) is a library that provides
seamless, performance-optimized integrations between  `Azure Storage`_ and `PyTorch`_.
Use this library to easily access and store data in Azure Storage while using PyTorch. The
library currently offers:

* :ref:`File-like object for saving and loading PyTorch models (i.e., checkpointing) with Azure Blob Storage <checkpoint-guide>`
* :ref:`PyTorch datasets for loading data samples from Azure Blob Storage <datasets-guide>`

Visit the :ref:`Getting Started <getting-started>` page for more information on how to start using
the Azure Storage Connector for PyTorch.

User Guide
----------
.. toctree::

   user-guide


API Reference
-------------
.. toctree::
   :maxdepth: 2

   api

.. _Azure Storage: https://learn.microsoft.com/azure/storage/common/storage-introduction
.. _PyTorch: https://pytorch.org/
