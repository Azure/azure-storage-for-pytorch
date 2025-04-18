# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------


class AZStorageTorchError(Exception):
    """Base class for exceptions raised by ``azstoragetorch``."""

    pass


class FatalBlobIOWriteError(AZStorageTorchError):
    """Raised when a fatal error occurs during :py:class:`~azstoragetorch.io.BlobIO` write operations.

    When this exception is raised, it indicates no more writing can be performed
    on the :py:class:`~azstoragetorch.io.BlobIO` object and no blocks staged as part of this
    :py:class:`~azstoragetorch.io.BlobIO` will be committed. It is recommended to create a
    new :py:class:`~azstoragetorch.io.BlobIO` object and retry all writes when attempting retries.
    """

    _MSG_FORMAT = (
        "Fatal error occurred while writing data. No data written using this BlobIO instance "
        "will be committed to blob. Encountered exception:\n{underlying_exception}"
    )

    def __init__(self, underlying_exception: BaseException):
        super().__init__(
            self._MSG_FORMAT.format(underlying_exception=underlying_exception)
        )
