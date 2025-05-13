# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import pytest


def test_can_import_package():
    try:
        import azstoragetorch  # noqa: F401
    except ImportError:
        pytest.fail("Expected to be able to import azstoragetorch")
