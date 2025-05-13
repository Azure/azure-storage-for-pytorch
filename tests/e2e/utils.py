# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import random
import string


def sample_data(data_length=20):
    return os.urandom(data_length)


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )
