# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import dataclasses
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

import pytest
import torch.utils.data

from azstoragetorch.datasets import BlobDataset, IterableBlobDataset


def generate_and_upload_data_samples(num_samples, container_client, blob_prefix=""):
    futures = []
    with ThreadPoolExecutor() as executor:
        for i in range(num_samples):
            futures.append(
                executor.submit(
                    generate_and_upload_data_sample,
                    i,
                    container_client,
                    blob_prefix=blob_prefix,
                )
            )
    return [future.result() for future in futures]


def generate_and_upload_data_sample(index, container_client, blob_prefix=""):
    blob_name = f"{blob_prefix}{index}.txt"
    content = f"Sample content for {blob_name}".encode("utf-8")
    container_client.upload_blob(name=blob_name, data=content)
    return {
        "url": f"{container_client.url}/{blob_name}",
        "data": content,
    }


def parse_blob_name_from_url(blob_url, container_url):
    if not container_url.endswith("/"):
        container_url += "/"
    return blob_url.removeprefix(container_url)


def sort_samples(samples):
    return sorted(samples, key=lambda x: x["url"])


def blob_properties_only_transform(blob):
    return {
        "url": blob.url,
        "blob_name": blob.blob_name,
        "container_name": blob.container_name,
    }


def get_blob_properties_only_data_samples(data_samples, container_name, container_url):
    return [
        {
            "url": sample["url"],
            "blob_name": parse_blob_name_from_url(sample["url"], container_url),
            "container_name": container_name,
        }
        for sample in data_samples
    ]


def get_blob_urls(data_samples):
    return [sample["url"] for sample in data_samples]


@pytest.fixture(scope="module")
def blob_prefix():
    return "prefix/"


@pytest.fixture(scope="module")
def prefixed_data_samples(blob_prefix, dataset_container):
    return generate_and_upload_data_samples(
        5, dataset_container, blob_prefix=blob_prefix
    )


@pytest.fixture(scope="module")
def data_samples(prefixed_data_samples, dataset_container):
    return (
        generate_and_upload_data_samples(5, dataset_container) + prefixed_data_samples
    )


@pytest.fixture(scope="module")
def other_data_samples(other_dataset_container):
    return generate_and_upload_data_samples(
        5, other_dataset_container, blob_prefix="other-"
    )


@pytest.fixture(scope="module")
def many_data_samples(many_samples_dataset_container):
    # The default page size for list blobs is 5000. So generate
    # a dataset with more than 5000 blobs to confirm the from_container_url
    # methods fully load all blobs, even if it requires multiple pages.
    #
    # Also, sort the samples based on the blob name to match the return order
    # from the list blobs API.
    return sort_samples(
        generate_and_upload_data_samples(
            6000, many_samples_dataset_container, blob_prefix="many-samples-"
        )
    )


@pytest.fixture(scope="module")
def dataset_container(create_container):
    container_client = create_container()
    yield container_client
    container_client.delete_container()


@pytest.fixture(scope="module")
def other_dataset_container(create_container):
    container_client = create_container()
    yield container_client
    container_client.delete_container()


@pytest.fixture(scope="module")
def many_samples_dataset_container(create_container):
    container_client = create_container()
    yield container_client
    container_client.delete_container()


@dataclasses.dataclass
class DatasetCase:
    dataset_from_url_method: Callable[..., Union[BlobDataset, IterableBlobDataset]]
    url: Union[str, Iterable[str]]
    expected_data_samples: list[Any]
    dataset_from_url_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dataset(self) -> Union[BlobDataset, IterableBlobDataset]:
        return self.dataset_from_url_method(self.url, **self.dataset_from_url_kwargs)


@pytest.fixture(scope="module")
def create_dataset_case(dataset_container, data_samples, credential):
    def _create_dataset_case(
        dataset_cls,
        from_url_method_name,
        url=None,
        expected_data_samples=None,
        dataset_from_url_kwargs=None,
    ):
        if url is None:
            if from_url_method_name == "from_container_url":
                url = dataset_container.url
            else:
                url = get_blob_urls(data_samples)
        if expected_data_samples is None:
            expected_data_samples = data_samples
        if dataset_from_url_kwargs is None:
            dataset_from_url_kwargs = {}
        dataset_from_url_kwargs["credential"] = credential
        return DatasetCase(
            dataset_from_url_method=getattr(dataset_cls, from_url_method_name),
            url=url,
            expected_data_samples=expected_data_samples,
            dataset_from_url_kwargs=dataset_from_url_kwargs,
        )

    return _create_dataset_case


@pytest.fixture(scope="module")
def default_from_container_url_case_kwargs():
    return {
        "from_url_method_name": "from_container_url",
    }


@pytest.fixture(scope="module")
def default_from_blob_urls_case_kwargs():
    return {
        "from_url_method_name": "from_blob_urls",
    }


@pytest.fixture(scope="module")
def prefix_case_kwargs(blob_prefix, prefixed_data_samples):
    return {
        "from_url_method_name": "from_container_url",
        "expected_data_samples": prefixed_data_samples,
        "dataset_from_url_kwargs": {"prefix": blob_prefix},
    }


@pytest.fixture(scope="module")
def transform_from_container_url_case_kwargs(dataset_container, data_samples):
    return {
        "from_url_method_name": "from_container_url",
        "expected_data_samples": get_blob_properties_only_data_samples(
            data_samples,
            dataset_container.container_name,
            dataset_container.url,
        ),
        "dataset_from_url_kwargs": {"transform": blob_properties_only_transform},
    }


@pytest.fixture(scope="module")
def transform_from_blob_urls_case_kwargs(dataset_container, data_samples):
    return {
        "from_url_method_name": "from_blob_urls",
        "expected_data_samples": get_blob_properties_only_data_samples(
            data_samples,
            dataset_container.container_name,
            dataset_container.url,
        ),
        "dataset_from_url_kwargs": {"transform": blob_properties_only_transform},
    }


@pytest.fixture(scope="module")
def single_blob_url_case_kwargs(data_samples):
    return {
        "from_url_method_name": "from_blob_urls",
        "url": get_blob_urls(data_samples)[0],
        "expected_data_samples": data_samples[:1],
    }


@pytest.fixture(scope="module")
def different_containers_case_kwargs(request, data_samples, other_data_samples):
    urls = get_blob_urls(data_samples) + get_blob_urls(other_data_samples)
    return {
        "from_url_method_name": "from_blob_urls",
        "url": urls,
        "expected_data_samples": data_samples + other_data_samples,
    }


@pytest.fixture(scope="module")
def many_samples_case_kwargs(many_samples_dataset_container, many_data_samples):
    return {
        "from_url_method_name": "from_container_url",
        "url": many_samples_dataset_container.url,
        "expected_data_samples": many_data_samples,
    }


CORE_KWARG_CASES = [
    "default_from_container_url",
    "default_from_blob_urls",
    "prefix",
    "transform_from_container_url",
    "transform_from_blob_urls",
    "single_blob_url",
    "different_containers",
]
# Separate the many samples cases from the other cases to help reduce
# end-to-end test runtime. These cases have magnitudes more blobs and
# including them with the other cases would drastically slow down the
# test suite. So we selectively run the many samples cases to make
# sure the implementations can handle larger number of blobs but not
# necessarily run them for every test case.
ALL_KWARG_CASES = CORE_KWARG_CASES + ["many_samples"]


@pytest.fixture(scope="module", params=CORE_KWARG_CASES)
def core_case_kwargs(request):
    return request.getfixturevalue(f"{request.param}_case_kwargs")


@pytest.fixture(scope="module", params=ALL_KWARG_CASES)
def all_case_kwargs(request):
    return request.getfixturevalue(f"{request.param}_case_kwargs")


class TestDatasets:
    def batched(self, data_samples, batch_size=1):
        batched_samples = []
        keys = data_samples[0].keys()
        for i in range(0, len(data_samples), batch_size):
            batch = data_samples[i : i + batch_size]
            sample = {}
            for key in keys:
                sample[key] = [item[key] for item in batch]
            batched_samples.append(sample)
        return batched_samples

    def test_map_dataset(self, create_dataset_case, core_case_kwargs):
        dataset_case = create_dataset_case(BlobDataset, **core_case_kwargs)
        dataset = dataset_case.to_dataset()
        assert isinstance(dataset, BlobDataset)
        assert len(dataset) == len(dataset_case.expected_data_samples)
        for i in range(len(dataset)):
            assert dataset[i] == dataset_case.expected_data_samples[i]

    def test_iterable_dataset(self, create_dataset_case, core_case_kwargs):
        dataset_case = create_dataset_case(IterableBlobDataset, **core_case_kwargs)
        dataset = dataset_case.to_dataset()
        assert isinstance(dataset, IterableBlobDataset)
        assert list(dataset) == dataset_case.expected_data_samples

    @pytest.mark.parametrize("dataset_cls", [BlobDataset, IterableBlobDataset])
    def test_default_loader(self, dataset_cls, create_dataset_case, core_case_kwargs):
        dataset_case = create_dataset_case(dataset_cls, **core_case_kwargs)
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset)
        loaded_samples = list(loader)
        assert loaded_samples == self.batched(
            dataset_case.expected_data_samples, batch_size=1
        )

    @pytest.mark.parametrize("dataset_cls", [BlobDataset, IterableBlobDataset])
    def test_can_load_across_multiple_epochs(
        self, dataset_cls, create_dataset_case, core_case_kwargs
    ):
        dataset_case = create_dataset_case(dataset_cls, **core_case_kwargs)
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=None)
        for _ in range(2):
            loaded_samples = list(loader)
            assert loaded_samples == dataset_case.expected_data_samples

    @pytest.mark.parametrize("dataset_cls", [BlobDataset, IterableBlobDataset])
    def test_loader_with_workers_and_epochs(
        self, dataset_cls, create_dataset_case, all_case_kwargs
    ):
        dataset_case = create_dataset_case(dataset_cls, **all_case_kwargs)
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)
        # Include multiple epochs to ensure the dataset is compatible when multiple
        # processes are used as well as across dataloader iterations, which will
        # create new workers/processes for each iteration.
        for _ in range(2):
            loaded_samples = list(loader)
            assert loaded_samples == dataset_case.expected_data_samples

    @pytest.mark.parametrize("dataset_cls", [BlobDataset, IterableBlobDataset])
    def test_loader_with_batch_size(
        self, dataset_cls, create_dataset_case, core_case_kwargs
    ):
        dataset_case = create_dataset_case(dataset_cls, **core_case_kwargs)
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        loaded_samples = list(loader)
        assert loaded_samples == self.batched(
            dataset_case.expected_data_samples, batch_size=4
        )

    def test_loader_with_shuffle(self, create_dataset_case, core_case_kwargs):
        dataset_case = create_dataset_case(BlobDataset, **core_case_kwargs)
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
        loaded_samples = list(loader)
        assert sort_samples(loaded_samples) == sort_samples(
            dataset_case.expected_data_samples
        )
