# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""PolEmo2 dataset."""
from dataclasses import dataclass
from typing import List, Dict, Generator, Union, Optional, Tuple

import datasets

_DESCRIPTION = """PolEmo 2.0:  Corpus of Multi-Domain Consumer Reviews, evaluation data for article presented at CoNLL."""
_CITATION = """
@inproceedings{kocon-etal-2019-multi,
    title = "Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews",
    author = "Koco{\'n}, Jan  and
      Mi{\l}kowski, Piotr  and
      Za{\'s}ko-Zieli{\'n}ska, Monika",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/K19-1092",
    doi = "10.18653/v1/K19-1092",
    pages = "980--991",}
"""
_HOMEPAGE = "https://clarin-pl.eu/dspace/handle/11321/710"
_LICENSE = "CC-BY-4.0"

_DOMAINS = [
    "all",
    "hotels",
    "medicine",
    "products",
    "reviews",
]
_OUT_DOMAINS = ["Nhotels", "Nmedicine", "Nproducts", "Nreviews"]
_CONFIGS_TEXT = ["text", "sentence"]

_LABELS = ["zero", "minus", "plus", "amb"]

URL_PATH = (
    "https://huggingface.co/datasets/clarin-pl/polemo2-official/resolve/main/data"
)
_URLS = {
    cfg: {
        **{
            domain: {
                split_type: f"{URL_PATH}/{domain}.{cfg}.{split_type}.txt"
                for split_type in ["train", "dev", "test"]
            }
            for domain in _DOMAINS
        },
        **{
            domain: {
                split_type: f"{URL_PATH}/{domain}.{cfg}.{split_type}.txt"
                for split_type in ["train", "dev"]
            }
            for domain in _OUT_DOMAINS
        },
    }
    for cfg in _CONFIGS_TEXT
}


@dataclass
class PolEmo2Config(datasets.BuilderConfig):
    text_cfg: Optional[str] = None
    domain: Optional[str] = None
    train_domains: Optional[List[str]] = None
    dev_domains: Optional[List[str]] = None
    test_domains: Optional[List[str]] = None


class PolEmo2(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PolEmo2Config
    BUILDER_CONFIGS = [
        *[
            PolEmo2Config(
                name=f"{domain}_{text_type}",
                domain=domain,
                text_cfg=text_type,
                train_domains=[domain],
                dev_domains=[domain],
                test_domains=[domain],
            )
            for domain in _DOMAINS
            for text_type in _CONFIGS_TEXT
        ]
    ]
    DEFAULT_CONFIG_NAME = "all_text"

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "target": datasets.features.ClassLabel(
                        names=_LABELS, num_classes=len(_LABELS)
                    ),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE
        )

    def _get_files_by_domains(self, domains: List[str], split: str) -> List[str]:
        return [_URLS[self.config.text_cfg][domain][split] for domain in domains]

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        files = {
            "train": dl_manager.download_and_extract(
                self._get_files_by_domains(
                    domains=self.config.train_domains, split="train"
                )
            ),
            "dev": dl_manager.download_and_extract(
                self._get_files_by_domains(domains=self.config.dev_domains, split="dev")
            ),
            "test": dl_manager.download_and_extract(
                self._get_files_by_domains(
                    domains=self.config.test_domains, split="test"
                )
            ),
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": files["test"]},
            ),
        ]

    def _generate_examples(
        self, filepath: Union[str, List[str]]
    ) -> Generator[Tuple[int, Dict[str, str]], None, None]:

        gid = 0
        for path in filepath:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    splitted_line = line.split(" ")
                    yield gid, {
                        "text": " ".join(splitted_line[:-1]),
                        "target": (
                            splitted_line[-1]
                            .strip()
                            .replace("minus_m", "minus")
                            .replace("plus_m", "plus")
                            .split("_")[-1]
                        ),
                    }
                    gid += 1
