# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
import json
import os
from typing import Generator, Tuple, Dict, List

import datasets
from datasets import DownloadManager
from datasets.info import SupervisedKeysData

_CITATION = """@misc{11321/849,	
 title = {{AspectEmo} 1.0: Multi-Domain Corpus of Consumer Reviews for Aspect-Based Sentiment Analysis},	
 author = {Koco{\'n}, Jan and Radom, Jarema and Kaczmarz-Wawryk, Ewa and Wabnic, Kamil and Zaj{\c a}czkowska, Ada and Za{\'s}ko-Zieli{\'n}ska, Monika},	
 url = {http://hdl.handle.net/11321/849},	
 note = {{CLARIN}-{PL} digital repository},	
 copyright = {The {MIT} License},	
 year = {2021}	
}"""

_DESCRIPTION = """AspectEmo dataset: Multi-Domain Corpus of Consumer Reviews for Aspect-Based 
                Sentiment Analysis"""

_HOMEPAGE = "https://clarin-pl.eu/dspace/handle/11321/849"

_LICENSE = "The MIT License"

_URLs = {
    "1.0": "https://huggingface.co/datasets/clarin-pl/aspectemo/resolve/main/data/aspectemo1.zip",
    # '2.0': "",
}

_CLASSES = ["O", "a_minus_m", "a_minus_s", "a_zero", "a_plus_s", "a_plus_m", "a_amb"]


class AspectEmo(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="1.0",
            version=VERSION,
            description="AspectEmo 1.0 Corpus, used in the original paper.",
        ),
        # datasets.BuilderConfig(
        #     name="2.0",
        #     version=VERSION,
        #     description="",
        # ),
    ]

    DEFAULT_CONFIG_NAME = "1.0"

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=_CLASSES, num_classes=len(_CLASSES)
                        )
                    ),
                }
            ),
            supervised_keys=SupervisedKeysData(input="tokens", output="labels"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: DownloadManager
    ) -> List[datasets.SplitGenerator]:
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data.json"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(
        self,
        filepath: str,
        split: str,
    ) -> Generator[Tuple[int, Dict[str, str]], None, None]:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)[split]
            for id_, row in data.items():
                yield id_, {
                    "tokens": row["tokens"],
                    "labels": row["labels"],
                }
