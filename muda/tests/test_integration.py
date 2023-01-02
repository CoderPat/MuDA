import unittest
import os
import tempfile
import json
import enum
from itertools import chain
from parameterized import parameterized  # type: ignore
from typing import List

from muda import main

TEST_DIR = "./example_data/tests"


# To create new test cases for different phenomena, add a new entry to this enum
# The "expected" file should be a space-separated list of tag values for each token, with "0" meaning "no tag"
# If there are multiple expected phenomena in a token, the values should be separated by a comma (no space)
class Phenomena(enum.Enum):
    lexical_cohesion = 1
    formality = 2
    verb_form = 3
    pronouns = 4
    ellipsis = 5


class BaseTestCase:
    def __init__(self, langcode: str):
        super().__init__()
        self.langcode = langcode

    def runMuda(self) -> None:
        test_dir = os.path.join(TEST_DIR, self.langcode)
        docids_file = os.path.join(test_dir, "example.docids")
        fr_file = os.path.join(test_dir, f"example.{self.langcode}")
        en_file = os.path.join(test_dir, "example.en")
        results_file = os.path.join(test_dir, "example.expected")

        self.temp_tags_file = tempfile.NamedTemporaryFile()

        main_args = {
            "src": en_file,
            "tgt": fr_file,
            "docids": docids_file,
            "tgt_lang": self.langcode,
            "hyps": [],
            "dump_tags": os.path.join("/tmp", self.temp_tags_file.name),
            "phenomena": ["lexical_cohesion", "formality", "verb_form", "pronouns"],
            "awesome_align_model": "bert-base-multilingual-cased",
            "cohesion_threshold": 2,
        }

        main(main_args)

        self.tags_data = list(
            chain.from_iterable(json.loads(self.temp_tags_file.read()))
        )

        with open(results_file, "r") as results_f:
            self.expected_tags = results_f.read().splitlines()

    def test_all(self) -> List[bool]:
        token_results = []

        if self.tags_data is None or self.expected_tags is None:
            self.runMuda()

        for i, tags in enumerate(self.tags_data):
            expected_tags = [x for x in self.expected_tags[i].split(" ")]
            for j, token_tags in enumerate(expected_tags):
                for tag_val in token_tags.split(","):
                    tag_int = int(tag_val)
                    if tag_int == 0:
                        token_results.append(len(self.tags_data[i][j]["tags"]) == 0)
                    else:
                        token_results.append(
                            Phenomena(tag_int).name in self.tags_data[i][j]["tags"]
                        )

        return token_results


class TestLanguages(unittest.TestCase):
    @parameterized.expand(  # type: ignore
        [
            ["Spanish", "es"],
            ["French", "fr"],
            # ["Japanese", "jp"], # the example in the paper is not correct / not tagged by our tagger
            ["Portuguese", "pt"],
            # ["Turkish", "tr"], # not testing ellipsis right now
            ["Chinese", "zh"],
        ]
    )
    def test_all(self, name: str, langcode: str) -> None:
        test_case = BaseTestCase(langcode)
        test_case.runMuda()
        token_results = test_case.test_all()
        error_indices = ",".join([str(i) for i, x in enumerate(token_results) if not x])
        self.assertTrue(
            all(token_results), f"[{name}] errors at tokens {error_indices}"
        )
