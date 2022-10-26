import unittest
import os
import tempfile
import json
import enum
from typing import List, IO
from itertools import chain

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


class TestFr(unittest.TestCase):
    temp_tags_file: IO[str]
    tags_data: List[List[List[str]]]
    expected_tags: List[str]

    @classmethod
    def setUpClass(self) -> None:
        test_dir = os.path.join(TEST_DIR, "fr")
        docids_file = os.path.join(test_dir, "example.docids")
        fr_file = os.path.join(test_dir, "example.fr")
        en_file = os.path.join(test_dir, "example.en")
        results_file = os.path.join(test_dir, "example.expected")

        self.temp_tags_file = tempfile.NamedTemporaryFile()  # type: ignore

        main_args = {
            "src": en_file,
            "tgt": fr_file,
            "docids": docids_file,
            "tgt_lang": "fr",
            "dump_tags": os.path.join("/tmp", self.temp_tags_file.name),
            "phenomena": ["lexical_cohesion", "formality", "verb_form", "pronouns"],
            "awesome_align_model": "bert-base-multilingual-cased",
            "awesome_align_cachedir": "/projects/tir5/users/patrick/awesome",
        }

        main(main_args)

        self.tags_data = list(
            chain.from_iterable(json.loads(self.temp_tags_file.read()))
        )

        with open(results_file, "r") as results_f:
            self.expected_tags = results_f.read().splitlines()

    def test_all(self) -> None:
        for i, tags in enumerate(self.tags_data):
            expected_tags = [x for x in self.expected_tags[i].split(" ")]
            for j, token_tags in enumerate(expected_tags):
                for tag_val in token_tags.split(","):
                    tag_int = int(tag_val)
                    if tag_int == 0:
                        self.assertTrue(len(self.tags_data[i][j]) == 0)
                    else:
                        self.assertTrue(Phenomena(tag_int).name in self.tags_data[i][j])
