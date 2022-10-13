import unittest
import spacy_stanza
import os
from collections import defaultdict
import sys
import tempfile
import json
import pdb

from ..langs import TAGGER_REGISTRY, create_tagger
from muda import main

TEST_DIR = "./example_data/tests"

class TestFr(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        formality_dir = os.path.join(TEST_DIR, "fr", "formality")
        docids_file = os.path.join(formality_dir, "example.docids")
        fr_file = os.path.join(formality_dir, "example.fr")
        en_file = os.path.join(formality_dir, "example.en")
        results_file = os.path.join(formality_dir, "example.expected")

        self.temp_tags_file = tempfile.NamedTemporaryFile()

        main_args = {
            "src": en_file,
            "tgt": fr_file,
            "docids": docids_file,
            "tgt_lang": "fr",
            "dump_tags": os.path.join("/tmp", self.temp_tags_file.name),
            "phenomena": ["lexical_cohesion", "formality", "verb_form", "pronouns"],
            "awesome_align_model": "bert-base-multilingual-cased",
            "awesome_align_cachedir": "/projects/tir5/users/patrick/awesome"
        }

        main(main_args)
        
        self.tags_data = json.loads(self.temp_tags_file.read())[0]

        with open(results_file, "r") as results_f:
            self.expected_tags = results_f.read().splitlines()

    def test_formality(self):
        for i, tags in enumerate(self.tags_data):
            expected_tags = [int(x) for x in self.expected_tags[i].split()]
            for j, tag in enumerate(tags):
                if expected_tags[j] == 1:
                    self.assertIn("formality", tag)
                else:
                    self.assertNotIn("formality", tag)


