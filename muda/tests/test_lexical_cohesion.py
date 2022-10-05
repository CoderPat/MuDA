import unittest
import spacy_stanza
from collections import defaultdict
import sys
import pdb

from muda.langs import create_tagger

class TestLexicalCohesionZh(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.zh_tokenizer = spacy_stanza.load_pipeline("zh", processors="tokenize")
        self.en_tokenizer = spacy_stanza.load_pipeline("en", processors="tokenize,mwt")
        self.tagger = create_tagger("zh")

    def test_cohesion(self):
        # making this a table based test so we can add more test cases if possible later
        test_cases = [
            {"name": "lexical_cohesion_exists", 
            "input": 
                {"en": "Avelile’s mother had HIV virus. Avelile had the virus, she was born with the virus.",
                "zh": "阿维利尔的母亲是携有艾滋病病毒。 阿维利尔也有艾滋病病毒。她一生下来就有。",
                "align": },
            "expected": ([True, True, True, True, True] + [False] * , 
                        ["Avelile's", "mother", "had", "HIV", "virus"])
            }
        ]
        
        for test in test_cases:
            cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))

            zh_tok = self.zh_tokenizer(test["input"]["zh"])
            en_tok = self.en_tokenizer(test["input"]["en"])
            
            # TODO: not sure expected form of align dict, run this through python awesome-align
            align = None
            lexical_tags, cohesion_words = self.tagger.lexical_cohesion(
                test["input"]["en"], test["input"]["en"], test["input"]["zh"], align, cohesion_words
            )   

            # TODO: also test cohesion words since not sure what format they should be in
            self.assertListEqual(
                test["expected"], 
                lexical_tags,
                "failed test {}, expected {} but got {}".format(
                    test["name"], test["expected"][0], lexical_tags
                )
            )



