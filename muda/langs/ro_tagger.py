from typing import Any
import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("ro_tagger")
class RomanianTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.formality_classes = {
            "t_class": {
                "tu",
                "el",
                "ea",
                "voi",
                "ei",
                "ele",
                "tău",
                "ta",
                "tale",
                "tine",
            },
            "v_class": {
                "dumneavoastră",
                "dumneata",
                "mata",
                "matale",
                "dânsul",
                "dânsa" "dumnealui",
                "dumneaei",
                "dumnealor",
            },
        }
        from spacy.lang.ro.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["el", "ei", "ea", "ele"]
        self.ambiguous_pronouns = {
            "it": ["el", "ea"],
            "they": ["ei", "ele"],
            "them": ["ei", "ele"],
        }
        self.ambiguous_verbform = ["Past", "Imp", "Fut"]
        self.stop_words = STOP_WORDS
        # self.tgt_pipeline = spacy.load("ro_core_news_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "ro", processors="tokenize,pos,lemma,depparse"
        )
