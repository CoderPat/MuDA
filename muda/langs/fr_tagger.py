from typing import Any
import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("fr_tagger")
class FrenchTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.formality_classes = {
            "t_class": {
                "tu",
                "ton",
                "ta",
                "tes",
                "toi",
                "te",
                "tien",
                "tiens",
                "tienne",
                "tiennes",
            },
            "v_class": {"vous", "votre", "vos"},
        }
        # self.tgt_gendered_pronouns = ["il", "ils", "elle", "elles"]
        self.ambiguous_pronouns = {
            "it": ["il", "elle", "lui"],
            "they": ["ils", "elles"],
            "them": ["ils", "elles"],
            "you": ["tu", "vous", "on"],
            "we": ["nous", "on"],
            "this": ["celle", "ceci"],
            "that": ["celle", "celui"],
            "these": ["celles", "ceux"],
            "those": ["celles", "ceux"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Past"]

        from spacy.lang.fr.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        # self.tgt_pipeline = spacy.load("fr_core_news_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "fr", processors="tokenize,pos,lemma,depparse"
        )
