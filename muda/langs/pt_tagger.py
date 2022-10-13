from typing import Any
import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("pt_tagger")
class PortugueseTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # TODO: this is incomplete
        # TODO: shit I think brazilian rules are different
        self.formality_classes = {
            "t_class": {"tu", "tua", "teu", "teus", "tuas", "te"},
            "v_class": {"você", "sua", "seu", "seus", "suas", "lhe"},
        }
        from spacy.lang.pt.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["ele", "ela", "eles", "elas"]
        self.ambiguous_pronouns = {
            "this": ["este", "esta", "esse", "essa"],
            "that": ["este", "esta", "esse", "essa"],
            "these": ["estes", "estas", "esses", "essas"],
            "those": ["estes", "estas", "esses", "essas"],
            "it": ["ele", "ela", "o", "a"],
            "they": ["eles", "elas"],
            "them": ["eles", "elas", "os", "as"],
        }
        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pqp"]

        # self.tgt_pipeline = spacy.load("pt_core_news_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "pt", processors="tokenize,pos,lemma,depparse"
        )
