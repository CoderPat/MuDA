from typing import Any
import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("jp_tagger")
class JapaneseTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Formality verb forms from https://www.aclweb.org/anthology/D19-5203.pdf adapted to stanza tokens
        self.formality_classes = {
            "t_class": {"だ", "だっ", "じゃ", "だろう", "だ", "だけど", "だっ"},
            "v_class": {
                "ござい",
                "ます",
                "いらっしゃれ",
                "いらっしゃい",
                "ご覧",
                "伺い",
                "伺っ",
                "存知",
                "です",
                "まし",
            },
        }

        from spacy.lang.ja.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        # self.tgt_gendered_pronouns = ["私", "僕", "俺"]
        self.ambiguous_pronouns = {
            "i": ["私", "僕", "俺"],
        }
        # self.tgt_pipeline = spacy.load("ja_core_news_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "ja", processors="tokenize,pos,lemma,depparse"
        )
