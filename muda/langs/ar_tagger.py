import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("ar_tagger")
class ArabicTagger(Tagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from spacy.lang.ar.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_pronouns = {
            "you": [
                "انت",
                "انتَ",
                "انتِ",
                "انتى",
                "أنتم",
                "أنتن",
                "انتو",
                "أنتما",
                "أنتما",
            ],
            "it": ["هو", "هي"],
            "they": ["هم", "هن", "هما"],
            "them": ["هم", "هن", "هما"],
        }
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "ar", processors="tokenize,pos,lemma,depparse"
        )
