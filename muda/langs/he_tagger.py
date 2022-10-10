import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("he_tagger")
class HebrewTagger(Tagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: hebrew has t-v distinction only in extreme formality cases

        from spacy.lang.he.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "he", processors="tokenize,pos,lemma,depparse"
        )
