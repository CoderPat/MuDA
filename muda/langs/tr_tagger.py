import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("tr_tagger")
class TurkishTagger(Tagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.formality_classes = {
            "t_class": {"sen", "senin"},
            "v_class": {"siz", "sizin"},
        }
        from spacy.lang.tr.stop_words import STOP_WORDS

        self.ambiguous_verbform = ["Pqp"]

        self.stop_words = STOP_WORDS
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "tr", processors="tokenize,pos,lemma,depparse"
        )
