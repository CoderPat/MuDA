import spacy_stanza

from . import register_tagger, Tagger


@register_tagger("tr_tagger")
class TurkishTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"sen", "senin"},
            "v_class": {"siz", "sizin"},
        }
        from spacy.lang.tr.stop_words import STOP_WORDS

        self.ambiguous_verbform = ["Pqp"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "tr", processors="tokenize,pos,lemma,depparse"
        )
