import spacy_stanza

from . import register_tagger, Tagger


@register_tagger("zh_tw_tagger")
class TaiwaneseTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.pipeline = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )
