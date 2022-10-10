import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("zh_tw_tagger")
class TaiwaneseTagger(Tagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )
