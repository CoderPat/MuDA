import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("zh_tagger")
class ChineseTagger(Tagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.formality_classes = {
            "t_class": {"你"},
            "v_class": {"您"},
        }
        # self.tgt_pipeline = spacy.load("zh_core_web_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )
