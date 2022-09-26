import spacy_stanza

from . import register_tagger, Tagger


@register_tagger("zh_tagger")
class ChineseTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.formality_classes = {
            "t_class": {"你"},
            "v_class": {"您"},
        }
        # self.tagger = spacy.load("zh_core_web_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )
