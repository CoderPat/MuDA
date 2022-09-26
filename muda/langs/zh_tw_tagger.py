import spacy_stanza

from langs.base_tagger import Tagger


class TaiwaneseTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )
