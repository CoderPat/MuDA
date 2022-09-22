import spacy_stanza

from langs.base_tagger import Tagger


class ArabicTagger(Tagger):
    def __init__(self):
        super().__init__()

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
        self.tagger = spacy_stanza.load_pipeline(
            "ar", processors="tokenize,pos,lemma,depparse"
        )
