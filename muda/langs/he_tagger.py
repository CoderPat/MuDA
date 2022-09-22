from langs.base_tagger import Tagger


class HebrewTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: hebrew has t-v distinction only in extreme formality cases

        from spacy.lang.he.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]
        self.tagger = spacy_stanza.load_pipeline(
            "he", processors="tokenize,pos,lemma,depparse"
        )
