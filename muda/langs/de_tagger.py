from langs.base_tagger import Tagger


class GermanTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"du"},
            "v_class": {"sie"},  # formal 2nd person Sie is usually capitalized
        }
        from spacy.lang.de.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["er", "sie", "es"]
        self.ambiguous_pronouns = {
            "it": ["er", "sie", "es"],
        }
        # self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "de", processors="tokenize,pos,lemma,depparse"
        )
