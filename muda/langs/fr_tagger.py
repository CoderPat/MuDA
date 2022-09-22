from langs.base_tagger import Tagger

class FrenchTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {
                "tu",
                "ton",
                "ta",
                "tes",
                "toi",
                "te",
                "tien",
                "tiens",
                "tienne",
                "tiennes",
            },
            "v_class": {"vous", "votre", "vos"},
        }
        # self.tgt_gendered_pronouns = ["il", "ils", "elle", "elles"]
        self.ambiguous_pronouns = {
            "it": ["il", "elle", "lui"],
            "they": ["ils", "elles"],
            "them": ["ils", "elles"],
            "you": ["tu", "vous", "on"],
            "we": ["nous", "on"],
            "this": ["celle", "ceci"],
            "that": ["celle", "celui"],
            "these": ["celles", "ceux"],
            "those": ["celles", "ceux"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Past"]

        from spacy.lang.fr.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        # self.tagger = spacy.load("fr_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "fr", processors="tokenize,pos,lemma,depparse"
        )