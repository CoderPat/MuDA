import spacy_stanza

from . import register_tagger, Tagger


@register_tagger("ro_tagger")
class RomanianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {
                "tu",
                "el",
                "ea",
                "voi",
                "ei",
                "ele",
                "tău",
                "ta",
                "tale",
                "tine",
            },
            "v_class": {
                "dumneavoastră",
                "dumneata",
                "mata",
                "matale",
                "dânsul",
                "dânsa" "dumnealui",
                "dumneaei",
                "dumnealor",
            },
        }
        from spacy.lang.ro.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["el", "ei", "ea", "ele"]
        self.ambiguous_pronouns = {
            "it": ["el", "ea"],
            "they": ["ei", "ele"],
            "them": ["ei", "ele"],
        }
        self.ambiguous_verbform = ["Past", "Imp", "Fut"]
        self.stop_words = STOP_WORDS
        # self.pipeline = spacy.load("ro_core_news_sm")
        self.pipeline = spacy_stanza.load_pipeline(
            "ro", processors="tokenize,pos,lemma,depparse"
        )
