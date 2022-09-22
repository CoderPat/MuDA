from langs.base_tagger import Tagger

class DutchTagger(Tagger):
    def __init__(self):
        super().__init__()
        # Source: https://en.wikipedia.org/wiki/T%E2%80%93V_distinction_in_the_world%27s_languages#Dutch
        self.formality_classes = {
            "t_class": {"jij", "jouw", "jou", "jullie", "je"},
            "v_class": {"u", "men", "uw"},
        }
        from spacy.lang.nl.stop_words import STOP_WORDS

        self.ambiguous_verbform = ["Past"]
        self.stop_words = STOP_WORDS
        # self.tagger = spacy.load("nl_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "nl", processors="tokenize,pos,lemma,depparse"
        )