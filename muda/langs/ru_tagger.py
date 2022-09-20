from base_tagger import Tagger

class RussianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"ты", "тебя", "тебе", "тобой", "твой", "твоя", "твои", "тебе"},
            "v_class": {"вы", "вас", "вам", "вами", "ваш", "ваши"},
        }
        from spacy.lang.ru.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Past"]
        # self.tagger = spacy.load("ru_core_web_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "ru", processors="tokenize,pos,lemma,depparse"
        )