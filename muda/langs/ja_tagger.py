from langs.base_tagger import Tagger


class JapaneseTagger(Tagger):
    def __init__(self):
        super().__init__()
        # Formality verb forms from https://www.aclweb.org/anthology/D19-5203.pdf adapted to stanza tokens
        self.formality_classes = {
            "t_class": {"だ", "だっ", "じゃ", "だろう", "だ", "だけど", "だっ"},
            "v_class": {
                "ござい",
                "ます",
                "いらっしゃれ",
                "いらっしゃい",
                "ご覧",
                "伺い",
                "伺っ",
                "存知",
                "です",
                "まし",
            },
        }

        from spacy.lang.ja.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        # self.tgt_gendered_pronouns = ["私", "僕", "俺"]
        self.ambiguous_pronouns = {
            "i": ["私", "僕", "俺"],
        }
        # self.tagger = spacy.load("ja_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "ja", processors="tokenize,pos,lemma,depparse"
        )
