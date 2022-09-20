from base_tagger import Tagger

en_tagger = spacy_stanza.load_pipeline("en", processors="tokenize,pos,lemma,depparse")

class EnglishTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.en.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tagger = en_tagger

        self.pronoun_types = {
            "1sg": {"I", "me", "my", "mine", "myself"},
            "2": {"you", "your", "yours", "yourself", "yourselves"},
            "3sgm": {"he", "him", "his", "himself"},
            "3sgf": {"she", "her", "hers", "herself"},
            "3sgn": {"it", "its", "itself", "themself"},
            "1pl": {"we", "us", "our", "ours", "ourselves"},
            "3pl": {"they", "them", "their", "theirs", "themselves"},
        }

    def formality_tags(
        self,
        cur_src,
        ctx_src,
        cur_tgt=None,
        ctx_tgt=None,
        cur_align=None,
        ctx_align=None,
    ):
        # TODO?
        return [False for _ in cur_src.split(" ")]

    def pronouns(self, src, ref=None, align=None):
        src = src.split(" ")
        tags = []
        for word in src:
            word = self._normalize(word)
            for pro_type, pro_words in self.pronoun_types.items():
                if word in pro_words:
                    tags.append(pro_type)
                else:
                    tags.append("no_tag")
        return tags

    def ellipsis(self, src, ref=None, align=None):
        # TODO?
        # return [False for _ in cur_src.split(" ")]
        return None
