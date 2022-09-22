from langs.base_tagger import Tagger


class ItalianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"tu", "tuo", "tua", "tuoi"},
            "v_class": {"lei", "suo", "sua", "suoi"},
        }
        from spacy.lang.it.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tgt_gendered_pronouns = ["esso", "essa"]
        self.ambiguous_pronouns = {
            "it": ["esso", "essa"],
            "them": ["ellos", "ellas"],
            "this": ["questa", "questo"],
            "that": ["quella", "quello"],
            "these": ["queste", "questi"],
            "those": ["quelle", "quelli"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        # self.tagger = spacy.load("it_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "it", processors="tokenize,pos,lemma,depparse"
        )

    def verb_formality(
        self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags
    ):

        cur_src = cur_src.split(" ")
        cur_tgt = cur_tgt.split(" ")
        tags = [False] * len(cur_tgt)

        align = {
            self._normalize(cur_src[s]): self._normalize(cur_tgt[t])
            for s, t in cur_align.items()
        }
        you_verbs = []
        for tok in cur_src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))

        for i, tok in enumerate(cur_tgt_doc):
            if self._normalize(tok.text) in you_verbs:
                person = tok.morph.get("Person")
                if "2" in person and tok.morph.get("Number") == "Sing":
                    if "2" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("2")
                elif "3" in person and tok.morph.get("Number") == "Sing":
                    if "3" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("3")
        return tags
