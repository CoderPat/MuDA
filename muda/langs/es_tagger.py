import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("es_tagger")
class SpanishTagger(Tagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: usted/su/sus/suyo/suya works for V class and 3rd person
        self.formality_classes = {
            "t_class": {"tú", "tu", "tus", "ti", "contigo", "tuyo", "te", "tuya"},
            "v_class": {"usted", "vosotros", "vuestro", "vuestra", "vuestras", "os"},
        }
        from spacy.lang.es.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["él", "ella", "ellos", "ellas"]
        self.ambiguous_pronouns = {
            "it": ["él", "ella"],
            "they": ["ellos", "ellas"],
            "them": ["ellos", "ellas"],
            "this": ["ésta", "éste", "esto"],
            "that": ["esa", "ese"],
            "these": ["estos", "estas"],
            "those": ["aquellos", "aquellas", "ésos", "ésas"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        self.stop_words = STOP_WORDS
        # self.tgt_pipeline = spacy.load("es_core_news_sm")
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "es", processors="tokenize,pos,lemma,depparse"
        )

    def _verb_formality(
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
                if "2" in person:
                    if "2" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("2")
                elif "3" in person:
                    if "3" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("3")
        return tags, prev_formality_tags
