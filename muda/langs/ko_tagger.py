import spacy_stanza

from langs.base_tagger import Tagger


class KoreanTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"제가", "저희", "나"},
            "v_class": {
                "댁에",
                "성함",
                "분",
                "생신",
                "식사",
                "연세",
                "병환",
                "약주",
                "자제분",
                "뵙다",
                "저",
            },
        }

        from spacy.lang.ko.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "ko", processors="tokenize,pos,lemma,depparse"
        )

    def verb_formality(
        self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags
    ):
        tags = []
        for tok in cur_tgt_doc:
            honorific = False
            if tok.pos_ == "VERB":
                for suffix in [
                    "어",
                    "아",
                    "여",
                    "요",
                    "ㅂ니다",
                    "습니다",
                    "었어",
                    "았어",
                    "였어",
                    "습니다",
                    "겠어",
                    "습니다",
                ]:
                    if tok.text.endswith(suffix):
                        honorific = True
                        break
            for _ in tok.text.split(" "):
                if honorific:
                    if "honorific" in prev_formality_tags:  # TODO for Korean specific
                        tags.append(True)
                    else:
                        tags.append(False)
                        prev_formality_tags.add("honorific")
                else:
                    tags.append(False)

        return tags, prev_formality_tags
