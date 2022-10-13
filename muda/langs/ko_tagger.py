from typing import Any, List, Dict, Set
import spacy
import spacy_stanza  # type: ignore

from muda import Tagger

from . import register_tagger


@register_tagger("ko_tagger")
class KoreanTagger(Tagger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
        self.tgt_pipeline = spacy_stanza.load_pipeline(
            "ko", processors="tokenize,pos,lemma,depparse"
        )

    def _verb_formality(
        self,
        src_sent: spacy.tokens.doc.Doc,
        tgt_sent: spacy.tokens.doc.Doc,
        align_sent: Dict[int, int],
        prev_formality: Set[str],
    ) -> List[bool]:
        # TODO: this still needs to be refactored/verified
        raise NotImplementedError()
        # tags = []
        # for tok in cur_tgt_doc:
        #     honorific = False
        #     if tok.pos_ == "VERB":
        #         for suffix in [
        #             "어",
        #             "아",
        #             "여",
        #             "요",
        #             "ㅂ니다",
        #             "습니다",
        #             "었어",
        #             "았어",
        #             "였어",
        #             "습니다",
        #             "겠어",
        #             "습니다",
        #         ]:
        #             if tok.text.endswith(suffix):
        #                 honorific = True
        #                 break
        #     for _ in tok.text.split(" "):
        #         if honorific:
        #             if "honorific" in prev_formality_tags:  # TODO for Korean specific
        #                 tags.append(True)
        #             else:
        #                 tags.append(False)
        #                 prev_formality_tags.add("honorific")
        #         else:
        #             tags.append(False)

        # return tags, prev_formality_tags
