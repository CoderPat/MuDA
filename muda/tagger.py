import argparse
from collections import defaultdict
from allennlp.predictors.predictor import Predictor  # type: ignore

from langs import (
    base_tagger,
    ar_tagger,
    de_tagger,
    en_tagger,
    es_tagger,
    fr_tagger,
    he_tagger,
    it_tagger,
    ja_tagger,
    ko_tagger,
    nl_tagger,
    pt_tagger,
    ro_tagger,
    ru_tagger,
    tr_tagger,
    zh_tagger,
)


def build_tagger(lang) -> base_tagger.Tagger:
    taggers = {
        "en": en_tagger.EnglishTagger,
        "fr": fr_tagger.FrenchTagger,
        "pt_br": pt_tagger.PortugueseTagger,
        "es": es_tagger.SpanishTagger,
        "de": de_tagger.GermanTagger,
        "he": he_tagger.HebrewTagger,
        "nl": nl_tagger.DutchTagger,
        "ro": ro_tagger.RomanianTagger,
        "tr": tr_tagger.TurkishTagger,
        "ar": ar_tagger.ArabicTagger,
        "it": it_tagger.ItalianTagger,
        "ko": ko_tagger.KoreanTagger,
        "ru": ru_tagger.RussianTagger,
        "ja": ja_tagger.JapaneseTagger,
        "zh_cn": zh_tagger.ChineseTagger,
        "zh_tw": zh_tagger.TaiwaneseTagger,
    }
    return taggers[lang]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-tok-file", required=True, help="")
    parser.add_argument("--tgt-tok-file", required=True, help="")
    parser.add_argument("--src-detok-file", required=True, help="")
    parser.add_argument("--tgt-detok-file", required=True, help="")
    parser.add_argument(
        "--ellipsis-file",
        default="/projects/tir4/users/kayoy/contextual-mt/data-with-de/ellipsis-nofrag.en",
        help="file with source ellipsis bools",
    )
    parser.add_argument(
        "--ellipsis-filt-file",
        default="/projects/tir4/users/kayoy/contextual-mt/data-with-de/ellipsis.manual.en",
        help="file with source ellipsis bools",
    )
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument(
        "--alignments-file", required=True, help="file with word alignments"
    )
    # parser.add_argument("--polysemous-file", required=True, help="file with polysemous words")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.src_tok_file, "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.tgt_tok_file, "r", encoding="utf-8") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args.ellipsis_file, "r", encoding="utf-8") as file:
        ellipsis_sents = [line.split("|||")[0].strip() == "True" for line in file]
    with open(args.ellipsis_filt_file, "r", encoding="utf-8") as file:
        ellipsis_sents_filt = [line.split("|||")[0].strip() == "True" for line in file]
    with open(args.src_detok_file, "r", encoding="utf-8") as src_f:
        detok_srcs = [line.strip() for line in src_f]
    with open(args.tgt_detok_file, "r", encoding="utf-8") as tgt_f:
        detok_tgts = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r", encoding="utf-8") as docids_f:
        docids = [idx for idx in docids_f]
    with open(args.alignments_file, "r", encoding="utf-8") as file:
        alignments = file.readlines()

    alignments = list(
        map(
            lambda x: dict(
                list(map(lambda y: list(map(int, y.split("-"))), x.strip().split(" ")))
            ),
            alignments,
        )
    )

    tagger = build_tagger(args.target_lang)
    en_coref = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )

    src_docs = en_tagger.pipe(detok_srcs)
    tgt_docs = tagger.tagger.pipe(detok_tgts)

    prev_docid = None
    failed_coref = 0
    tag_len_mismatch = 0
    with open(args.output, "w", encoding="utf-8") as output_file:
        for (
            source,
            target,
            ellipsis_sent,
            ellipsis_sent_filt,
            cur_src_doc,
            cur_tgt_doc,
            docid,
            align,
        ) in zip(
            srcs,
            tgts,
            ellipsis_sents,
            ellipsis_sents_filt,
            src_docs,
            tgt_docs,
            docids,
            alignments,
        ):
            if prev_docid is None or docid != prev_docid:
                prev_docid = docid
                source_context = []
                target_context = []
                # align_context = []
                cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))
                prev_formality_tags = set()
                verb_forms = set()
                verbs = set()
                nouns = set()
                verbs_filt = set()
                nouns_filt = set()

            # TODO: this might be used for V2?
            # current_src_ctx = source_context[
            #     len(source_context) - args.source_context_size :
            # ]
            # current_tgt_ctx = target_context[
            #     len(target_context) - args.target_context_size :
            # ]
            # current_align_ctx = align_context[
            #     len(align_context)
            #     - max(args.source_context_size, args.target_context_size) :
            # ]

            has_ante = [False for _ in range(len(source.split(" ")))]
            # TODO: proper exception capturing
            try:
                coref = en_coref.predict(document=source)
                if len(source.split(" ")) != len(coref["document"]):
                    raise Exception()

                for cluster in coref["clusters"]:
                    for mention in cluster[1:]:
                        for i in range(mention[0], mention[1] + 1):
                            has_ante[i] = True
            except BaseException:
                failed_coref += 1
                pass

            lexical_tags, cohesion_words = tagger.lexical_cohesion(
                target, cur_src_doc, cur_tgt_doc, align, cohesion_words
            )
            formality_tags, prev_formality_tags = tagger.formality_tags(
                source, cur_src_doc, target, cur_tgt_doc, align, prev_formality_tags
            )
            verb_tags, verb_forms = tagger.verb_form(cur_tgt_doc, verb_forms)
            pronouns_tags = tagger.pronouns(cur_src_doc, cur_tgt_doc, align, has_ante)
            ellipsis_tags, verbs, nouns = tagger.ellipsis(
                cur_src_doc, cur_tgt_doc, align, verbs, nouns, ellipsis_sent
            )
            ellipsis_tags_filt, verbs_filt, nouns_filt = tagger.ellipsis(
                cur_src_doc,
                cur_tgt_doc,
                align,
                verbs_filt,
                nouns_filt,
                ellipsis_sent_filt,
            )
            posmorph_tags = tagger.pos_morph(target, cur_tgt_doc)
            tags = []

            # TODO: fix this
            if any(
                len(p_tags) != len(target.split(" "))
                for p_tags in (
                    pronouns_tags,
                    formality_tags,
                    ellipsis_tags,
                    ellipsis_tags_filt,
                    posmorph_tags,
                    verb_tags,
                )
            ):
                tags = ["all" for _ in range(len(target.split(" ")))]
                tag_len_mismatch += 1
            else:
                for i in range(len(lexical_tags)):
                    tag = ["all"]

                    if pronouns_tags[i]:
                        tag.append("pronouns")
                    if formality_tags[i]:
                        tag.append("formality")
                    if verb_tags[i]:
                        tag.append("verb_tense")
                    if ellipsis_tags[i]:
                        tag.append("ellipsis")
                    if ellipsis_tags_filt[i]:
                        tag.append("ellipsis_filt")
                    if lexical_tags[i]:
                        tag.append("lexical")
                    if len(tag) == 1:
                        tag.append("no_tag")
                    tag.append(posmorph_tags[i])
                    tags.append("+".join(tag))

            assert len(tags) == len(target.split(" "))
            print(" ".join(tags), file=output_file)

            source_context.append(source)
            target_context.append(target)
        print(f"corref_errors: {failed_coref}/{len(srcs)}")
        print(f"tagmismatch_errors: {tag_len_mismatch}/{len(srcs)}")


if __name__ == "__main__":
    main()
