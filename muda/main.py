import argparse
import json

from muda.langs import TAGGER_REGISTRY, create_tagger


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    # base arguments
    parser.add_argument("--src", required=True, help="File with source sentences")
    parser.add_argument("--tgt", required=True, help="File with target sentences")
    parser.add_argument("--docids", required=True, help="File with document ids")
    parser.add_argument(
        "--tgt-lang",
        required=True,
        choices=[x.replace("_tagger", "") for x in TAGGER_REGISTRY.keys()],
        help="Target language. Used to select the correct tagger.",
    )
    parser.add_argument("--max-ctx-size", type=int, default=None)
    parser.add_argument(
        "--phenomena",
        nargs="+",
        default=["lexical_cohesion", "formality", "verb_form", "pronouns"],
        help="Phenomena to tag. By default, all phenomena are tagged.",
    )
    parser.add_argument(
        "--dump-tags",
        required=True,  # This might change when MuDA has other functionalities
        help="If set, dumps the tags to the specified file.",
    )

    # aligner arguments
    parser.add_argument(
        "--awesome-align-model",
        default="bert-base-multilingual-cased",
        help="Awesome-align model to use. Default: bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--awesome-align-cachedir",
        default="/projects/tir5/users/patrick/awesome",
        help="Cache directory to save awesome-align models",
    )

    args = parser.parse_args()

    args_dict = vars(args)
    return args_dict


def main(args: dict) -> None:
    with open(args["src"], "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args["tgt"], "r", encoding="utf-8") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args["docids"], "r", encoding="utf-8") as docids_f:
        docids = [int(idx) for idx in docids_f]

    tagger = create_tagger(
        args["tgt_lang"],
        align_model=args["awesome_align_model"],
        align_cachedir=args["awesome_align_cachedir"],
    )

    preproc = tagger.preprocess(srcs, tgts, docids)

    tagged_docs = []
    for doc in zip(*preproc):
        tagged_doc = tagger.tag(*doc, phenomena=args["phenomena"])
        tagged_docs.append(tagged_doc)

    if args["dump_tags"]:
        with open(args["dump_tags"], "w", encoding="utf-8") as f:
            json.dump(tagged_docs, f, indent=2)


if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)
