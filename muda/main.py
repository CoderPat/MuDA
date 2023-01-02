import argparse
import json
from typing import Dict, Any, Callable

import os

from muda.langs import TAGGER_REGISTRY, create_tagger
from muda.metrics import compute_metrics


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    # base arguments
    parser.add_argument("--src", required=True, help="File with source sentences")
    parser.add_argument("--tgt", required=True, help="File with target sentences")
    parser.add_argument("--docids", required=True, help="File with document ids")
    parser.add_argument(
        "--hyps",
        nargs="*",
        default=[],
        help="One or more hypothesis files, to compare to the reference",
    )
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
        default=None,
        help="Cache directory to save awesome-align models",
    )

    parser.add_argument(
        "--cohesion-threshold",
        default=3,
        type=int,
        help="Threshold for number of (previous) occurances to be considered lexical cohesion."
        "Default: 3",
    )

    args = parser.parse_args()

    args_dict = vars(args)
    return args_dict


def recursive_map(func: Callable[[Any], Any], obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: recursive_map(func, v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_map(func, v) for v in obj]
    else:
        return func(obj)


def main(args: Dict[str, Any]) -> None:
    with open(args["src"], "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args["tgt"], "r", encoding="utf-8") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args["docids"], "r", encoding="utf-8") as docids_f:
        docids = [int(idx) for idx in docids_f]

    all_hyps = []
    for hyps in args["hyps"]:
        with open(hyps, "r", encoding="utf-8") as hyps_f:
            hyps = [line.strip() for line in hyps_f]
        all_hyps.append(hyps)

    if (
        args.get("awesome_align_cachedir") is None
        and os.environ.get("AWESOME_CACHEDIR") is not None
    ):
        args["awesome_align_cachedir"] = os.environ.get("AWESOME_CACHEDIR")

    tagger = create_tagger(
        args["tgt_lang"],
        align_model=args["awesome_align_model"],
        align_cachedir=args.get("awesome_align_cachedir"),
        cohesion_threshold=args["cohesion_threshold"],
    )

    preproc = tagger.preprocess(srcs, tgts, docids)

    tagged_refs = []
    for doc in zip(*preproc):
        tagged_doc = tagger.tag(*doc, phenomena=args["phenomena"])
        tagged_refs.append(tagged_doc)

    all_tagged_hyps = []
    for hyps in all_hyps:
        preproc = tagger.preprocess(srcs, hyps, docids)
        tagged_hyps = []
        for doc in zip(*preproc):
            tagged_doc = tagger.tag(*doc, phenomena=args["phenomena"])
            tagged_hyps.append(tagged_doc)
        all_tagged_hyps.append(tagged_hyps)

    if all_tagged_hyps:
        # compare f1 for each tag
        for tagged_hyps in all_tagged_hyps:
            tag_prec, tag_rec, tag_f1 = compute_metrics(tagged_refs, tagged_hyps)
            print("-- Hypothesis Set 1 --")
            for tag in tag_f1:
                print(
                    f"{tag} -- Prec: {tag_prec[tag]:.2f} Rec: {tag_rec[tag]:.2f} F1: {tag_f1[tag]:.2f}"
                )
            print()

    if args["dump_tags"]:
        with open(args["dump_tags"], "w", encoding="utf-8") as f:
            json.dump(recursive_map(lambda t: t._asdict(), tagged_refs), f, indent=2)


if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)
