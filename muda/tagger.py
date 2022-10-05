from typing import List, Dict, Tuple, Optional
import argparse
from collections import defaultdict
from allennlp.predictors.predictor import Predictor  # type: ignore
import spacy
import spacy_stanza # type: ignore
import tempfile
import subprocess
import inspect


from langs import TAGGER_REGISTRY, create_tagger


def build_alignments(
    src_pproc: List[List[spacy.tokens.Token]],
    tgt_pproc: List[List[spacy.tokens.Token]],
    model: str,
    cache_dir: str,
) -> List[Dict[int, int]]:
    """Builds alignments between source and target sentences."""

    data_inf = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
    for src, tgt in zip(src_pproc, tgt_pproc):
        src_tks = " ".join([x.text for x in src])
        tgt_tks = " ".join([x.text for x in tgt])
        if len(src_tks.strip()) > 0 and len(tgt_tks.strip()) > 0:
            data_inf.write(f"{src_tks.strip()} ||| {tgt_tks.strip()}\n")
        elif len(tgt_tks.strip()) > 0:
            data_inf.write(f"{src_tks.strip()} ||| <blank>\n")
        else:
            data_inf.write("<blank> ||| <blank>\n")
    data_inf.flush()

    # we run it using subprocess because it the python library is not very easy to use
    alignment_outf = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
    subproc = subprocess.Popen(
        [
            "awesome-align",
            "--output_file",
            alignment_outf.name,
            "--model_name_or_path",
            model,
            "--data_file",
            data_inf.name,
            "--extraction",
            "softmax",
            "--batch_size",
            "32",
            "--cache_dir",
            cache_dir,
        ]
    )
    # TODO: check if subproc exited successfully
    subproc.wait()

    with open(alignment_outf.name, "r", encoding="utf-8") as alignment_readf:
        # TODO: refactor this to be more readable
        alignments = list(
            map(
                lambda x: dict(
                    list(
                        map(
                            lambda y: list(map(int, y.split("-"))), x.strip().split(" ")
                        )
                    )
                ),
                alignment_readf.readlines(),
            )
        )
    return alignments


def build_corefs(src_pproc: List[spacy.tokens.Token]) -> List[List[bool]]:
    """Builds coreference chains for the source (english) sentences."""
    # this is done in order to know which ambiguous pronoun need context to be resolved
    # TODO: encapsulate this as part of the tagger?
    en_coref = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )
    antecs = []
    coref_errors = 0
    for src in src_pproc:
        has_antec = [False] * len(src)
        try:
            coref = en_coref.predict(document=src.text)
            if len(src) != len(coref["document"]):
                raise ValueError()

            for cluster in coref["clusters"]:
                for mention in cluster[1:]:
                    for i in range(mention[0], mention[1] + 1):
                        has_antec[i] = True

        # sometimes tokenizers are not consistent, or some other error happens in the coreference resolution
        # in that case we just ignore the coref assuming it has no antencedents (might lead to some false positives)
        except (IndexError, ValueError):
            coref_errors += 1

        antecs.append(has_antec)
    return antecs


def build_docs(docids: List[int], *args, max_ctx_size: Optional[int] = None) -> Tuple:
    """Builds "document-level" structures based on docids and sentence-level structures."""
    prev_docid = None
    all_docs = []
    for docid, *elements in zip(docids, *args):
        # if first sentence of a new document, reset context
        if prev_docid is None or docid != prev_docid:
            if prev_docid is not None:
                all_docs.append(list(zip(*doc_elements)))
            doc_elements = []
            prev_docid = docid

        doc_elements.append(elements)
    return tuple(zip(*all_docs))


def main() -> None:
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

    with open(args.src, "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.tgt, "r", encoding="utf-8") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args.docids, "r", encoding="utf-8") as docids_f:
        docids = [int(idx) for idx in docids_f]

    # create taggers
    # currently src language is fixed to English
    src_tagger = create_tagger("en")
    tgt_tagger = create_tagger(args.tgt_lang)

    # TODO: encapsulate the 'pipe' method
    src_pproc = list(src_tagger.pipeline.pipe(srcs))
    tgt_pproc = list(tgt_tagger.pipeline.pipe(tgts))

    # build extra information, such as alignments and coref chains
    alignments = build_alignments(
        src_pproc,
        tgt_pproc,
        model=args.awesome_align_model,
        cache_dir=args.awesome_align_cachedir,
    )
    antecs = build_corefs(src_pproc)

    src_docs_tagged, tgt_docs_tagged, antecs_docs, align_docs = build_docs(
        docids, src_pproc, tgt_pproc, antecs, alignments, max_ctx_size=args.max_ctx_size
    )

    tagged_docs = []
    for src_doc, tgt_doc, antecs_doc, align_doc in zip(
        src_docs_tagged, tgt_docs_tagged, antecs_docs, align_docs
    ):
        # To avoid having to specify the arguments needed for each phenomena, we use a dictionary
        # with all the arguments needed for each possible phenomena
        kwargs = {
            "src_doc": src_doc,
            "tgt_doc": tgt_doc,
            "antecs_doc": antecs_doc,
            "align_doc": align_doc,
        }
        tagged_doc = [[set() for _ in tgt] for tgt in tgt_doc]
        for phenomenon in args.phenomena:
            assert hasattr(tgt_tagger, phenomenon), "Phenomenon doesn't exist"
            phenomenon_fn = getattr(tgt_tagger, phenomenon)
            # we call it with the arguments it needs
            tags = phenomenon_fn(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in inspect.signature(phenomenon_fn).parameters
                }
            )
            for i, sent_tags in enumerate(tags):
                assert len(sent_tags) == len(tagged_doc[i])
                for j, tag in enumerate(sent_tags):
                    if tag:
                        tagged_doc[i][j].add(tag)
        tagged_docs.append(tagged_doc)
    
    # TODO: what to do with the tagged docs?

if __name__ == "__main__":
    main()
