import argparse
from collections import defaultdict
from allennlp.predictors.predictor import Predictor  # type: ignore
import spacy_stanza
import tempfile
import subprocess


from langs import TAGGER_REGISTRY, create_tagger


def build_alignments(src_pproc, tgt_pproc, model, cache_dir):
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
            args.awesome_align_model,
            "--data_file",
            data_inf.name,
            "--extraction",
            "softmax",
            "--batch_size",
            "32",
            "--cache_dir",
            args.awesome_align_cachedir,
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


def build_corefs(src_pproc):
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


def build_docs(docids, *args, max_ctx_size=None):
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


def main():
    parser = argparse.ArgumentParser()

    # base arguments
    parser.add_argument("--src", required=True, help="File with source sentences")
    parser.add_argument("--tgt", required=True, help="File with target sentences")
    parser.add_argument("--docids", required=True, help="File with document ids")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--tgt-lang",
        required=True,
        choices=[x.replace("_tagger", "") for x in TAGGER_REGISTRY.keys()],
    )
    parser.add_argument("--max-ctx-size", type=int, default=None)
    parser.add_argument(
        "--phenomena",
        nargs="+",
        default=["lexical_cohesion", "formality", "verb_form", "pronouns", "pos_morph"],
    )

    # aligner arguments
    parser.add_argument(
        "--awesome-align-model",
        default="bert-base-multilingual-cased",
        help="Model to use for src-tgt alignment",
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
        docids = [idx for idx in docids_f]

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

    import pdb

    pdb.set_trace()

    tag_len_mismatch = 0
    with open(args.output, "w", encoding="utf-8") as output_file:
        for src_doc, tgt_doc, antecs_doc, align_doc in zip(
            src_docs_tagged, tgt_docs_tagged, antecs_docs, align_docs
        ):
            kwargs = {
                "src_doc": src_doc,
                "tgt_doc": tgt_doc,
                "antecs_doc": antecs_doc,
                "align_doc": align_doc,
            }
            all_tags = []
            for phenomenon in args.phenomena:
                assert hasattr(tgt_tagger, phenomenon)
                phenomenon_fn = getattr(tgt_tagger, phenomenon)
                all_tags.append(
                    phenomenon(
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k in inspect.signature(phenomenon_fn).parameters
                        }
                    )
                )
            import idpb

            ipdb.set_trace()

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

        print(f"corref_errors: {failed_coref}/{len(srcs)}")
        print(f"tagmismatch_errors: {tag_len_mismatch}/{len(srcs)}")


if __name__ == "__main__":
    main()
