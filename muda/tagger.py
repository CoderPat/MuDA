import abc
import inspect
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional

import spacy
import spacy_stanza  # type: ignore
from allennlp.predictors.predictor import Predictor

Document = List[spacy.tokens.doc.Doc]
Alignment = List[Dict[int, int]]
Antecs = List[List[bool]]


def build_docs(docids, *args):  # type: ignore
    """Builds "document-level" structures based on docids and sentence-level structures."""
    assert all(
        len(x) == len(docids) for x in args
    ), "all arguments must have the same length"

    prev_docid = None
    doc_elements: List[Any] = []
    all_docs: List[List[Any]] = []

    for docid, *elements in zip(docids, *args):
        # if first sentence of a new document, reset context
        if prev_docid is None or docid != prev_docid:
            if prev_docid is not None:
                all_docs.append(list(zip(*doc_elements)))
            doc_elements = []
            prev_docid = docid

        doc_elements.append(elements)
    all_docs.append(list(zip(*doc_elements)))
    return tuple(zip(*all_docs))


class Tagger(abc.ABC):
    """
    Abstact class that represent a tagger for a (target) language.
    It implements the core preprocessing and tagging functionality.
    Subclasses need to override the __init__ to set language-specific parameters and,
    if applicable, implement the _verb_formality method.
    """

    def __init__(
        self,
        align_model: str = "bert-base-multilingual-cased",
        align_cachedir: Optional[str] = None,
    ) -> None:
        """Initializes the tagger, loading the necessary models."""
        self.src_pipeline = spacy_stanza.load_pipeline(
            "en", processors="tokenize,pos,lemma,depparse"
        )

        # override this in subclasses
        self.tgt_pipeline: spacy.language.Language
        self.formality_classes: Dict[str, Set[str]] = {}
        self.ambiguous_pronouns: Dict[str, List[str]] = {}
        self.ambiguous_verbform: List[str] = []

        self.align_model = align_model
        self.align_cachedir = align_cachedir

    def _normalize(self, word: str) -> str:
        """default normalization"""
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def preprocess(
        self, srcs: List[str], tgts: List[str], docids: List[int]
    ) -> Tuple[List[Document], List[Document], List[Antecs], List[Alignment]]:
        """
        Preprocesses a list of source and target sentences, creating a document-level
        structures necessary for tagging.

        Args:
            srcs: list of source sentences
            tgts: list of target sentences
            docids: list of document ids, mapping each sentence to a document
        Returns:
            src_docs: list of source documents, each document is a list of sentences,
                each sentence is a list of tokens
            tgt_docs: list of target documents, ...
            antecs_docs: list of document antecedent markers, where the antecend marker
                for every sentence in the document is a list of booleans specifying if
                each token has an antecedent in the current sentence
            align_docs: list of document alignments, where the alignment for every
                sentence in the document is a dictionary mapping source token indices
                to target token indices
        """
        src_pproc = list(self.src_pipeline.pipe(srcs))
        tgt_pproc = list(self.tgt_pipeline.pipe(tgts))

        # build extra information, such as alignments and coref chains
        alignments = self._build_alignments(src_pproc, tgt_pproc)
        antecs = self._build_corefs(src_pproc, docids)

        return build_docs(docids, src_pproc, tgt_pproc, antecs, alignments)  # type: ignore

    def tag(
        self,
        src_doc: Document,
        tgt_doc: Document,
        antecs_doc: Antecs,
        align_doc: Alignment,
        phenomena: List[str] = [
            "lexical_cohesion",
            "formality",
            "verb_form",
            "pronouns",
        ],
    ) -> List[List[List[str]]]:
        """Tags a src-tgt document pair, returning the tags associated with each token
        in each sentence of the target document.

        Args:
            src_doc: list of source sentences, each sentence is a list of tokens
            tgt_doc: list of target sentences, each sentence is a list of tokens
            antecs_doc: list of coref chains, each chain is a list of booleans
            align_doc: list of alignments, each alignment is a
                dictionary mapping source token indices to target token indices
            phenomena: list of phenomena to tag
        Returns:
            list of list of list of tags, with the tags for each token in each sentence
                in the target document
        """
        kwargs = {
            "src_doc": src_doc,
            "tgt_doc": tgt_doc,
            "antecs_doc": antecs_doc,
            "align_doc": align_doc,
        }
        tagged_doc: List[List[List[str]]] = [[[] for _ in tgt] for tgt in tgt_doc]
        for phenomenon in phenomena:
            assert hasattr(self, phenomenon), "Phenomenon doesn't exist"
            phenomenon_fn = getattr(self, phenomenon)
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
                        tagged_doc[i][j].append(phenomenon)
        return tagged_doc

    def _build_corefs(
        self, src_pproc: List[spacy.tokens.doc.Doc], docids: List[int]
    ) -> List[List[bool]]:
        """Builds coreference chains for the source (english) sentences."""
        # this is done in order to know which ambiguous pronoun need context to be resolved
        # TODO: encapsulate this as part of the tagger?
        en_coref = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
        )
        antecs = []
        coref_errors = 0
        prev_docid = None
        for src, docid in zip(src_pproc, docids):
            # we check if this is the first sentence of a new document
            # since in this case there is no context that could help
            if docid != prev_docid:
                has_antec = [True] * len(src)
            else:
                has_antec = [False] * len(src)
                try:
                    coref = en_coref.predict(document=src.text) # type: ignore
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
                    print("coref error")

            antecs.append(has_antec)
            prev_docid = docid
        return antecs

    def _build_alignments(
        self,
        src_pproc: List[spacy.tokens.doc.Doc],
        tgt_pproc: List[spacy.tokens.doc.Doc],
    ) -> List[Dict[int, int]]:
        """Builds alignments between source and target sentences."""
        # TODO: refactor this to use HFs rather than subprocess
        data_inf = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
        for i, (src, tgt) in enumerate(zip(src_pproc, tgt_pproc)):
            if i:
                data_inf.write("\n")
            src_tks = " ".join([x.text for x in src])
            tgt_tks = " ".join([x.text for x in tgt])
            if len(src_tks.strip()) > 0 and len(tgt_tks.strip()) > 0:
                data_inf.write(f"{src_tks.strip()} ||| {tgt_tks.strip()}")
            elif len(tgt_tks.strip()) > 0:
                data_inf.write(f"{src_tks.strip()} ||| <blank>")
            else:
                data_inf.write("<blank> ||| <blank>")
        data_inf.flush()

        # we run it using subprocess because it the python library is not very easy to use
        alignment_outf = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")

        extra_args = []
        if self.align_cachedir is not None:
            extra_args.extend(["--cache_dir", self.align_cachedir])

        command = [
            "awesome-align",
            "--output_file",
            alignment_outf.name,
            "--model_name_or_path",
            self.align_model,
            "--data_file",
            data_inf.name,
            "--extraction",
            "softmax",
            "--batch_size",
            "32",
            *extra_args,
        ]
        subproc = subprocess.Popen(
            command,
        )

        # TODO: check if subproc exited successfully
        subproc.wait()

        with open(alignment_outf.name, "r", encoding="utf-8") as alignment_readf:
            # TODO: refactor this to be more readable
            alignments = []
            for alignment_str in alignment_readf.readlines():
                alignment = {}
                for pair in alignment_str.strip().split(" "):
                    src_idx, tgt_idx = pair.split("-")
                    alignment[int(src_idx)] = int(tgt_idx)
                alignments.append(alignment)

        # For some reason, sometimes awesome-align outputs an extra alignment,
        # which is a copy of the last one. In this case, we remove it.
        if len(alignments) != len(src_pproc):
            if len(alignments) == len(src_pproc) + 1:
                alignments = alignments[:-1]
            else:
                raise ValueError("Alignment length mismatch")

        return alignments

    def formality(
        self,
        src_doc: Document,
        tgt_doc: Document,
        align_doc: Alignment,
    ) -> List[List[bool]]:
        """Checks a (preprocessed) document for formality-related (e.g. pronouns, verb forms, etc.)
        that require context to be disambiguated.

        Args:
            src_doc: list of list of spacy tokens
            tgt_doc: list of list of spacy tokens
            align_doc: list of alignment dictionaries, mapping source to target tokens
        Returns:
            list of list of bools indicating if a given token is formal
        """
        doc_tags = []
        formality_classes = {
            word: formality
            for formality, words in self.formality_classes.items()
            for word in words
        }
        formality_words = list(formality_classes.keys())
        prev_formality = set()
        # import pdb; pdb.set_trace()
        for src, tgt, align in zip(src_doc, tgt_doc, align_doc):
            tags = []
            for word in tgt:
                norm_word = self._normalize(word.text)
                if norm_word in formality_words:
                    # if a formality-related word is found, tag it if has appeared before
                    if formality_classes[norm_word] in prev_formality:
                        tags.append(True)
                    # otherwise record that this formality class has appeared
                    else:
                        tags.append(False)
                        prev_formality.add(formality_classes[norm_word])
                else:
                    tags.append(False)

                # if the subclasses implements a verb formality check, use it
                try:
                    verb_tags = self._verb_formality(src, tgt, align, prev_formality)
                    assert len(tags) == len(verb_tags)
                    tags = [a or b for a, b in zip(tags, verb_tags)]
                except NotImplementedError:
                    pass

            doc_tags.append(tags)

        return doc_tags

    def _verb_formality(
        self,
        src_sent: spacy.tokens.doc.Doc,
        tgt_sent: spacy.tokens.doc.Doc,
        align_sent: Dict[int, int],
        prev_formality: Set[str],
    ) -> List[bool]:
        """Checks a (preprocessed) sentence for formality-related verbs forms that
        require context to be disambiguated. Needs to be implemented by subclasses for language-specific ruls

        Args:
            src_sent: list of spacy tokens
            tgt_doc: list of spacy tokens
            align: list of alignment dictionaries, mapping source to target tokens
        Returns:
            list of list of bools indicating if a given token is formal
        """
        raise NotImplementedError("")

    def verb_form(self, tgt_doc: Document) -> List[List[bool]]:
        """TODO: add documentation"""
        doc_tags = []
        verb_forms = set()
        for tgt in tgt_doc:
            tags = []
            for tok in tgt:
                tag = False
                if tok.pos_ == "VERB":
                    amb_verb_forms = [
                        a
                        for a in tok.morph.get("Tense")
                        if a in self.ambiguous_verbform
                    ]
                    for form in set(amb_verb_forms):
                        if form in verb_forms:
                            tag = True  # Set tag to true if ambiguous form appeared before
                        else:
                            verb_forms.add(form)  # Add ambiguous form to memory

                tags.append(tag)

            doc_tags.append(tags)

        return doc_tags

    def lexical_cohesion(
        self,
        src_doc: Document,
        tgt_doc: Document,
        align_doc: Alignment,
        cohesion_threshold: int = 2,
    ) -> List[List[bool]]:
        """TODO: add documentation"""
        doc_tags = []
        cohesion_words: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        for src, tgt, align in zip(src_doc, tgt_doc, align_doc):
            tags = [False] * len(tgt)

            # get non-stopwords
            # TODO: check if we still need `tok.text.split(" ")` or why it was added
            src_lemmas = [
                t if not tok.is_stop and not tok.is_punct else None
                for tok in src
                for t in tok.text.split(" ")
            ]
            lemmas_idx, tgt_lemmas = zip(
                *[
                    (i, t if not tok.is_stop and not tok.is_punct else None)
                    for i, tok in enumerate(tgt)
                    for t in tok.text.split(" ")
                ]
            )

            tmp_cohesion_words: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(lambda: 0)
            )
            for s, t in align.items():
                src_lemma = src_lemmas[s]
                tgt_lemma = tgt_lemmas[t]
                # for every aligned src-tgt word, check if it has appear more than
                # `cohesion_threshold` times in previous sentences
                # and update the temporary cohesion words dictionary
                if src_lemma is not None and tgt_lemma is not None:
                    if cohesion_words[src_lemma][tgt_lemma] > cohesion_threshold:
                        tags[lemmas_idx[t]] = True
                    tmp_cohesion_words[src_lemma][tgt_lemma] += 1

            # update global cohesion words with the temporary dictionary
            for src_lemma in tmp_cohesion_words.keys():
                for tgt_lemma in tmp_cohesion_words[src_lemma].keys():
                    cohesion_words[src_lemma][tgt_lemma] += tmp_cohesion_words[
                        src_lemma
                    ][tgt_lemma]

            doc_tags.append(tags)

        return doc_tags

    def pronouns(
        self,
        src_doc: Document,
        tgt_doc: Document,
        align_doc: Alignment,
        antecs_doc: Antecs,
    ) -> List[List[bool]]:
        """TODO: add documentation"""
        doc_tags = []
        for src, tgt, align, antecs in zip(src_doc, tgt_doc, align_doc, antecs_doc):
            tags = [False] * len(tgt)
            if not self.ambiguous_pronouns:
                doc_tags.append(tags)
                continue

            src_text, src_pos = zip(
                *[
                    (tok.text, tok.pos_) if not tok.is_punct else (None, None)
                    for tok in src
                    for _ in tok.text.split(" ")
                ]
            )
            tgt_idx, tgt_text, tgt_pos = zip(
                *[
                    (i, *((tok.text, tok.pos_) if not tok.is_punct else (None, None)))
                    for i, tok in enumerate(tgt)
                    for _ in tok.text.split(" ")
                ]
            )

            for s, r in align.items():
                if s > len(src_text):
                    print(f"IndexError{s}: {src_text}")
                if r > len(tgt_text):
                    print(f"IndexError{r}: {tgt_text}")
                if (
                    not antecs[s]
                    and src_pos[s] == "PRON"
                    and tgt_pos[r] == "PRON"
                    and self._normalize(tgt_text[r])
                    in self.ambiguous_pronouns.get(self._normalize(src_text[s]), [])
                ):
                    tags[tgt_idx[r]] = True
            doc_tags.append(tags)

        return doc_tags
