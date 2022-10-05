from typing import List, Dict, Set
import abc
from collections import defaultdict
import re
import os
import importlib
from typing import Callable
import spacy

TAGGER_REGISTRY = {}


class Tagger(abc.ABC):
    """Abstact class that represent a tagger for a (target) language"""

    def __init__(self) -> None:
        self.formality_classes = {}
        self.ambiguous_pronouns = None
        self.ambiguous_verbform = []

    def _normalize(self, word: str) -> str:
        """default normalization"""
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def formality(
        self,
        src_doc: List[List[spacy.tokens.Token]],
        tgt_doc: List[List[spacy.tokens.Token]],
        align_doc: List[Dict[int, int]],
    ) -> List[List[bool]]:
        """TODO: add documentation"""
        doc_tags = []
        formality_classes = {
            word: formality
            for formality, words in self.formality_classes.items()
            for word in words
        }
        formality_words = list(formality_classes.keys())
        prev_formality = set()
        for src, tgt, align in zip(src_doc, tgt_doc, align_doc):
            tags = []
            for word in tgt:
                if word.text in formality_words:
                    # if a formality-related word is found, tag it if has appeared before
                    if formality_classes[word.text] in prev_formality:
                        tags.append(True)
                    # otherwise record that this formality class has appeared
                    else:
                        tags.append(False)
                        prev_formality.add(formality_classes[word.text])
                else:
                    tags.append(False)

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
        src_sent: List[spacy.tokens.Token],
        tgt_sent: List[spacy.tokens.Token],
        align: Dict[int, int],
        prev_formality: Set[str],
    ) -> List[bool]:
        """TODO: add documentation"""
        raise NotImplementedError()

    def verb_form(self, tgt_doc: List[List[spacy.tokens.Token]]) -> List[List[bool]]:
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
        src_doc: List[List[spacy.tokens.Token]],
        tgt_doc: List[List[spacy.tokens.Token]],
        align_doc: List[List[spacy.tokens.Token]],
        cohesion_threshold: int = 2,
    ) -> List[List[bool]]:
        """TODO: add documentation"""
        doc_tags = []
        cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))
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

            tmp_cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))
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
        src_doc: List[List[spacy.tokens.Token]],
        tgt_doc: List[List[spacy.tokens.Token]],
        align_doc: List[Dict[int, int]],
        antecs_doc: List[List[bool]],
    ) -> List[List[bool]]:
        """TODO: add documentation"""
        doc_tags = []
        for src, tgt, align, antecs in zip(src_doc, tgt_doc, align_doc, antecs_doc):
            tags = [False] * len(tgt)
            if self.ambiguous_pronouns is None:
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

    def pos_morph(self, tgt_doc):
        """TODO: documentation"""
        doc_tags = []
        for tgt in tgt_doc:
            tags = []
            for tok in tgt:
                tag = tok.pos_
                if tag == "PRON":
                    morph_tags = tok.morph.get("Person") + tok.morph.get("Number")
                    for i in range(len(morph_tags)):
                        m_tag = ".".join(morph_tags[: i + 1])
                        tag += "+" + "PRON." + m_tag
                elif tag == "VERB":
                    m_tag = tok.morph.get("Tense")
                    if len(m_tag) > 0:
                        tag += "+" + "VERB." + ".".join(m_tag)

                tags.append(tag)

            doc_tags.append(tags)
        return doc_tags


def register_tagger(tagger_name: str) -> Callable:
    tagger_name = tagger_name.lower()

    def register_tagger_cls(cls):
        if tagger_name in TAGGER_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(tagger_name))
        if not issubclass(cls, Tagger):
            raise ValueError(
                "Model ({}: {}) must extend Tagger".format(tagger_name, cls.__name__)
            )

        TAGGER_REGISTRY[tagger_name] = cls

    return register_tagger_cls


def import_taggers(langdir: str, namespace: str) -> None:
    for file in os.listdir(langdir):
        path = os.path.join(langdir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            tagger_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + tagger_name)


def create_tagger(langcode: str) -> Tagger:
    # standardize tagger name by langcode
    tagger_name = f"{langcode}_tagger"
    tagger = TAGGER_REGISTRY[tagger_name]()
    return tagger


langdir = os.path.dirname(__file__)
import_taggers(langdir, "langs")
