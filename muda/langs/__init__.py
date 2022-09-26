import abc
from collections import defaultdict
import re
import os
import importlib
from typing import Callable

TAGGER_REGISTRY = {}


class Tagger(abc.ABC):
    """Abstact class that represent a tagger for a language"""

    def __init__(self):
        # self.tagger = spacy.load("xx_ent_wiki_sm")
        self.formality_classes = {}
        self.ambiguous_pronouns = None
        self.ambiguous_verbform = []

    def _normalize(self, word):
        """default normalization"""
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def formality_tags(
        self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags
    ):
        # TODO: inter-sentential especification needs to be added
        # this would go by checking if the formality already appeared in the context
        # by for example, passing a set of seen formalities in the previsous sentences
        # similar to what happens in lexical cohesion
        # NOTE: every language specific verb formality checker will have to do this aswell
        tags = []
        # formality_words = [v for vs in self.formality_classes.values() for v in vs]
        formality_classes = {
            word: formality
            for formality, words in self.formality_classes.items()
            for word in words
        }
        formality_words = list(formality_classes.keys())
        for word in cur_tgt.split(" "):
            word = self._normalize(word)
            if word in formality_words:
                if formality_classes[word] in prev_formality_tags:
                    tags.append(True)
                else:
                    tags.append(False)
                    prev_formality_tags.add(formality_classes[word])
            else:
                tags.append(False)
        assert len(tags) == len(cur_tgt.split(" "))

        try:
            tags2, prev_formality_tags = self.verb_formality(
                cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align
            )
            assert len(tags2) == len(cur_tgt.split(" "))
            return [a or b for a, b in zip(tags, tags2)], prev_formality_tags
        except:  # noqa: E722
            return tags, prev_formality_tags

    def lexical_cohesion(self, target, src_doc, tgt_doc, align, cohesion_words):
        src_lemmas = [
            t if not tok.is_stop and not tok.is_punct else None
            for tok in src_doc
            for t in tok.text.split(" ")
        ]
        tgt_lemmas = [
            t if not tok.is_stop and not tok.is_punct else None
            for tok in tgt_doc
            for t in tok.text.split(" ")
        ]
        tags = [False] * len(tgt_lemmas)

        try:
            tmp_cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))
            for s, t in align.items():
                src_lemma = src_lemmas[s]
                tgt_lemma = tgt_lemmas[t]
                if src_lemma is not None and tgt_lemma is not None:
                    if cohesion_words[src_lemma][tgt_lemma] > 2:
                        tags[t] = True
                    tmp_cohesion_words[src_lemma][tgt_lemma] += 1

            for src_lemma in tmp_cohesion_words.keys():
                for tgt_lemma in tmp_cohesion_words[src_lemma].keys():
                    cohesion_words[src_lemma][tgt_lemma] += tmp_cohesion_words[
                        src_lemma
                    ][tgt_lemma]
        except IndexError:
            print("cohesion error")
            return [False] * len(target.split(" ")), cohesion_words

        return tags, cohesion_words

    def verb_form(self, cur_doc, verb_forms):
        # TODO: inter-sentential especification needs to be added
        # this would go by checking if a specific verb_form already appeared in the context
        # by for example, passing a set of seen verb_forms in the previsous sentences
        # similar to what happens in lexical cohesion
        # NOTE: every language specific verb formality checker will have to do this aswell
        tags = []
        for tok in cur_doc:
            tag = False
            if tok.pos_ == "VERB":
                amb_verb_forms = [
                    a for a in tok.morph.get("Tense") if a in self.ambiguous_verbform
                ]
                for form in set(amb_verb_forms):
                    if form in verb_forms:
                        tag = True  # Set tag to true if ambiguous form appeared before
                    else:
                        verb_forms.add(form)  # Add ambiguous form to memory
            for _ in tok.text.split(" "):
                tags.append(tag)
        return tags, verb_forms

    def pronouns(self, src_doc, tgt_doc, align, has_ante):
        # TODO: inter-sentential especification needs to be added
        # this would go by adding a coreference resolution that would
        # check if the coreferent is part of the context rather than the current sentence
        src = [
            tok.text if not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        src_pos = [
            tok.pos_ if not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        tgt = [
            tok.text if not tok.is_punct else None
            for tok in tgt_doc
            for _ in tok.text.split(" ")
        ]
        tgt_pos = [
            tok.pos_ if not tok.is_punct else None
            for tok in tgt_doc
            for _ in tok.text.split(" ")
        ]
        tags = [False] * len(tgt)
        # if self.src_neutral_pronouns is None or self.tgt_gendered_pronouns is None:
        if self.ambiguous_pronouns is None:
            return tags
        try:
            for s, r in align.items():
                # if self._normalize(src[s]) in self.src_neutral_pronouns:
                #     if self._normalize(ref[r]) in self.tgt_gendered_pronouns:
                if s > len(src):
                    print(f"IndexError{s}: {src}")
                if r > len(tgt):
                    print(f"IndexError{r}: {tgt}")
                if (
                    not has_ante[s]
                    and src_pos[s] == "PRON"
                    and tgt_pos[r] == "PRON"
                    and self._normalize(tgt[r])
                    in self.ambiguous_pronouns.get(self._normalize(src[s]), [])
                ):
                    tags[r] = True
        except IndexError:
            print("pronoun error")
            return [False] * len(tgt)

        return tags

    def ellipsis(self, src, ref, align, verbs, nouns, ellipsis_sent):
        # src_pos = "+".join([tok.pos_ for tok in src for _ in tok.text.split(" ")])
        # src_text = [
        #     self._normalize(tok.text) for tok in src for _ in tok.text.split(" ")
        # ]
        ref = [tok for tok in ref for _ in tok.text.split(" ")]
        tags = [False] * len(ref)

        for i, tok in enumerate(ref):
            if (
                not tok.is_stop
                and tok.pos_ == "VERB"
                and len(self._normalize(tok.text)) > 1
            ):  # VP ellipsis
                # if i not in align.values() and (("AUX" in src_pos and "AUX+VERB" not in src_pos) or "to" in src_text) and tok.lemma_ in verbs:
                if i not in align.values() and tok.lemma_ in verbs and ellipsis_sent:
                    tags[i] = True
                verbs.add(tok.lemma_)
            if (
                not tok.is_stop
                and tok.pos_ in ["PRON", "PROPN", "NOUN"]
                and len(self._normalize(tok.text)) > 1
            ):  # NP classifier ellipsis
                # if i not in align.values() and ref[i-1].pos_ == "NUM" and "NUM" in src_pos:
                if i not in align.values() and tok.lemma_ in nouns and ellipsis_sent:
                    tags[i] = True
                nouns.add(tok.lemma_)
        return tags, verbs, nouns

    def pos_morph(self, current, doc):
        tags = []
        for tok in doc:
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
            for _ in tok.text.split(" "):
                tags.append(tag)

        if len(tags) != len(current.split(" ")):
            print("pos_morph error")
            return ["no_tag"] * len(current.split(" "))

        return tags

    def polysemous(self, src_doc, target, align, polysemous_words):
        src_lemmas = [
            tok.lemma_ if not tok.is_stop and not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        src_poss = [
            tok.pos_ if not tok.is_stop and not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        tags = [False] * len(target.split(" "))

        for s, t in align.items():
            if f"{src_lemmas[s]}.{src_poss[s]}" in polysemous_words:
                tags[t] = True

        return tags


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
            print(tagger_name)
            importlib.import_module(namespace + "." + tagger_name)


def create_tagger(langcode: str) -> Tagger:
    # standardize tagger name by langcode
    tagger_name = f"{langcode}_tagger"
    tagger = TAGGER_REGISTRY[tagger_name]()
    return tagger


langdir = os.path.dirname(__file__)
import_taggers(langdir, "langs")
