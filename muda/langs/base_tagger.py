import abc


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
