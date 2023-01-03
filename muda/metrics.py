from typing import List, Dict, Tuple

from collections import defaultdict
from itertools import chain

from muda.tagger import Tagger, Tagging


def compute_metrics(
    tagged_refs: List[List[List[Tagging]]], tagged_hyps: List[List[List[Tagging]]]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Computes the accuracy, recall and f1 for each tag based if word tagged in the
    reference/hypothesis exist and are also tagged in the hypothesis/reference.

    Inspired by the compare-mt's LabelWordBucketer:
    https://github.com/neulab/compare-mt/blob/master/compare_mt/bucketers.py"""
    flatten_refs = chain.from_iterable(tagged_refs)
    flatten_hyps = chain.from_iterable(tagged_hyps)

    tagref_matches: Dict[str, int] = defaultdict(int)
    tagref_total: Dict[str, int] = defaultdict(int)
    taghyp_matches: Dict[str, int] = defaultdict(int)
    taghyp_total: Dict[str, int] = defaultdict(int)

    for tgt_sent, hyp_sent in zip(flatten_refs, flatten_hyps):
        # get position of words in the reference
        # and mark the tags that appear in the reference
        ref_pos = defaultdict(list)
        for i, tagging in enumerate(tgt_sent):
            word = Tagger.normalize(tagging.token)
            ref_pos[word].append(i)
            for tag in tagging.tags:
                tagref_total[tag] += 1

        word_count: Dict[str, int] = defaultdict(int)
        for i, tagging in enumerate(hyp_sent):
            word = Tagger.normalize(tagging.token)
            for tag in tagging.tags:
                taghyp_total[tag] += 1

            if word in ref_pos and word_count[word] < len(ref_pos[word]):
                # mark intersection of tags in the reference and hypothesis
                for tag in tagging.tags:
                    if tag in tgt_sent[ref_pos[word][word_count[word]]].tags:
                        tagref_matches[tag] += 1
                        taghyp_matches[tag] += 1

            word_count[word] += 1

    prec: Dict[str, float] = defaultdict(float)
    rec: Dict[str, float] = defaultdict(float)
    prec.update({tag: taghyp_matches[tag] / taghyp_total[tag] for tag in taghyp_total})
    rec.update({tag: tagref_matches[tag] / tagref_total[tag] for tag in tagref_total})
    all_tags = set(tagref_total.keys()).union(set(taghyp_total.keys()))
    f1 = {
        tag: 2 * prec[tag] * rec[tag] / max(prec[tag] + rec[tag], 1e-20)
        for tag in all_tags
    }
    return prec, rec, f1
