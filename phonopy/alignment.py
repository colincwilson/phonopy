# Alignment (including self-alignment) and identification of contextual edits
# of strings or other sequences, possibly weighted by feature similarity.
# todo: add other (self-)alignment methods, see wyfst/extra
import re, sys
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pprint import pprint

#import panphon
sys.path.append("/home/colin/Library/Python/string2string")
from string2string.alignment import NeedlemanWunsch, SmithWaterman

from phonopy import config as phon_config
from phonopy import (features, strings)

fm = features.read_features()

# # # # # # # # # #


class Aligner():
    """
    Aligner for pairs of sequences (strings / lists / tuples) using the
    Needleman-Wunsch algorithm.
    """

    def __init__(self, match_func=None, epsilon=phon_config.epsilon):
        if match_func is None:
            match_func = self.match_func_default
        self.match_func = match_func
        self.epsilon = epsilon
        self.aligner = NeedlemanWunsch(match_func=match_func, gap_char=epsilon)

    def match_func_default(self, seg1, seg2):
        """
        Default match function returning segment similarity.
        """
        return features.segment_similarity(fm, seg1, seg2)

    def get_alignment(self, seq1, seq2, sep=" ", return_score_matrix=True):
        """
        Return minimum-cost alignment and contextual edits for a
        pair of sequences (strings / lists / tuples).
        """
        if isinstance(seq1, str):
            seq1 = strings.str_list(seq1, sep=sep)
        if isinstance(seq2, str):
            seq2 = strings.str_list(seq2, sep=sep)

        align = self.aligner.get_alignment(
            seq1, seq2, return_score_matrix=return_score_matrix)

        edits = self.get_edits(align)
        return align, edits

    def get_edits(self, align):
        """
        Get edit operations (insertions, deletions, substitutions) from an
        alignment.
        """
        epsilon = self.epsilon
        seq1, seq2 = align.strings
        edits = list(zip(seq1.split(" | "), seq2.split(" | ")))

        # Collapse consecutive insertions or deletions.
        collapsed_edits = []
        for edit in edits:
            if len(collapsed_edits) == 0:
                collapsed_edits.append(edit)
                continue
            prev_edit = collapsed_edits[-1]
            if (prev_edit[0] == epsilon and edit[0] == epsilon):
                collapsed_edits[-1] = \
                    (epsilon, prev_edit[1] + " " + edit[1])
            elif (prev_edit[1] == epsilon and edit[1] == epsilon):
                collapsed_edits[-1] = \
                    (prev_edit[0] + " " + edit[0], epsilon)
            else:
                collapsed_edits.append(edit)

        # Represent change edits in context.
        context_edits = []
        for i, edit in enumerate(collapsed_edits):
            if edit[0] == edit[1]:
                continue
            context_edit = f"{edit[0]} -> {edit[1]} / {collapsed_edits[i-1][0]} __ {collapsed_edits[i+1][0]}"
            context_edits.append(context_edit)

        return context_edits


class SelfAligner():
    """
    Self-aligner for a single sequence (string / list / tuple) using the
    Smith-Waterman algorithm.
    """

    def __init__(self,
                 match_func=None,
                 theta=0.75,
                 epsilon=phon_config.epsilon,
                 verbose=False):
        if match_func is None:
            match_func = self.get_match_func(theta)
        self.match_func = match_func
        self.exact_match_func = \
            lambda seg1, seg2: 1.0 if seg1 == seg2 else 0.0
        self.theta = theta
        self.epsilon = epsilon
        self.aligner = SmithWaterman(match_func=match_func, gap_char=epsilon)
        self.verbose = verbose

    def get_match_func(self, theta):
        """
        Default match function returning segment similarity.
        """

        def match_func(seg1, seg2):
            return features.segment_similarity(
                fm,
                seg1,
                seg2,
                sim_func=features.overlap_similarity,
                feature_weights=None,
                theta=theta,
                indel_weight=-1.0)

        return match_func

    def get_alignment(self, x, sep=" ", incremental=True, verbose=None):
        """
        Find the best alignment of non-overlapping substrings of a 
        string/tuple/list using the Smith-Waterman algorithm.
        If incremental is True, process each prefix of the string and 
        return the best alignment for each prefix.
        """
        if isinstance(x, (list, tuple)):
            x_ = x
        elif sep is not None and sep != "":
            x_ = x.split(sep)
        else:
            x_ = list(x)
        xlen = len(x_)

        verbose_flag = verbose if (verbose is not None) else self.verbose

        # Initialize an empty list to store the results
        ret = []

        # Iterate over prefixes if incremental, else entire string.
        if incremental:
            prefix_posns = range(2, xlen + 1)
        else:
            prefix_posns = [xlen]

        # Iterative over prefixes.
        idx = 0
        for i in prefix_posns:
            prefix = x_[:i]
            # Find best alignment within prefix.
            salign_best = None
            score_best = None
            # Iterate over split/offset points within prefix.
            for j in range(1, len(prefix)):
                prefix_ = prefix[:j]
                _suffix = prefix[j:]
                self.aligner.match_func = self.match_func
                align = self.aligner.get_alignment( \
                    prefix_, _suffix, return_indices=True)
                score = self.aligner.get_alignment_score( \
                    align.strings[0], align.strings[1])
                score = float(score)
                # Normalize exact matches by max possible given span lengths.
                self.aligner.match_func = self.exact_match_func
                exact_score = self.aligner.get_alignment_score( \
                    align.strings[0], align.strings[1])
                (span1_len, span2_len) = SelfAlignment.get_span_lengths(align)
                max_span_len = max(span1_len, span2_len)
                if max_span_len > 0:
                    exact_score = exact_score / max_span_len
                else:
                    exact_score = 0.0
                # Create object for this alignment.
                salign = SelfAlignment(score=score,
                                       exact_score=exact_score,
                                       x=x,
                                       i=i,
                                       j=j,
                                       align=align)
                # Sort alignments by score, exact match score, order (idx).
                ret.append((score, exact_score, idx, salign))
                if score_best is None or score > score_best:
                    score_best = score
                    salign_best = salign
                    offset_best = j
                idx += 1

            if verbose_flag:
                print(f"prefix len {i}, {salign_best.pprint()}, {score_best}")

        ret = list(ret)
        heapq.heapify_max(ret)
        return ret


class SelfAlignment():
    """
    Self-alignment result.
    """

    def __init__(self, score, exact_score, x, i, j, align):
        self.score = score  # Alignment score.
        self.exact_score = exact_score  # Exact match score.
        self.x = x  # Source string/sequence.
        self.i = i  # Prefix position.
        self.j = j  # Split/offset position within prefix.
        self.align = align  # Alignment object (see string2string.alignment).
        # Lengths of aligned spans, and maximum span length.
        self.span1_len, self.span2_len = self.get_span_lengths(align)
        self.max_span_len = max(self.span1_len, self.span2_len)

    def __rep__(self):
        return f"SelfAlignment(score={self.score}, exact_score={self.exact_score}, x={self.x}, i={self.i}, j={self.j}, span1_len={self.span1_len}, span2_len={self.span2_len}, align={self.align})"

    def pprint(self, sep=" "):
        # todo: add correspondence indices to aligned subparts
        x = self.x
        i = self.i
        j = self.j
        align = self.align
        align_strings = align.strings
        align_indices = align.indices
        if align_strings is None \
            or align_strings[0] == "" \
                or align_strings[1] == "":
            return None
        if align_indices is None:
            return None

        span1, span2 = self.adjust_indices(align, j)

        if isinstance(x, (list, tuple)):
            x_ = x
        elif sep is not None and sep != "":
            x_ = x.split(sep)
        else:
            x_ = list(x)

        y = x_[:span1[0]] \
            + ["⟨"] + x_[span1[0]:span1[1]] + ["⟩"] \
            + x_[span1[1]:span2[0]] \
            + ["⟨"] + x_[span2[0]:span2[1]] + ["⟩"] \
            + x_[span2[1]:]
        y = " ".join(y)

        return y

    @classmethod
    def adjust_indices(cls, align, offset):
        """
        Adjust the indices of an alignment to account for the fact that the
        beginning of the second string is offset relative to the first string.
        """
        if align.indices is None:
            return None
        span1 = align.indices[0]
        span2 = align.indices[1]
        span2 = (span2[0] + offset, span2[1] + offset)
        return (span1, span2)

    @classmethod
    def get_span_lengths(cls, align):
        """
        Get lengths of aligned spans.
        """
        span1, span2 = align.indices
        span1_len = (span1[1] - span1[0])
        span2_len = (span2[1] - span2[0])
        return (span1_len, span2_len)


# # # # # # # # # #

if __name__ == "__main__":
    bos = phon_config.bos
    eos = phon_config.eos

    aligner = Aligner(epsilon=phon_config.epsilon)
    x_str = strings.str_split(f"{bos}satu{eos}")
    y_str = strings.str_split(f"{bos}satuɲa{eos}")
    alignment, edits = aligner.get_alignment(x_str, y_str, sep=" ")
    print(f"Edits for alignment of {x_str} and {y_str}")
    print(edits)
    print()
    #sys.exit(0)

    # Self-alignment examples.
    self_aligner = SelfAligner(epsilon=phon_config.epsilon)
    examples = [
        "satu-satuɲa", "kəkaseh-kəkaseh", "sə-səpet", "asal-usol",
        "llama-llama", "doggy-oggy", "piggy-wiggy", "snalnal", "snalfak",
        "wofewo"
    ]
    x_str = examples[2]
    x_str = [x_str[i] for i in range(len(x_str)) if x_str[i] != "-"]
    x_str = "".join(x_str)
    print(x_str)
    salign = self_aligner.get_alignment(x_str, sep="", incremental=False)[0]
    print(salign)
    print(self_aligner.markup_alignment(salign, sep=""))
    sys.exit(0)

    #x_str = "s a t u s a t u ɲ a"
    x_str = "s ə s ə p e t"
    x_str_ = x_str.split(" ")
    x_prefix = x_str_[:4]
    x_suffix = x_str_[4:]
    print(x_prefix, x_suffix)

    def match_func(seg1, seg2):
        return features.segment_similarity(
            fm,
            seg1,
            seg2,
            sim_func=features.overlap_similarity,
            feature_weights=None,
            theta=0.75,
            indel_weight=-1.0)

    aligner = SmithWaterman(match_func=match_func,
                            gap_char=phon_config.epsilon)
    align = aligner.get_alignment(x_prefix,
                                  x_suffix,
                                  return_indices=True,
                                  return_score_matrix=True)
    print(len(align))
    pprint(align)

    print(align.score_matrix)
    plt.imshow(align.score_matrix)
    plt.show()
