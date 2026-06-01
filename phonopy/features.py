# Read and standardize phonological feature matrices;
# prepare features matrices for use in pytorch.
# Typical import:
# from phonopy import features as phon_features
import re, string, sys
from pathlib import Path
import pandas as pd  # todo: replace with polars
#import polars as pl
import numpy as np
from collections import namedtuple
#from unicodedata import normalize

from phonopy import config as phon_config

default_feature_file = Path.home() / \
    'Code/Python/phonopy/extern/hayes_features.csv'

default_segments = [
    'p', 'b', 't', 'd', 't͡ʃ', 'k', 'g', 'ʔ', 'f', 's', 'ʃ', 'h', 'm', 'n',
    'ɲ', 'ŋ', 'r', 'j', 'w', 'l'
] + ['i', 'e', 'a', 'o', 'u']
# ref. Maddieson (1986)

# # # # # # # # # #


class FeatureMatrix():
    """ Container for matrix of phonological features. """

    # todo: delegate to panphon or phoible if possible
    # todo: warn about missing/nan feature values in matrix
    # see related: torchtext.vocab.Vocab

    def __init__(self, segments, vowels, features, ftr_matrix):
        self.segments = segments  # Segments (incl. epsilon/bos/eos).
        self.vowels = vowels  # Segments that are vowels.
        self.features = features  # Feature names.
        # Feature matrix, values in {'+', '-', '0'}.
        # format: one row per segment, features in columns
        self.ftr_matrix = ftr_matrix
        # Feature matrix as numpy array, values in {+1., -1., 0.}.
        # format: one row per segment, features in columns
        self.ftr_matrix_vec = self.to_numpy(ftr_matrix)

        # Segment <-> idx.
        self.seg2idx = {}
        self.idx2seg = {}
        for idx, seg in enumerate(self.segments):
            self.seg2idx[seg] = idx
            self.idx2seg[idx] = seg

        # Segment -> feature-value dict and vector.
        self.seg2ftrs = {}
        self.seg2ftr_vec = {}
        for i, seg in enumerate(self.segments):
            ftrs = ftr_matrix.iloc[i, :].to_dict()
            self.seg2ftrs[seg] = ftrs
            self.seg2ftr_vec[seg] = tuple(ftrs.values())

    # todo: make class method
    def to_numpy(self, ftr_matrix):
        """
        Convert feature matrix to numpy ndarray.
        """
        ftr_vals = {'+': '1', '+1': '1', '-': '-1'}
        ftr_matrix_vec = ftr_matrix.copy().replace(ftr_vals)
        ftr_matrix_vec = ftr_matrix_vec.to_numpy(dtype=float)
        # for (key, val) in ftr_specs.items():
        #     ftr_matrix_vec = ftr_matrix_vec.replace( \
        #         to_replace=key, value=val).astype(float)
        # ftr_matrix_vec = np.array(ftr_matrix_vec.values)
        return ftr_matrix_vec

    # Methods defined outside of class.
    def get_features(self, x, **kwargs):
        return get_features(self, x, **kwargs)

    def get_change(self, x, y, **kwargs):
        return get_change(self, x, y, **kwargs)

    def subsumes(self, ftrs1, ftrs2, **kwargs):
        return subsumes(ftrs1, ftrs2, **kwargs)

    def natural_class(self, ftrs=None, **kwargs):
        return natural_class(self, ftrs, **kwargs)

    def to_regexp(self, ftrs, **kwargs):
        return to_regexp(self, ftrs, **kwargs)


def import_features(feature_file=default_feature_file,
                    segments=None,
                    standardize=True,
                    save_file=None,
                    verbose=True):
    """
    Read feature matrix from file with segments in first *column*. 
    If segments is specified, eliminates constant and redundant features. 
    If standardize flag is set, add:
    - epsilon 'segment' with all-zero feature vector.
    - non-epsilon feature 'sym'.
    - bos/eos delimiters and feature to identify them (begin:+1, end:-1).
    - feature 'seg' to identify all segments (non-epsilon/bos/eos).
    - feature 'C/V' to identify consonants (C) and vowels (V) (C:+1, V:-1).
    Otherwise these segments and features are assumed to be already 
    present in the feature matrix or file.
    todo: arrange segments in IPA order
    """

    # Read matrix from file (absolute path).
    ftr_matrix = pd.read_csv(feature_file,
                             sep=',',
                             encoding='utf-8',
                             comment='#')
    if verbose:
        print(ftr_matrix)

    # Add long segments and length feature ("let there be colons").
    if 0:  # todo: make config option
        ftr_matrix_short = ftr_matrix.copy()
        ftr_matrix_long = ftr_matrix.copy()
        ftr_matrix_short['long'] = '-'
        ftr_matrix_long['long'] = '+'
        ftr_matrix_long.iloc[:, 0] = \
            [x + 'ː' for x in ftr_matrix_long.iloc[:, 0]]
        ftr_matrix = pd.concat( \
            [ftr_matrix_short, ftr_matrix_long],
            axis=0,
            sort=False)

    # List all segments and features in the matrix, locate
    # syllabic feature, and remove first column (containing segments).
    # ftr_matrix.iloc[:,0] = [normalize('NFC', x) for x in ftr_matrix.iloc[:,0]]
    segments_all = [x for x in ftr_matrix.iloc[:, 0]]
    features_all = [x for x in ftr_matrix.columns[1:]]
    syll_ftr = [ftr for ftr in features_all \
        if re.match('^(syl|syll|syllabic)$', ftr)][0]
    ftr_matrix = ftr_matrix.iloc[:, 1:]

    # Standardize all segments.
    segments_all = [standardize_segment(x) for x in segments_all]
    #print('segments_all:', segments_all)

    # Handle segments with diacritics. [partial]
    # (feature names from Hayes matrix)
    diacritics = [ \
        ("[ˈ]", ('stress', '+')),
        ("[ʲ]", ('high', '+')),  # fixme: palatalization
        ("[ʼ]", ('constr.gl', '+')),
        ("[ʰ]", ('spread.gl', '+')),
        ("[ʱ]", ('spread.gl', '+')),  # Bengali
        ("[*]", ('constr.gl', '+')),  # Korean
        ("[ʷ]", ('round', '+')),
        ("[˞]", ('rhotic', '+')),
        ("\u0303", ('nasal', '+')),
    ]
    diacritic_segs = []
    if segments is not None:
        # Standardize segments.
        segments = [standardize_segment(seg) for seg in segments]
        for seg in segments:
            # Detect and strip diacritics.
            base_seg = seg
            diacritic_ftrs = []  # Features marked by diacritics.
            for (diacritic, ftrval) in diacritics:
                if re.search(diacritic, base_seg):
                    diacritic_ftrs.append(ftrval)
                    base_seg = re.sub(diacritic, '', base_seg)
            if len(diacritic_ftrs) == 0:
                continue
            # Specify diacritic features.
            try:
                idx = segments_all.index(base_seg)
            except:
                print(
                    f'Error: could not find index of base segment |{base_seg}| from {seg}'
                )
                raise
            base_ftr = [x for x in ftr_matrix.iloc[idx, :]]
            for ftr, val in diacritic_ftrs:
                idx = features_all.index(ftr)
                base_ftr[idx] = val
            diacritic_segs.append((seg, base_ftr))
        # Add segments with diacritics and features.
        if len(diacritic_segs) > 0:
            new_segs = [x[0] for x in diacritic_segs]
            new_ftr_vecs = pd.DataFrame([ftr for (seg, ftr) in diacritic_segs])
            new_ftr_vecs.columns = ftr_matrix.columns
            segments_all += new_segs
            ftr_matrix = pd.concat([ftr_matrix, new_ftr_vecs],
                                   ignore_index=True)
        #print(segments_all)
        #print(ftr_matrix)

    # Reduce feature matrix to observed segments (if provided), pruning
    # features other than syll_ftr that have constant values.
    if segments is not None:
        # Check that all segments appear in the feature matrix.
        missing_segments = \
            [x for x in segments if x not in segments_all]
        if len(missing_segments) > 0:
            raise Exception(f'Segments missing from feature matrix: '
                            f'{missing_segments}')

        segments = [x for x in segments_all if x in segments]
        ftr_matrix = ftr_matrix.loc[[x in segments for x in segments_all], :]
        ftr_matrix.reset_index(drop=True)

        features = [ftr for ftr in ftr_matrix.columns \
            if ftr == 'syll_ftr' or ftr_matrix[ftr].nunique() > 1]
        ftr_matrix = ftr_matrix.loc[:, features]
        ftr_matrix = ftr_matrix.reset_index(drop=True)
    else:
        segments = segments_all
        features = features_all

    # Syllabic segments.
    vowels = [x for i, x in enumerate(segments) \
        if ftr_matrix[syll_ftr][i] == '+']

    # Standardize feature matrix.
    ftr_matrix.index = segments
    fm = FeatureMatrix(segments, vowels, features, ftr_matrix)
    if standardize:
        fm = standardize_matrix(fm)

    # Write feature matrix.
    # todo: pickle FeatureMatrix
    if save_file:
        save_file = Path(save_file).with_suffix('.ftr')
        fm.ftr_matrix.to_csv(save_file, index_label='ipa')

    setattr(phon_config, 'feature_matrix', fm)
    return fm


read_features = import_features  # Alias.


def one_hot_features(segments=None,
                     vowels=None,
                     standardize=True,
                     save_file=None,
                     verbose=False):
    """
    Create one-hot feature matrix from list of segments
    (or number of ascii segments), optionally standardizing
    with epsilon/bos/eos and features.
    """
    if isinstance(segments, int):
        segments = string.ascii_lowercase[:segments]
    features = segments[:]
    ftr_matrix = pd.DataFrame( \
        np.eye(len(segments))
    )
    ftr_matrix.columns = segments
    fm = FeatureMatrix(segments, vowels, features, ftr_matrix)
    if standardize:
        fm = standardize_matrix(fm)

    if save_file:
        ftr_matrix = fm.ftr_matrix
        ftr_matrix.to_csv(save_file.with_suffix('.ftr'), index_label='ipa')

    setattr(phon_config, 'feature_matrix', fm)
    return fm


def default_features(**kwargs):
    """ Default features and segments for quick start. """
    fm = import_features( \
        default_feature_file, default_segments, **kwargs)
    return fm


def standardize_matrix(fm):
    """
    Add special segments (epsilon/bos/eos) and features 
    (non-epsilon sym, begin/end, seg, C/V) to feature matrix.
    """
    if fm.vowels is None:
        print('Vowels must be specified to standardize feature matrix')
        sys.exit(0)

    # # # # # # # # # #
    # Special sehments
    epsilon = phon_config.epsilon
    bos = phon_config.bos
    eos = phon_config.eos
    #wildcard = config.wildcard
    segments = [epsilon, bos, eos, *fm.segments]

    # Special segments are unspecified for all ordinary features.
    special_seg_vals = pd.DataFrame( \
        {ftr: '0' for ftr in fm.features},
        index=[0])

    # Special segments occupy first three rows of revised feature matrix.
    ftr_matrix = pd.concat( \
        [special_seg_vals] * 3 +
        [fm.ftr_matrix]).reset_index(drop=True)

    # # # # # # # # # #
    # Special features.
    # Non-epsilon feature: all segments except epsilon are +
    sym_ftr_vals = ['0'] + ['+'] * (len(segments) - 1)

    # Delim ftr: bos is +, eos is -,
    # all others segments are unspecified.
    delim_ftr_vals = ['0', '+', '-'] + ['0'] * (len(segments) - 3)

    # Ordinary seg ftr: consonants and vowels are '+',
    # all other segments are unspecified.
    seg_ftr_vals = ['0', '0', '0'] + ['+'] * (len(segments) - 3)

    # C/V ftr: consonants are +, vowels are -,
    # all other segments are unspecified.
    cv_ftr_vals = ['0', '0', '0'] + \
        ['-' if seg in fm.vowels else '+' for seg in fm.segments]

    # Special features occupy first three
    # columns of revised feature matrix.
    special_ftrs = pd.DataFrame({
        'sym': sym_ftr_vals,
        'begin/end': delim_ftr_vals,
        'seg': seg_ftr_vals,
        'C/V': cv_ftr_vals
    })
    ftr_matrix = pd.concat([special_ftrs, ftr_matrix], axis=1) \
                   .reset_index(drop=True)
    ftr_matrix.index = segments  # todo: checkme
    features = ['sym', 'begin/end', 'seg', 'C/V', *fm.features]
    phon_config.sym_ftr = sym_ftr = 0
    phon_config.delim_ftr = delim_ftr = 1
    phon_config.seg_ftr = seg_ftr = 2
    phon_config.cv_ftr = cv_ftr = 3

    fm = FeatureMatrix(segments, fm.vowels, features, ftr_matrix)
    return fm


# # # # # # # # # #
# Segments and natural classes.


def standardize_segment(x):
    """
    Standardize segment (partial implementation):
    no script g, no tiebars, ...
    """
    ipa_substitutions = {'\u0261': 'g', 'ɡ': 'g', 'ɡ': 'g', '͡': ''}
    y = x
    for (s, r) in ipa_substitutions.items():
        y = re.sub(s, r, y)
    return y


def get_features(fm, x, keep_zero=True):
    """
    Return feature values of one segment, or feature
    values shared by a collection of segments.
    """
    empty = dict()
    # None / empty string / empty collection.
    if not x:  #
        return empty
    # Single segment.
    if isinstance(x, str):
        ret = fm.sym2ftrs.get(x, empty).items()
    # Collection of segments.
    else:
        ret = None
        for xi in x:
            ftrsi = fm.sym2ftrs.get(xi, empty)
            if ret is None:
                ret = ftrsi.items()
            else:
                ret = ret & ftrsi.items()
    # Optionally remove zero-valued features.
    if not keep_zero:
        ret = [(ftr, val) for (ftr,val) in ret \
            if not is_zero(val)]
    return dict(ret)


def get_change(fm, x, y):
    """
    Return (features of y) - (features of x).
    """
    if isinstance(x, str):
        ftrs_x = get_features(fm, x)
    else:
        ftrs_x = x
    if isinstance(x, str):
        ftrs_y = get_features(fm, y)
    else:
        ftrs_y = y
    ret = {}
    for ftr in fm.features:
        val = ftrs_y.get(ftr, '0')
        if ftrs_x.get(ftr, '0') != val:
            ret[ftr] = val
    return ret


def subsumes(ftrs1, ftrs2):
    """
    Feature-value dict ftrs1 subsumes ftrs2 iff every
    non-zero feature value in ftrs1 is also in ftrs2.
    """
    for ftr, val in ftrs1.items():
        if is_zero(val):
            continue
        if ftrs2.get(ftr) != val:
            return False
    return True


def natural_class(fm, ftrs=None, **kwargs):
    """
    Return natural class (= set of symbols)
    defined by feature-value dict ftrs.
    """
    # Handle feature-matrix string.
    if isinstance(ftrs, str):
        ftrs = from_str(fm, ftrs)
        return [natural_class(fm, ftrs1) for ftrs1 in ftrs]
    # Handle feature-value dict and keyword args.
    if not ftrs:
        ftrs = dict()
    for (key, val) in kwargs.items():
        ftrs[key] = val
    # Handle numeric/verbose feature vals.
    for key in ftrs:
        val = ftrs[key]
        if (val == 1 or val == '+1'):
            ftrs[key] = '+'
        elif (val == -1 or val == '-1'):
            ftrs[key] = '-'
    # Natural class determined by subsumption.
    if not ftrs:
        ret = set([x for x in fm.symbols if x != phon_config.epsilon])
    else:
        ret = set([
            x for x, ftrs_x in fm.sym2ftrs.items()
            if subsumes(ftrs, ftrs_x) and x != phon_config.epsilon
        ])
    return ret


def from_str(fm, ftrs):
    """
    Convert feature-matrix string to feature-value dict.
    note: '[]' is interpreted as [+seg(ment)].
    """
    ftrs = re.sub(r'\s+', '', ftrs)
    ftrs = re.sub(r'\[', '', ftrs)
    ftrs = ftrs.split(r']')[:-1]
    ret = []
    for ftrs1 in ftrs:
        if ftrs1 == '':
            ftrs1 = '+seg'
        ftrs1 = ftrs1.split(',')
        ftrs1 = {x[1:]: x[0] for x in ftrs1 if len(x) > 1}
        ret.append(ftrs1)
    return ret


def to_str(fm, ftrs):
    """
    Convert sequence of feature-value dicts to
    feature-matrix string.
    """
    if not ftrs:
        return ''
    if not isinstance(ftrs, (list, tuple)):
        ftrs = [ftrs]
    ret = []
    for ftrs1 in ftrs:
        ftrs1 = list(ftrs1.items())
        ftrs1.sort(key=lambda ftr_val: fm.features.index(ftr_val[0]))
        ret1 = [f'{val}{ftr}' for ftr, val in ftrs1 if not is_zero(val)]
        ret1 = '[' + ', '.join(ret1) + ']'
        ret.append(ret1)
    return ''.join(ret)


def to_regexp(fm, segs):
    """
    Convert sequence of natural classes (segment sets) or
    feature-value dicts or a feature-matrix string to regexp.
    note: '[]' is interpreted as [+seg(ment)].
    """
    # Convert feature-matrix string to features.
    if isinstance(segs, str):
        segs = from_str(fm, segs)
    # Promote singleton segs arg to list.
    if not isinstance(segs, (list, tuple)):
        segs = [segs]
    # Create regexp.
    ret = []
    for segs1 in segs:
        if isinstance(segs1, dict):
            segs1 = natural_class(fm, segs1)
        segs1 = list(segs1)
        segs1.sort(key=lambda x: fm.segments.index(x))
        ret.append('(' + '|'.join(segs1) + ')')
    return ''.join(ret)


def is_zero(val):
    ret = (val == '0') or (val == 0) or (val is None)
    return ret


# # # # # # # # # #

if __name__ == "__main__":
    fm = default_features()
    print(fm.segments)
    print(fm.vowels)
    print(fm.features)
    print(fm.ftr_matrix)
    print(fm.ftr_matrix_vec)
    #print(fm.ftr_matrix_vec.shape, len(fm.segments), len(fm.features))
    print(get_features(fm, 'a'))
    print(fm.get_features('a'))
    print(natural_class(fm, '[+ syllabic ]'))
    print(to_regexp(fm, '[+ syllabic ][-syllabic]'))
    print(fm.to_regexp('[+ syllabic ][-syllabic]'))
    print(fm.to_regexp('[][+syllabic]'))
    print(get_change(fm, 'o', 'u'))
    print(fm.get_change('o', 'u'))
    delta = fm.get_change('o', 't')
    result = fm.get_features('o') | delta
    print(fm.natural_class(result))

    ftrs = get_features(fm, ['i', 'e', 'a', 'o', 'u'])
    ftrs_str = to_str(fm, ftrs)
    print(ftrs_str)

# # # # # # # # # #

# deprecated
# def ftrspec2vec(ftrspecs, feature_matrix=None):
#     """

#     Convert dictionary of feature specifications (ftr -> +/-/0)
#     to feature + 'attention' vectors.
#     If feature_matrix is omitted, default to environ.config.
#     """
#     if feature_matrix is not None:
#         features = feature_matrix.features
#     else:
#         features = config.ftrs

#     specs = {'+': 1., '-': -1., '0': 0.}
#     n = len(features)
#     w = np.zeros(n)
#     a = np.zeros(n)
#     for ftr, spec in ftrspecs.items():
#         if spec == '0':
#             continue
#         i = features.index(ftr)
#         if i < 0:
#             print('ftrspec2vec: could not find feature', ftr)
#         w[i] = specs[spec]  # non-zero feature specification
#         a[i] = 1.  # 'attention' weight identifying non-zero feature
#     return w, a
