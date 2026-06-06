# Convenience functions operating on string or list of strings.
# todo: func to remove punctuation
# todo: mirror str_util.R

import re
import itertools
import string
import polars as pl
from phonopy import config as phon_config
from collections import Counter

punc = string.punctuation
smart_punc = '“”‘’'
punc = punc + smart_punc
punc_regexp = r'[' + re.escape(punc) + r']'

collection_types = (list, set, tuple)  # disjunctive type


def str_squish(word):
    """
    Collapse consecutive space chars to single space,
    remove leading/trailing spaces.
    see: https://stringr.tidyverse.org/reference/str_trim.html
    """
    if isinstance(word, collection_types):
        return [str_squish(word_) for word_ in word]
    ret = re.sub(r'\s+', ' ', word)
    ret = ret.strip()
    return ret


def str_sep(word, segs=None, regexp=None):
    """
    Separate segments in word with spaces using
    alphabet (list of segments) or regexp.
    # see: torchtext.data.functional.simple_space_split
    """
    if segs is None and regexp is None:
        regexp = '(.)'

    if regexp is None:
        # Longer segments take priority.
        segs.sort(key=lambda x: len(x), reverse=True)
        regexp = '(' + '|'.join(segs) + ')'

    if isinstance(word, collection_types):
        return [str_sep(word_, segs, regexp) for word_ in word]

    ret = re.sub(regexp, '\\1 ', word)
    ret = str_squish(ret)
    return ret


def add_delim(word, edge='both', iostring=False):
    """
    Add begin/end delimiters to space-separated string.
    """
    if isinstance(word, collection_types):
        return [add_delim(word_, edge, iostring) for word_ in word]
    bos = phon_config.bos
    eos = phon_config.eos
    if iostring:
        bos = f'{bos}:{bos}'
        eos = f'{eos}:{eos}'
    if edge == 'begin':
        ret = f'{bos} {word}'
    elif edge == 'end':
        ret = f'{word} {eos}'
    else:  # default
        ret = f'{bos} {word} {eos}'
    return ret


delim = add_delim  # Alias.


def remove_delim(word):
    """
    Remove begin/end delimiters.
    """
    if isinstance(word, collection_types):
        return [remove_delim(word_) for word_ in word]
    bos = phon_config.bos
    eos = phon_config.eos
    ret = word
    # Remove from iostring.
    ret = re.sub(f'{bos}:{bos}', '', ret)
    ret = re.sub(f'{eos}:{eos}', '', ret)
    # Remove from istring or ostring.
    ret = re.sub(f'{bos}', '', ret)
    ret = re.sub(f'{eos}', '', ret)
    ret = str_squish(ret)
    return ret


undelim = remove_delim  # Alias.


def remove_epsilon(word):
    """
    Remove epsilons.
    """
    if isinstance(word, collection_types):
        return [remove_epsilon(word_) for word_ in word]
    epsilon = phon_config.epsilon
    ret = word
    # Remove from iostring.
    ret = re.sub(f'{epsilon}:{epsilon}', '', ret)
    # Remove from istring or ostring.
    ret = re.sub(f'{epsilon}', '', ret)
    ret = str_squish(ret)
    return ret


def remove_segs(word, segs=None, regexp=None, sep=' '):
    """
    Remove designated segments.
    """
    if segs is None and regexp is None:
        return word
    if regexp is None:
        regexp = '(' + '|'.join(segs) + ')'

    if isinstance(word, collection_types):
        return [remove(word_, segs, regexp, sep) for word_ in word]

    ret = re.sub(regexp, '', word)
    ret = str_squish(ret)
    return ret


def remove_punc(word):
    """
    Remove punctuation from word.
    todo: sep argument
    """
    if isinstance(word, collection_types):
        return [remove_punc(word_) for word_ in word]
    ret = re.sub(punc_regexp, '', word)
    ret = str_squish(ret)
    return ret


def str_pad(word, n=1, sep=' ', pad=None, edge='end'):
    """
    Pad string up to length n at designated edge.
    """
    if isinstance(word, collection_types):
        return [str_pad(word_, n, sep, pad, edge) for word_ in word]
    if word is None:
        word = ''
    ret = word.split(sep) if sep != '' else list(word)
    if pad is None:
        pad = phon_config.epsilon
    if len(ret) < n:
        padding = [pad] * (n - len(ret))
        if edge == 'end':
            ret = ret + padding
        elif edge == 'begin':
            ret = padding + ret
    ret = sep.join(ret)
    return ret


def str_pad2(input_, output_=None, sep=' ', pad=None, edge='end'):
    """
    Pad input/output strings to same length.
    """
    if isinstance(input_, collection_types):
        if output_:
            pairs = zip(input_, output_)
        else:
            pairs = input_  # assume pre-zipped
        return [str_pad2(x, y, sep, pad, edge) for (x, y) in pairs]

    if input_ is None:
        input_ = ''
    if output_ is None:
        output_ = ''

    n_input = len(input_.split(sep)) if sep != '' else len(input_)
    n_output = len(output_.split(sep)) if sep != '' else len(output_)

    if n_input < n_output:
        input_ = str_pad(input_, n_output, sep, pad, edge)
    if n_input > n_output:
        output_ = str_pad(output_, n_input, sep, pad, edge)
    return (input_, output_)


def str_subs(word, subs={}, sep=' '):
    """
    Change transcription by applying dictionary of
    substitutions to string(s).
    note: handles deterministic substitutions only.
    note: alternative to native str.maketrans / 
    str.translate for space-separated segment sequences.
    """
    if isinstance(word, collection_types):
        return [str_subs(word_, subs, sep) for word_ in word]
    sep_flag = (sep is not None and sep != '')
    ret = word.split(sep) if sep_flag else word
    for s, r in subs.items():
        ret = [subs[x] if x in subs else x for x in ret]
    ret = sep.join(ret) if sep_flag else ''.join(ret)
    ret = str_squish(ret)
    return ret


retranscribe = str_subs  # Alias.

# # # # # # # # # #
# Phonological pseudo-regexps.
# todo: replace with equiv. wyfst functions


def combos(s):
    """
    Convert string representations of phonological sets
    (e.g., set of acceptable onsets) to list of tuples.
    Ex. combos('(k|g) (r|l|w)') => [(k,r), (k,l), ... (g,w)])
    """
    if isinstance(s, collection_types):
        return [tuple(x.split(' ')) for x in s]
    parts = s.split(' ')
    parts = [str_squish(re.sub('[()]', '', part)) for part in parts]
    parts = [part.split('|') for part in parts]
    ret = itertools.product(*parts)
    ret = map(lambda x: tuple(x), ret)
    ret = list(dict.fromkeys(ret))
    return ret


# # # # # # # # # #
# Correspondence indices.

digits = '0123456789-'
subscript_digits = '₀₁₂₃₄₅₆₇₈₉₋'
digit2subscript = str.maketrans( \
    digits, subscript_digits)


def str_index(word, skip=[], sep=' ', subscript=True):
    """
    Add integer indices (numbered left-to-right)
    to end of segments in separated word(s).
    """
    if isinstance(word, collection_types):
        ret = [str_index(word_, skip, sep, subscript) for word_ in word]
        return ret
    # Split.
    segs = word.split(sep) if sep != '' else word
    # Apply.
    use_skip = (skip is not None and len(skip) > 0)
    segs_idx, idx = [], 0
    for seg in segs:
        if use_skip and seg in skip:
            segs_idx.append(seg)
            continue
        if subscript:
            segs_idx.append(f'{seg}{as_index(idx)}')
        else:
            segs_idx.append(f'{seg}_{idx}')
        idx += 1
    # Combine.
    ret = sep.join(segs_idx)
    return ret


def str_deindex(word, sep=' ', subscript=True):
    """
    Remove integer indices from end of segments in 
    separated word(s).
    """
    if isinstance(word, collection_types):
        ret = [str_deindex(word_, sep) for word_ in word]
        return ret
    # Split.
    segs = word.split(sep) if sep != '' else word
    # Apply.
    if subscript:
        segs = [re.sub(f'[{subscript_digits}]+$', '', seg) for seg in segs]
    else:
        segs = [re.sub(f'_[{digits}]+$', '', seg) for seg in segs]
    # Combine.
    ret = sep.join(segs)
    return ret


def to_index(idx, subscript=True):
    """ Convert integer to subscript index. """
    idx = str(idx)
    if not subscript:
        return idx
    # if not re.search(f'^[{digits}]+$', idx):
    #     return None
    ret = idx.translate(digit2subscript)
    return ret


# Alias.
as_index = to_index

# def retranscribe_sep(x, subs, sep=' '):
#     """
#     Change transcription by applying dictionary of
#     substitutions to string(s) of separated segments.
#     """
#     if isinstance(x, list):
#         return [retranscribe_sep(xi, subs, sep) for xi in x]
#     y = x.split(sep)
#     y = [subs[yi] if yi in subs else yi for yi in y]
#     y = sep.join(y)
#     return y

# # # # # # # # # #
# Unigram/bigram frequencies.


def unigram_tokens(word, sep=' '):
    """
    Get unigram tokens from word(s).
    """
    if word is None:
        return []
    if isinstance(word, collection_types):
        toks = []
        for word_ in word:
            toks += unigram_tokens(word_, sep)
        return toks
    try:
        if sep is not None and sep != '':
            toks = word.split(sep)
        else:
            toks = list(word)
    except Exception as e:
        #print(f'Error in unigram_tokens: {e} on word {word}')
        toks = []
    return toks


def unigrams(word, sep=' '):
    """
    Get unigram types and type frequencies from word(s).
    """
    ret = Counter()
    if isinstance(word, pl.Series):
        word = word.to_list()
    elif isinstance(word, str):
        word = [word]
    for word_ in word:
        ret.update(unigram_tokens(word_, sep))
    ret = pl.DataFrame({ \
        'seg': ret.keys(),
        'freq': ret.values() })
    ret = ret.sort(by='freq', descending=True)
    return ret


get_segments = unigrams  # Alias.


def get_words(text, sep=' '):
    """
    Get words types and type frequencies from text(s).
    """
    ret = unigrams(text, sep)
    ret = ret.rename({'seg': 'word'})
    return ret


def bigram_tokens(word, sep=' '):
    """
    Get bigram tokens from one word.
    """
    if word is None:
        return []
    if isinstance(word, collection_types):
        toks = []
        for word_ in word:
            toks += bigram_tokens(word_, sep)
        return toks
    if sep is not None and sep != '':
        word = word.split(sep)
    toks = list(zip(word[:-1], word[1:]))
    return toks


def bigrams(word, sep=' '):
    """
    Get bigrams and their type frequencies from word(s).
    """
    ret = Counter()
    if isinstance(word, pl.Series):
        word = word.to_list()
    elif isinstance(word, str):
        word = [word]
    for word_ in word:
        ret.update(bigram_tokens(word_, sep))
    ret = pl.DataFrame({ \
        'bigram': ret.keys(),
        'freq': ret.values() })
    ret = ret.sort(by='freq', descending=True)
    return ret


def gram_tokens(word, k=1, sep=' '):
    if k == 1:
        return unigram_tokens(word, sep)
    if k == 2:
        return bigram_tokens(word, sep)
    print('gram_tokens() not yet implemented for k > 2')
    return None


def grams(word, k=1, sep=' '):
    if k == 1:
        return unigrams(word, sep)
    if k == 2:
        return bigrams(word, sep)
    print(f'Nope: grams() not yet implemented for k={k} > 2.')
    return None


def lcp(x, y, prefix=True):
    """
    Longest common prefix (or suffix) of two segment sequences.
    """
    if x == y:
        return x
    if not prefix:
        x = x[::-1]
        y = y[::-1]
    n_x, n_y = len(x), len(y)
    n = max(n_x, n_y)
    for i in range(n + 1):
        if i >= n_x:
            match = x
            break
        if i >= n_y:
            match = y
            break
        if x[i] != y[i]:
            match = x[:i]
            break
    if not prefix:
        match = match[::-1]
    return match


if __name__ == "__main__":
    print(phon_config.bos, phon_config.eos, phon_config.epsilon)
    print(phon_config.eos)
    print(str_squish(' t  e s  t   '))
    print(str_sep('cheek', segs=['ch', 'ee', 'k']))
    print(str_sep('cheek', regexp='(ch|ee|k)'))
    print(add_delim('test'))
    print(remove_segs('testing', segs='aeiou'))
    print(remove_punc('[(testing).!]?'))
