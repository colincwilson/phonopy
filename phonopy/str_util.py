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
punc_regexp = r"[" + re.escape(punc) + r"]"


def squish(word):
    """
    Collapse consecutive spaces, remove leading/trailing spaces.
    see: https://stringr.tidyverse.org/reference/str_trim.html
    """
    if isinstance(word, list):
        return [squish(word_) for word_ in word]
    ret = re.sub('[ ]+', ' ', word)
    ret = ret.strip()
    return ret


def str_sep(word, syms=None, regexp=None):
    """
    Separate symbols in word with spaces using
    symbol list or regexp.
    # see: torchtext.data.functional.simple_space_split
    """
    if syms is None and regexp is None:
        regexp = "(.)"
    if regexp is None:
        syms.sort(key=lambda x: len(x), reverse=True)
        regexp = '(' + '|'.join(syms) + ')'

    if isinstance(word, list):
        return [str_sep(word_, syms, regexp, sep) for word_ in word]

    ret = re.sub(regexp, "\\1 ", word)
    ret = squish(ret)
    return ret


def add_delim(word, sep=False, edge='both'):
    """
    Add begin/end symbols to space-separated string.
    """
    if isinstance(word, list):
        return [add_delim(word_, sep, edge) for word_ in word]
    ret = word
    if sep:
        ret = ' '.join(ret)
    if edge == 'begin':
        ret = f'{phon_config.bos} {ret}'
    elif edge == 'end':
        ret = f'{ret} {phon_config.eos}'
    else:  # default edge == 'both'
        ret = f'{phon_config.bos} {ret} {phon_config.eos}'
    return ret


def remove_delim(word):
    """
    Remove begin/end delimiters.
    todo: sep argument
    """
    if isinstance(word, list):
        return [remove_delim(word_) for word_ in word]
    ret = word
    ret = re.sub(f'{phon_config.bos}', '', ret)
    ret = re.sub(f'{phon_config.eos}', '', ret)
    ret = squish(ret)
    return ret


def remove_syms(word, syms=None, regexp=None, sep=' '):
    """
    Remove designated symbols.
    """
    if syms is None and regexp is None:
        return word
    if regexp is None:
        regexp = '(' + '|'.join(syms) + ')'

    if isinstance(word, list):
        return [remove(word_, syms, regexp, sep) for word_ in word]

    ret = re.sub(regexp, '', word)
    ret = squish(ret)
    return ret


def remove_punc(word):
    """
    Remove punctuation from word.
    todo: sep argument
    """
    if isinstance(word, list):
        return [remove_punc(word_) for word_ in word]
    ret = re.sub(punc_regexp, '', word)
    ret = squish(ret)
    return ret


def str_pad(word, n, sep=' ', pad=phon_config.epsilon):
    """
    Pad end of string or list up to length n.
    """
    if word is None:
        ret = [pad]
    elif isinstance(word, str) and sep != '':
        ret = word.split(' ')
    else:
        ret = word
    if len(ret) < n:
        ret += [pad] * (n - len(ret))
    if isinstance(word, str):
        sep.join(ret)
    return ret


def str_subs(word, subs={}, sep=' '):
    """
    Change transcription by applying dictionary of
    substitutions to string(s).
    note: handles deterministic substitutions only.
    note: alternative to native str.maketrans / 
    str.translate for space-separated symbol sequences.
    """
    if isinstance(word, list):
        return [str_subs(word_, subs, sep) for word_ in word]
    sep_flag = (sep is not None and sep != '')
    ret = word.split(sep) if sep_flag else word
    for s, r in subs.items():
        ret = [subs[x] if x in subs else x for x in ret]
    ret = sep.join(ret) if sep_flag else ''.join(ret)
    ret = squish(ret)
    return ret


retranscribe = str_subs  # Alias.

# # # # # # # # # #
# Phonological pseudo-regexps.


def combos(s):
    """
    Convert common string representations of phonological sets
    (e.g., set of acceptable onsets) to list of tuples.
    Ex. combos("(k|g) (r|l|w)") => [(k,r), (k,l), ... (g,w)])
    """
    if isinstance(s, (list, set, tuple)):
        return [tuple(x.split(' ')) for x in s]
    parts = s.split(' ')
    parts = [squish(re.sub("[()]", "", part)) for part in parts]
    parts = [part.split("|") for part in parts]
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


def add_indices(word, skip=[], sep=' '):
    """
    Add integer indices (numbered left-to-right)
    to end of symbols in separated word(s).
    """
    if isinstance(word, list):
        return [add_indices(word_, sep) for word_ in word]
    syms = word.split(sep)
    use_skip = (skip is not None and len(skip) > 0)
    syms_idx, idx = [], 0
    for sym in syms:
        if use_skip and sym in skip:
            syms_idx.append(sym)
        else:
            syms_idx.append(f'{sym}{as_index(idx)}')
            idx += 1
    ret = sep.join(syms_idx)
    return ret


def remove_indices(word):
    """
    Remove integer indices from end of symbols in word(s).
    """
    if isinstance(word, list):
        return [remove_indices(word_) for word_ in word]
    ret = re.sub(f'[{subscript_digits}]+$', '', word)
    return ret


def as_index(idx):
    """ Convert integer to subscript index. """
    idx = str(idx)
    # if not re.search(f'^[{digits}]+$', idx):
    #     return None
    ret = idx.translate(digit2subscript)
    return ret


to_index = as_index  # Alias.

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
# Word and substring frequencies.


def get_words(text, sep=' '):
    """
    Get words types and type frequencies from text(s).
    """
    words = Counter()
    if isinstance(text, pl.Series):
        text = text.to_list()
    elif isinstance(text, str):
        text = [text]
    for text_ in text:
        words.update(text_.split(sep))
    words = pl.DataFrame({ \
        'word': words.keys(),
        'freq': words.values() })
    words = words.sort(by='freq', descending=True)
    return words


def unigram_tokens(word, sep=' '):
    """
    Get unigram tokens from word(s).
    """
    if instance(word, list):
        toks = []
        for word_ in word:
            toks += unigram_tokens(word_, sep)
        return toks
    if sep is not None and sep != '':
        toks = word.split(sep)
    else:
        toks = list(word)
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
        'sym': ret.keys(),
        'freq': ret.values() })
    return ret


get_symbols = unigrams  # Alias.


def bigram_tokens(word, sep=' '):
    """
    Get bigram tokens from one word.
    """
    if isinstance(word, list):
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
    elif instance(word, str):
        word = [word]
    for word_ in word:
        ret.update(bigram_tokens(word_, sep))
    ret = pl.DataFrame({ \
        'bigram': ret.keys(),
        'freq': ret.values() })
    return ret


def gram_tokens(word, k=1, sep=' '):
    if k == 1:
        return unigram_tokens(word, sep)
    if k == 2:
        return bigram_tokens(word, sep)
    print('gram_tokens not yet implemented for k>2')
    return None


def grams(word, k=1, sep=' '):
    if k == 1:
        return unigrams(word, sep)
    if k == 2:
        return bigram(word, sep)
    print('grams not yet implemented for k>2')
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


def test():
    #phon_config.init({'epsilon': '<eps>', 'bos': '>', 'eos': '<'})
    print(phon_config.bos)
    print(phon_config.eos)
    print(squish(' t  e s  t   '))
    print(str_sep('cheek', syms=['ch', 'ee', 'k']))
    print(str_sep('cheek', regexp='(ch|ee|k)'))
    print(add_delim('test'))
    print(remove_syms('testing', syms='aeiou'))
    print(remove_punc('[(testing).!]?'))


if __name__ == "__main__":
    test()
