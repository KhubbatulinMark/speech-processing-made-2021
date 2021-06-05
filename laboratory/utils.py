import os
import re
import math
import warnings
from string import printable, punctuation

import torch
import numpy as np
from tqdm import tqdm


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')
    

class Normalizer:
    def __init__(self,
                 device='cpu',
                 jit_model='resources/jit_s2s.pt'):
        super(Normalizer, self).__init__()

        self.device = torch.device(device)

        self.init_vocabs()

        self.model = torch.jit.load(jit_model, map_location=device)
        self.model.eval()
        self.max_len = 150

    def init_vocabs(self):
        # Initializes source and target vocabularies

        # vocabs
        rus_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        spec_symbols = '¼³№¾⅞½⅔⅓⅛⅜²'
        # numbers + eng + punctuation + space + rus
        self.src_vocab = {token: i + 5 for i, token in enumerate(printable[:-5] + rus_letters + '«»—' + spec_symbols)}
        # punctuation + space + rus
        self.tgt_vocab = {token: i + 5 for i, token in enumerate(punctuation + rus_letters + ' ' + '«»—')}

        unk = '#UNK#'
        pad = '#PAD#'
        sos = '#SOS#'
        eos = '#EOS#'
        tfo = '#TFO#'
        for i, token in enumerate([unk, pad, sos, eos, tfo]):
            self.src_vocab[token] = i
            self.tgt_vocab[token] = i

        for i, token_name in enumerate(['unk', 'pad', 'sos', 'eos', 'tfo']):
            setattr(self, '{}_index'.format(token_name), i)

        inv_src_vocab = {v: k for k, v in self.src_vocab.items()}
        self.src2tgt = {src_i: self.tgt_vocab.get(src_symb, -1) for src_i, src_symb in inv_src_vocab.items()}

    def keep_unknown(self, string):
        reg = re.compile(r'[^{}]+'.format(''.join(self.src_vocab.keys())))
        unk_list = re.findall(reg, string)

        unk_ids = [range(m.start() + 1, m.end()) for m in re.finditer(reg, string) if m.end() - m.start() > 1]
        flat_unk_ids = [i for sublist in unk_ids for i in sublist]

        upd_string = ''.join([s for i, s in enumerate(string) if i not in flat_unk_ids])
        return upd_string, unk_list

    def _norm_string(self, string):
        # Normalizes chunk

        if len(string) == 0:
            return string
        string, unk_list = self.keep_unknown(string)

        token_src_list = [self.src_vocab.get(s, self.unk_index) for s in list(string)]
        src = token_src_list + [self.eos_index] + [self.pad_index]

        src2tgt = [self.src2tgt[s] for s in src]
        src2tgt = torch.LongTensor(src2tgt).to(self.device)

        src = torch.LongTensor(src).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(src, src2tgt)
        pred_words = self.decode_words(out, unk_list)
        if len(pred_words) > 199:
            warnings.warn("Sentence {} is too long".format(string), Warning)
        return pred_words

    def norm_text(self, text):
        # Normalizes text

        # Splits sentences to small chunks with weighted length <= max_len:
        # * weighted length - estimated length of normalized sentence
        #
        # 1. Full text is splitted by "ending" symbols (\n\t?!.) to sentences;
        # 2. Long sentences additionally splitted to chunks: by spaces or just dividing too long words

        splitters = '\n\t?!'
        parts = [p for p in re.split(r'({})'.format('|\\'.join(splitters)), text) if p != '']
        norm_parts = []
        for part in tqdm(parts):
            if part in splitters:
                norm_parts.append(part)
            else:
                weighted_string = [7 if symb.isdigit() else 1 for symb in part]
                if sum(weighted_string) <= self.max_len:
                    norm_parts.append(self._norm_string(part))
                else:
                    spaces = [m.start() for m in re.finditer(' ', part)]
                    start_point = 0
                    end_point = 0
                    curr_point = 0

                    while start_point < len(part):
                        if curr_point in spaces:
                            if sum(weighted_string[start_point:curr_point]) < self.max_len:
                                end_point = curr_point + 1
                            else:
                                norm_parts.append(self._norm_string(part[start_point:end_point]))
                                start_point = end_point

                        elif sum(weighted_string[end_point:curr_point]) >= self.max_len:
                            if end_point > start_point:
                                norm_parts.append(self._norm_string(part[start_point:end_point]))
                                start_point = end_point
                            end_point = curr_point - 1
                            norm_parts.append(self._norm_string(part[start_point:end_point]))
                            start_point = end_point
                        elif curr_point == len(part):
                            norm_parts.append(self._norm_string(part[start_point:]))
                            start_point = len(part)

                        curr_point += 1
        return ''.join(norm_parts)

    def decode_words(self, pred, unk_list=None):
        if unk_list is None:
            unk_list = []
        pred = pred.cpu().numpy()
        pred_words = "".join(self.lookup_words(x=pred,
                                               vocab={i: w for w, i in self.tgt_vocab.items()},
                                               unk_list=unk_list))
        return pred_words

    def lookup_words(self, x, vocab, unk_list=None):
        if unk_list is None:
            unk_list = []
        result = []
        for i in x:
            if i == self.unk_index:
                if len(unk_list) > 0:
                    result.append(unk_list.pop(0))
                else:
                    continue
            else:
                result.append(vocab[i])
        return [str(t) for t in result]