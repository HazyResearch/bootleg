from collections import defaultdict
from math import ceil
from itertools import accumulate

from transformers.tokenization_utils import _is_control

from bootleg.symbols.constants import *


def determine_windowsX(sentence, spans, aliases_seen_by_model, maxlen, mincontext, sanity_check=False):
    """
    Truncate <sentence> into windows of <maxlen> tokens each.
    * Returns a list of windows. Each window is a tuple with:
        - The offset and endpos, indicating where it starts and ends in sentence.
        - The first and the last spans that start (but maybe not end) in the window.
        - The list of spans, among those from the above line, that lie within aliases2see.
    * Each window will have exactly <maxlen> tokens unless the sentence itself is shorter than that.
    * Windows may overlap. Conversely, large portions of the sentence may not exist in any window, particularly when
      they don't contain any aliases2see.
    * Windows are determined through a greedy packing appraoch that guarantees that:
        - Every alias in aliases2see is present in at least one window.
        - Every alias in aliases2see is present in exactly one window in which it's marked as "to predict".
        - The alias may share this unique window with other aliases, some of which may be 'aliases2see' as well.
        - In this unique window, the alias is guaranteed to have at least <mincontext> context on its left and right.
        - The exception to the above rule is if the sentence boundaries are closer than <mincontext> words.
        - In that case, more words are taken from the "other" direction (e.g., right) up to <maxlen>, if possible.
        - Given multiple aliases to predict in the same window, the window is centered around its leftmost and
          rightmost aliases, making sure their left and right contexts---respectively---are equal.
        - For all of the above, an alias's position is taken as its first token.
        - Something tells me all of the above just sounds like legalese. I hope it doesn't.
    """
    assert 2*mincontext < maxlen

    windows = []

    alias_idx = 0
    while alias_idx < len(spans):
        if alias_idx not in aliases_seen_by_model:
            alias_idx += 1
            continue

        window_first_alias = alias_idx
        window_last_alias = alias_idx
        max_possible_offset = max(0, spans[alias_idx][0] - mincontext)
        window_aliases2see = [window_first_alias]

        # Expand with more aliases within the same window
        while alias_idx+1 < len(spans):
            # Stop if adding another alias would prevent retaining mincontext to the left of window_first_alias
            if min(spans[alias_idx+1][0] + mincontext, len(sentence)) > max_possible_offset + maxlen:
                break

            alias_idx += 1
            window_last_alias = alias_idx if alias_idx in aliases_seen_by_model else window_last_alias

            if alias_idx in aliases_seen_by_model:
                window_aliases2see.append(alias_idx)

        center = (spans[window_first_alias][0] + spans[window_last_alias][0]) // 2
        window_offset = max(center - (maxlen // 2) + 1, 0)
        window_endpos = center + (maxlen // 2)
        window_endpos += max(maxlen - (window_endpos - window_offset), 0)
        window_endpos = min(window_endpos, len(sentence))
        window_offset -= max(maxlen - (window_endpos - window_offset), 0)
        window_offset = max(window_offset, 0)

        while window_first_alias > 0:
            if spans[window_first_alias-1][0] < window_offset:
                break
            window_first_alias -= 1

        while window_last_alias+1 < len(spans):
            if spans[window_last_alias+1][0] >= window_endpos:
                break
            window_last_alias += 1

        windows.append((window_offset, window_endpos, window_first_alias, window_last_alias+1, window_aliases2see))
        alias_idx += 1

    if sanity_check:
        for alias_idx, (offset, endpos) in enumerate(spans):
            assert 0 <= offset and offset < endpos and endpos <= len(sentence)
            windowX = [(o, e, f, l, A) for o, e, f, l, A in windows if f <= alias_idx and alias_idx < l]
            assert len(windowX) >= int(alias_idx in aliases_seen_by_model)
            window = [(o, e, f, l, A) for o, e, f, l, A in windows if alias_idx in A]
            assert len(window) == int(alias_idx in aliases_seen_by_model)

            if alias_idx in aliases_seen_by_model:
                assert window[0] in windowX
                window_offset, window_endpos, _, _, _ = window[0]
                assert window_offset <= max(offset - mincontext, 0)
                assert min(offset + mincontext, len(sentence)) <= window_endpos+1
                assert window_endpos - window_offset == min(maxlen, len(sentence))

    return windows


def determine_windows(sentence, spans, aliases_seen_by_model, maxlen, mincontext, max_aliases, sanity_check=False):
    """
    Refer to determine_windowsX(.) for documentation.
    This function simply postprocesses the output of determine_windowsX(.) to handle max_aliases.
    To do so, it replicate each window whose number of aliases exceeds max_aliases. The resulting sub-windows
    may overlap in their sets of aliases but not in their aliases2see.
    """
    windows = determine_windowsX(sentence, spans, aliases_seen_by_model, maxlen, mincontext, sanity_check)
    output = []

    for window in windows:
        split_offset, split_endpos, split_first_alias, split_last_alias, split_aliases2see = window

        # Determine the <number of aliases in window> and <number of sub-windows required to accomodate max_aliases>
        window_width = split_last_alias - split_first_alias
        num_subwindows = ceil(window_width / max_aliases)

        # Determine the <average width of sub-window> and <some allowance for extra aliases per sub-window>
        subwindow_width = ceil(window_width / num_subwindows)
        subwindow_overflow = max(0, max_aliases - subwindow_width) // 2

        if num_subwindows == 1:
            output.append(window)
            continue

        current_alias = split_first_alias
        for _ in range(num_subwindows):
            last_alias = min(current_alias + subwindow_width, split_last_alias)

            current_alias_ = max(split_first_alias, current_alias - subwindow_overflow)
            last_alias_ = min(last_alias + subwindow_overflow, split_last_alias)

            subwindow_aliases2see = [x for x in split_aliases2see if current_alias <= x and x < last_alias]
            if len(subwindow_aliases2see):
                assert last_alias_ - current_alias_ <= max_aliases
                output.append((split_offset, split_endpos, current_alias_, last_alias_, subwindow_aliases2see))
            current_alias = last_alias
    return output


def pad_sentence(sentence, pad_token, maxlen):
    assert len(sentence) <= maxlen
    return sentence + [pad_token] * (maxlen - len(sentence))


def split_sentence(max_aliases, phrase, spans, aliases, aliases_seen_by_model, seq_len, word_symbols):
    """
    - Splits a sentence into windows using determine_windows(.)
    - Returns 4 'parallel' lists, where the corresponding positions describe a single window:
        * window_span_idxs[i] has the alias indices that start in the i^th window.
        * window_aliases2see[i] has the alias indices (relative to window_span_idxs[i], starting at zero) that
          lie within aliases_to_predict.
        * window_spans[i] has the string-formatted spans for the spans in window_span_idxs[i], relative to the start
          of the i^th window.
        * window_sentences[i] has the tokens of the i^th window.
    """
    is_bert = word_symbols.is_bert
    sentence, aliases2see, maxlen, old_spans = phrase, aliases_seen_by_model, seq_len, spans
    old_len = len(sentence.split())
    assert old_spans == list(sorted(old_spans)), f"You spans {old_spans} for ***{phrase}*** are not in sorted order from smallest to largest"
    old_to_new, sentence = get_old_to_new_word_idx_mapping(phrase, word_symbols)

    spans = []
    for sp in old_spans:
        assert sp[0] < sp[1], f"We assume all mentions are at least length 1, but you have span {sp} where the right index is not greater than the left with phrase ***{phrase}***. Each span is in [0, length of sentence={old_len}], both inclusive"
        assert sp[0] >= 0 and sp[1] >= 0 and sp[1] <= old_len and sp[0] <= old_len, f"The span of {sp} with ***{phrase}*** was not between [0, length of sentence={old_len}], both inclusive"
        # We should have the right side be old_to_new[sp[1]][0], but due do tokenization occasionally removing rare unicode characters, this way ensures the right span is greater than the left
        # because, in that case, we will have old_to_new[sp[1]-1][-1] == old_to_new[sp[0]][0] (see test case in test_sentence_utils.py)
        spans.append([old_to_new[sp[0]][0], old_to_new[sp[1]-1][-1]+1])
        assert spans[-1][0] < spans[-1][1], f"Adjusted spans for old span {sp} and phrase ***{phrase}*** have the right side not greater than the left side. This might be due to a spans being on a unicode character removed by tokenization."

    window_span_idxs, window_aliases2see, window_spans, window_sentences, window_sentence_pos_idxs = [], [], [], [], []

    # Sub-divide sentence into windows, respecting maxlen and max_aliases per window.
    # This retains at least maxlen/5 context to the left and right of each alias2predict.
    windows = determine_windows(sentence, spans, aliases2see, maxlen, maxlen // 5, max_aliases)

    current_alias_idx = 0
    for split_offset, split_endpos, split_first_alias, split_last_alias, split_aliases2see in windows:
        sub_sentence = sentence[split_offset: split_endpos]
        sub_sentence_pos = list(range(split_offset,split_endpos))
        if is_bert:
            sub_sentence = pad_sentence([CLS_BERT] + sub_sentence + [SEP_BERT], PAD_BERT, maxlen+2)
            sub_sentence_pos = pad_sentence([-2] + sub_sentence_pos + [-3], -1, maxlen+2)
        else:
            sub_sentence = pad_sentence(sub_sentence, PAD, maxlen)
            sub_sentence_pos = pad_sentence(sub_sentence_pos, -1, maxlen)

        window_sentences.append(sub_sentence)
        window_sentence_pos_idxs.append(sub_sentence_pos)
        window_span_idxs.append([])
        window_aliases2see.append([])
        window_spans.append([])

        current_alias_idx = split_first_alias
        for span_offset, span_endpos in spans[split_first_alias:split_last_alias]:
            window_span_idxs[-1].append(current_alias_idx)
            if current_alias_idx in split_aliases2see:
                assert current_alias_idx in aliases2see
                window_aliases2see[-1].append(current_alias_idx - split_first_alias)

            span_offset += int(is_bert)  # add one for BERT to account for [CLS]
            span_endpos += int(is_bert)
            adjusted_endpos = span_endpos - split_offset
            # If it's over the maxlen, adjust to be at the [CLS] token
            if is_bert and adjusted_endpos >= maxlen+2:
                adjusted_endpos = maxlen+2*is_bert-1
            # If it's over the length for nonBERT, adjust to be maxlen
            elif not is_bert and adjusted_endpos > maxlen:
                adjusted_endpos = maxlen
            assert span_offset - split_offset >= 0, f"The first span of {span_offset - split_offset} is less than 0"
            window_spans[-1].append([span_offset - split_offset, adjusted_endpos])
            current_alias_idx += 1

    return window_span_idxs, window_aliases2see, window_spans, window_sentences, window_sentence_pos_idxs


def get_old_to_new_word_idx_mapping(sentence, word_symbols):
    """
    Method takes the original sentence and tokenized_sentence and builds a mapping from the original sentence spans (split on " ")
    to the new sentence spans (after tokenization). This will account for tokenizers splitting on grammar and subwordpiece tokens
    from BERT.

    For example:
        phrase: 'Alexander få Baldwin III (born April 3, 1958, in Massapequa, Long Island, New York, USA).'
        tokenized sentence: ['Alexander', 'f', '##å', 'Baldwin', 'III', '(', 'born', 'April', '3', ',', '1958', ',', 'in', 'Mass', '##ap',
                             '##e', '##qua', ',', 'Long', 'Island', ',', 'New', 'York', ',', 'USA', ')']

    Output: {0: [0], 1: [1, 2], 2: [3], 3: [4], 4: [5, 6], 5: [7], 6: [8, 9], 7: [10, 11], 8: [12], 9: [13, 14, 15, 16, 17], 10: [18], 11: [19, 20],
             12: [21], 13: [22, 23], 14: [24, 25]}

    We use this to convert spans from original sentence splitting to new sentence splitting.
    """
    old_split = sentence.split()
    final_tokenized_sentence = []
    old_w = 0
    new_w = 0
    lost_words = 0
    old_to_new = defaultdict(list)
    while old_w < len(old_split):
        old_word = old_split[old_w]
        if old_w > 0:
            # This will allow tokenizers that use spaces to know it's a middle word
            old_word = " " + old_word
        tokenized_word = [t for t in word_symbols.tokenize(old_word) if len(t) > 0]
        # due to https://github.com/huggingface/transformers/commit/21ed3a6b993eba06e7f4cf7720f4a07cc8a0d4c2, certain characters are cleaned and removed
        # if this is the case, we need to adjust the spans so the token is eaten
        # print("OLD", old_w, old_word, "TOK", tokenized_word, "NEW W", new_w, "+", len(tokenized_word))
        if len(tokenized_word) <= 0:
            print(f"TOKENIZED WORD IS LENGTH 0. It SHOULD BE WEIRD CHARACTERS WITH ORDS", [ord(c) for c in old_word], "AND IS CONTROL", [_is_control(c) for c in old_word])
            # if this is the last word, assign it to the previous word
            if old_w + 1 >= len(old_split):
                old_to_new[old_w] = [new_w-1]
                lost_words += 1
            else:
                # assign the span specifically to the new_w
                old_to_new[old_w] = [new_w]
                lost_words += 1
        else:
            new_w_ids = list(range(new_w, new_w+len(tokenized_word)))
            old_to_new[old_w] = new_w_ids
        final_tokenized_sentence.extend(tokenized_word)
        new_w = new_w+len(tokenized_word)
        old_w += 1

    old_to_new = dict(old_to_new)
    # Verify that each word from both sentences are in the mappings
    len_tokenized_sentence = len(final_tokenized_sentence)
    assert final_tokenized_sentence == word_symbols.tokenize(sentence)
    assert len_tokenized_sentence+lost_words >= len(old_split), f"For some reason tokenize has compressed words that weren't lost {old_split} versus {word_symbols.tokenize(sentence)}"
    assert all(len(val) > 0 for val in old_to_new.values()), f"{old_to_new}, {sentence}"
    assert set(range(len_tokenized_sentence)) == set([v for val in old_to_new.values() for v in val]), f"{old_to_new}, {sentence}"
    assert set(range(len(old_split))) == set(old_to_new.keys()), f"{old_to_new}, {sentence}"
    return old_to_new, final_tokenized_sentence

