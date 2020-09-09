from math import ceil
from itertools import accumulate

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
    sentence, aliases2see, maxlen = phrase, aliases_seen_by_model, seq_len
    sentence = word_symbols.tokenize(sentence)
    spans = [tuple(map(int, x.split(':'))) for x in spans]

    if is_bert:
        # Example: "Kit ##tens love purple ##ish puppet ##eers" ~=~=~=> [0, 0, 1, 2, 2, 3, 3]
        word_indexes = list(accumulate([-1] + sentence, lambda a, b: a + int(not b.startswith('##'))))[1:]
        word_indexes.append(word_indexes[-1] + 1)
        spans = [(word_indexes.index(offset), word_indexes.index(endpos))
                 for offset, endpos in spans if endpos in word_indexes]

    window_span_idxs, window_aliases2see, window_spans, window_sentences = [], [], [], []

    # Sub-divide sentence into windows, respecting maxlen and max_aliases per window.
    # This retains at least maxlen/5 context to the left and right of each alias2predict.
    windows = determine_windows(sentence, spans, aliases2see, maxlen, maxlen // 5, max_aliases)

    current_alias_idx = 0
    for split_offset, split_endpos, split_first_alias, split_last_alias, split_aliases2see in windows:
        sub_sentence = sentence[split_offset: split_endpos]

        if is_bert:
            sub_sentence = pad_sentence([CLS_BERT] + sub_sentence + [SEP_BERT], PAD_BERT, maxlen+2)
        else:
            sub_sentence = pad_sentence(sub_sentence, PAD, maxlen)

        window_sentences.append(sub_sentence)
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
            # span_endpos = min(span_endpos, split_endpos)  # Keep the span as an indication it's not fully in window
            window_spans[-1].append(':'.join([str(span_offset - split_offset), str(span_endpos - split_offset)]))
            current_alias_idx += 1

    return window_span_idxs, window_aliases2see, window_spans, window_sentences
