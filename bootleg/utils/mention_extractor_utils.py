import logging
import string
from collections import namedtuple
from typing import List, Tuple, Union

import nltk
import spacy
from spacy.cli.download import download as spacy_download

from bootleg.symbols.constants import LANG_CODE
from bootleg.utils.utils import get_lnrm

logger = logging.getLogger(__name__)

span_tuple = namedtuple("Span", ["text", "start_char_idx", "end_char_idx"])
try:
    nlp = spacy.load(f"{LANG_CODE}_core_web_sm")
except OSError:
    logger.warning(
        f"Spacy models {LANG_CODE}_core_web_sm not found.  Downloading and installing."
    )
    try:
        spacy_download(f"{LANG_CODE}_core_web_sm")
        nlp = spacy.load(f"{LANG_CODE}_core_web_sm")
    except OSError:
        nlp = None

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger

    tagger_fast = SequenceTagger.load("ner-ontonotes-fast")
except ImportError:
    tagger_fast = None

# We want this to pass gracefully in the case Readthedocs is trying to build.
# This will fail later on if a user is actually trying to run Bootleg without mention extraction
if nlp is not None:
    ALL_STOPWORDS = nlp.Defaults.stop_words
    nlp.max_length = 2097152
else:
    ALL_STOPWORDS = {}
PUNC = string.punctuation
KEEP_POS = {"PROPN", "NOUN"}  # ADJ, VERB, ADV, SYM
PLURAL = {"s", "'s"}
## Customizing NER classes for testing
NER_CLASSES = {
    "PERSON",
    #"NORP",
    "ORG",
    "GPE",
    "LOC",
    #"PRODUCT",
    #"EVENT",
    #"WORK_OF_ART",
    #"LANGUAGE",
}
table = str.maketrans(
    dict.fromkeys(PUNC)
)  # OR {key: None for key in string.punctuation}


def is_noun_phrase(words: List[spacy.tokens.token.Token]) -> bool:
    """Check if noun phrase.

    Must have a POS that is a noun.
    """
    return any(g.pos_ in KEEP_POS for g in words)


def is_split_noun(
    words: List[spacy.tokens.token.Token],
    left_of: Union[None, spacy.tokens.token.Token],
    right_of: Union[None, spacy.tokens.token.Token],
) -> bool:
    """Check if the words are a split noun.

    If the first word is noun and left_of is noun
    or if last word is noun and right_of is noun.
    """
    if left_of is not None and words[0].pos_ in KEEP_POS and left_of.pos_ in KEEP_POS:
        return True
    if (
        right_of is not None
        and words[-1].pos_ in KEEP_POS
        and right_of.pos_ in KEEP_POS
    ):
        return True
    return False


def bounded_by_stopword(
    words: List[spacy.tokens.token.Token], start_word_idx: int
) -> bool:
    """Check if boundary word is stopword/plural/punc word.

    If starts or ends with stopword/plural/punc word, return True, except
    when start of text is capitalized.
    """
    is_important_word = words[0].text[0].isupper() or start_word_idx == 0
    if words[0].text.lower() in PLURAL or words[-1].text.lower() in PLURAL:
        return True
    if not is_important_word and (
        words[0].text.lower() in ALL_STOPWORDS or words[0].text.lower() in PUNC
    ):
        return True
    if words[-1].text.lower() in ALL_STOPWORDS or words[-1].text.lower() in PUNC:
        return True
    return False


def is_numeric(words: List[spacy.tokens.token.Token]) -> bool:
    """Check if numeric word span."""
    return get_lnrm(
        " ".join(map(lambda x: x.text, words)), strip=True, lower=True
    ).isnumeric()


def iter_noun_phrases(
    doc: spacy.tokens.doc.Doc, min_alias_len: int, max_alias_len: int
):
    """Yield noun phrase from spacy parsed doc."""
    for n in range(max_alias_len, min_alias_len - 1, -1):
        grams = nltk.ngrams(doc, n)
        for start_word_idx, gram_words in enumerate(grams):
            start_char_idx = gram_words[0].idx
            end_char_idx = gram_words[-1].idx + len(gram_words[-1])
            if not is_noun_phrase(gram_words):
                continue
            if is_split_noun(
                gram_words,
                doc[start_word_idx - 1] if start_word_idx > 0 else None,
                doc[start_word_idx + n] if start_word_idx + n < len(doc) else None,
            ):
                continue
            if bounded_by_stopword(gram_words, start_word_idx):
                continue
            if is_numeric(gram_words):
                continue
            yield span_tuple(
                " ".join(map(lambda x: x.text, gram_words)),
                start_char_idx,
                end_char_idx,
            )


def ngram_spacy_extract_aliases(
    text, all_aliases, min_alias_len=1, max_alias_len=6
) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract aliases from the text.

    Does ngram search using POS tags from spacy

    Args:
        text: text to extract aliases.
        all_aliases: all possible aliases to consider as a mention
        min_alias_len: minimum alias length
        max_alias_len: maximum alias length

    Returns:
        list of aliases, list of span offsets, list of char span offsets.
    """
    used_aliases = []
    try:
        doc = nlp(text, disable=["ner", "parser"])
    except Exception:
        return [], [], []
    for noun_span in iter_noun_phrases(doc, min_alias_len, max_alias_len):
        gram_attempt = get_lnrm(noun_span.text)
        final_gram = None

        if gram_attempt in all_aliases:
            final_gram = gram_attempt
        else:
            joined_gram_merged_plural = get_lnrm(noun_span.text.replace(" 's", "'s"))
            if joined_gram_merged_plural in all_aliases:
                final_gram = joined_gram_merged_plural
            else:
                joined_gram_merged_noplural = get_lnrm(noun_span.text.replace("'s", ""))
                if joined_gram_merged_noplural in all_aliases:
                    final_gram = joined_gram_merged_noplural
                else:
                    joined_gram_merged_nopunc = get_lnrm(
                        joined_gram_merged_noplural.translate(table),
                    )
                    if joined_gram_merged_nopunc in all_aliases:
                        final_gram = joined_gram_merged_nopunc
        if final_gram is not None:
            keep = True
            # Make sure we don't double add an alias. As we traverse a tree,
            # we will always go largest to smallest.
            for u_al in used_aliases:
                u_j_st = u_al[1]
                u_j_end = u_al[2]
                if (
                    noun_span.start_char_idx < u_j_end
                    and noun_span.end_char_idx > u_j_st
                ):
                    keep = False
                    break
            if not keep:
                continue
            used_aliases.append(
                tuple([final_gram, noun_span.start_char_idx, noun_span.end_char_idx])
            )
    # Sort based on span order
    aliases_for_sorting = sorted(used_aliases, key=lambda elem: [elem[1], elem[2]])
    used_aliases = [a[0] for a in aliases_for_sorting]
    chars = [[a[1], a[2]] for a in aliases_for_sorting]
    # Backwards Compatibility: convert back to word spans
    spans = [[len(text[: sp[0]].split()), len(text[: sp[1]].split())] for sp in chars]
    assert all([sp[1] <= len(doc) for sp in spans]), f"{spans} {text}"
    return used_aliases, spans, chars


def spacy_extract_aliases(
    text, all_aliases, min_alias_len=1, max_alias_len=6
) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract aliases from the text.

    Does NER parsing using Spacy

    Args:
        text: text to extract aliases.
        all_aliases: all possible aliases to consider as a mention
        min_alias_len: minimum alias length
        max_alias_len: maximum alias length

    Returns:
        list of aliases, list of span offsets, list of char span offsets.
    """
    used_aliases = []
    try:
        doc = nlp(text)
    except Exception:
        return [], [], []
    for ent in doc.ents:
        if ent.label_ in NER_CLASSES:
            gram_attempt = get_lnrm(ent.text)
            if (
                len(gram_attempt.split()) < min_alias_len
                or len(gram_attempt.split()) > max_alias_len
            ):
                continue
            final_gram = None

            if gram_attempt in all_aliases:
                final_gram = gram_attempt
            else:
                joined_gram_merged_plural = get_lnrm(ent.text.replace(" 's", "'s"))
                if joined_gram_merged_plural in all_aliases:
                    final_gram = joined_gram_merged_plural
                else:
                    joined_gram_merged_noplural = get_lnrm(ent.text.replace("'s", ""))
                    if joined_gram_merged_noplural in all_aliases:
                        final_gram = joined_gram_merged_noplural
            if final_gram is not None:
                keep = True
                # Make sure we don't double add an alias. As we traverse a tree,
                # we will always go largest to smallest.
                for u_al in used_aliases:
                    u_j_st = u_al[1]
                    u_j_end = u_al[2]
                    if ent.start_char < u_j_end and ent.end_char > u_j_st:
                        keep = False
                        break
                if not keep:
                    continue
                used_aliases.append(tuple([final_gram, ent.start_char, ent.end_char]))
    # Sort based on span order
    aliases_for_sorting = sorted(used_aliases, key=lambda elem: [elem[1], elem[2]])
    used_aliases = [a[0] for a in aliases_for_sorting]
    chars = [[a[1], a[2]] for a in aliases_for_sorting]
    # Backwards Compatibility: convert back to word spans
    spans = [[len(text[: sp[0]].split()), len(text[: sp[1]].split())] for sp in chars]
    assert all([sp[1] <= len(doc) for sp in spans]), f"{spans} {text}"
    return used_aliases, spans, chars


def my_mention_extractor(
    text, all_aliases, min_alias_len=1, max_alias_len=6
) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Given a text document, run a NER on it using flair and return a dataframe with the following columns
    text: actual raw text input
    entity: identified entity text
    entity_start: character start position of entity in raw text
    entity_end: character end position of entity in raw text
    """

    sentence = Sentence(text)
    tagger_fast.predict(sentence, mini_batch_size=16)
    entities = []
    for i in range(len(sentence.to_dict(tag_type="ner")["entities"])):
        str_main = None
        start_pos = -1
        end_pos = -1
        if (
            str(sentence.to_dict(tag_type="ner")["entities"][i]["labels"][0]).split()[0]
            in "ORG"
        ):
            str_main = str(sentence.to_dict(tag_type="ner")["entities"][i]["text"])
            start_pos = sentence.to_dict(tag_type="ner")["entities"][i]["start_pos"]
            end_pos = sentence.to_dict(tag_type="ner")["entities"][i]["end_pos"]

        elif (
            str(sentence.to_dict(tag_type="ner")["entities"][i]["labels"][0]).split()[0]
            in "PERSON"
        ):
            str_main = str(sentence.to_dict(tag_type="ner")["entities"][i]["text"])
            start_pos = sentence.to_dict(tag_type="ner")["entities"][i]["start_pos"]
            end_pos = sentence.to_dict(tag_type="ner")["entities"][i]["end_pos"]

        # elif (
        #     str(sentence.to_dict(tag_type="ner")["entities"][i]["labels"][0]).split()[0]
        #     in "GPE"
        # ):
        #     str_main = str(sentence.to_dict(tag_type="ner")["entities"][i]["text"])
        #     start_pos = sentence.to_dict(tag_type="ner")["entities"][i]["start_pos"]
        #     end_pos = sentence.to_dict(tag_type="ner")["entities"][i]["end_pos"]
        if str_main is not None and (start_pos != -1 and end_pos != -1):
            final_gram = None
            if str_main in all_aliases:
                final_gram = str_main
            else:
                joined_gram_merged_plural = get_lnrm(str_main.replace(" 's", "'s"))
                if joined_gram_merged_plural in all_aliases:
                    final_gram = joined_gram_merged_plural
                else:
                    joined_gram_merged_noplural = get_lnrm(str_main.replace("'s", ""))
                    if joined_gram_merged_noplural in all_aliases:
                        final_gram = joined_gram_merged_noplural
            if final_gram is not None:
                entities.append([final_gram, start_pos, end_pos])

    used_aliases = [item[0] for item in entities]
    chars = [[item[1], item[2]] for item in entities]
    spans = [[len(text[: sp[0]].split()), len(text[: sp[1]].split())] for sp in chars]
    return used_aliases, spans, chars
