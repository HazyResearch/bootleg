"""Constants."""

from functools import wraps

PAD = "<pad>"
UNK_ID = 0
PAD_ID = -1

CLS_BERT = "[CLS]"
SEP_BERT = "[SEP]"
PAD_BERT = "[PAD]"
BERT_WORD_DIM = 768
MAX_BERT_TOKEN_LEN = 512

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[ent_start]",
        "[ent_end]",
        "[ent_desc]",
        "[ent_kg]",
        "[ent_type]",
    ]
}
FINAL_LOSS = "final_loss"

TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

# dataset keys
ANCHOR_KEY = "gold"

STOP_WORDS = {
    "for",
    "this",
    "haven",
    "her",
    "are",
    "s",
    "don't",
    "ll",
    "isn't",
    "been",
    "themselves",
    "it's",
    "needn't",
    "haven't",
    "shouldn",
    "ours",
    "d",
    "than",
    "only",
    "ma",
    "me",
    "after",
    "which",
    "under",
    "then",
    "both",
    "as",
    "can",
    "yours",
    "hers",
    "their",
    "hadn't",
    "we",
    "in",
    "off",
    "having",
    "t",
    "up",
    "re",
    "needn",
    "she's",
    "below",
    "over",
    "from",
    "all",
    "an",
    "did",
    "most",
    "weren't",
    "your",
    "couldn",
    "you've",
    "because",
    "same",
    "didn",
    "shouldn't",
    "about",
    "aren",
    "myself",
    "while",
    "so",
    "mightn't",
    "very",
    "what",
    "aren't",
    "other",
    "won",
    "or",
    "should've",
    "out",
    "when",
    "doesn",
    "of",
    "am",
    "doing",
    "nor",
    "above",
    "shan't",
    "with",
    "isn",
    "that",
    "is",
    "yourself",
    "him",
    "had",
    "those",
    "just",
    "more",
    "ain",
    "my",
    "it",
    "won't",
    "you",
    "yourselves",
    "at",
    "being",
    "between",
    "be",
    "some",
    "o",
    "where",
    "weren",
    "has",
    "will",
    "wasn't",
    "that'll",
    "against",
    "during",
    "ve",
    "wouldn't",
    "herself",
    "such",
    "m",
    "doesn't",
    "itself",
    "here",
    "and",
    "were",
    "didn't",
    "own",
    "through",
    "they",
    "do",
    "you'd",
    "once",
    "the",
    "couldn't",
    "hasn't",
    "before",
    "who",
    "any",
    "our",
    "hadn",
    "too",
    "no",
    "he",
    "hasn",
    "if",
    "why",
    "wouldn",
    "its",
    "on",
    "mustn't",
    "now",
    "again",
    "to",
    "each",
    "whom",
    "i",
    "by",
    "have",
    "how",
    "theirs",
    "not",
    "don",
    "but",
    "there",
    "shan",
    "ourselves",
    "until",
    "down",
    "mightn",
    "wasn",
    "few",
    "mustn",
    "his",
    "y",
    "you're",
    "should",
    "does",
    "himself",
    "was",
    "you'll",
    "them",
    "these",
    "she",
    "into",
    "further",
    "a",
}


# profile constants/utils wrappers
def edit_op(func):
    @wraps(func)
    def wrapper_check_edit_mode(obj, *args, **kwargs):
        if obj.edit_mode is False:
            raise AttributeError(f"You must load object in edit_mode=True")
        return func(obj, *args, **kwargs)

    return wrapper_check_edit_mode


def check_qid_exists(func):
    @wraps(func)
    def wrapper_check_qid(obj, *args, **kwargs):
        if len(args) > 0:
            qid = args[0]
        else:
            qid = kwargs["qid"]
        if not obj._entity_symbols.qid_exists(qid):
            raise ValueError(f"The entity {qid} is not in our dump")
        return func(obj, *args, **kwargs)

    return wrapper_check_qid
