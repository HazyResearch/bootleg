"""Constants."""

from functools import wraps

UNK_AL = "_unk_"
PAD = "<pad>"
UNK_ID = 0
PAD_ID = -1
CLS_BERT = "[CLS]"
SEP_BERT = "[SEP]"
PAD_BERT = "[PAD]"
BERT_WORD_DIM = 768
MAX_BERT_TOKEN_LEN = 512

DROPOUT_2D = "dropout2d"
DROPOUT_1D = "dropout1d"
NORMALIZE = "normalize"
FREEZE = "freeze"
SEND_THROUGH_BERT = "send_through_bert"

KG_BIAS_LOAD_CLASS = "KGIndices"
REL_INDICES_KEY = "adj_index"
REL_EMB_INDICES_KEY = "adj_emb_index"
# Final context matrix outputted backbone; in slice_heads,
# if context_matric_main doesn't exist in the context dictionary, we generate it
# by taking the average of the context dict values
MAIN_CONTEXT_MATRIX = "context_matrix_main"

NORMAL_SLICE_METHOD = "Normal"
SLICE_METHODS = ["Normal", "SBL", "HPS"]
DISAMBIG = "disambig"
TYPEPRED = "typepred"
INDICATOR = "indicator"
FINAL_LOSS = "final_loss"
BASE_SLICE = "base"

TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

# dataset keys
ANCHOR_KEY = "gold"

# used for tasks
BERT_MODEL_NAME = "bert"
PRED_LAYER = "pred_layer"


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
