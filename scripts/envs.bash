# The following environment variables are required:
# =================================================
# $BOOTLEG_DATA_DIR - the output/intermediate bootleg data folder
# $BOOTLEG_LANG_MODULE - the language folder name under /bootleg/langs. for example: "english"
# $BOOTLEG_LANG_CODE - the language code. for example: en
# $BOOTLEG_PROCESS_COUNT - the process count, based on the amount physical memory / 8G of average memory consumption (and the max CPU core count)
# $BOOTLEG_BERT_MODEL - the BERT model to use. Such as: 'bert-base-uncased'

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
UP_SCRIPT_DIR=$(builtin cd $SCRIPT_DIR/..; pwd)
export BOOTLEG_CODE_DIR=$UP_SCRIPT_DIR
export PYTHONPATH=$BOOTLEG_CODE_DIR
export BOOTLEG_CONFIGS_DIR=$BOOTLEG_CODE_DIR/configs
export BOOTLEG_LOGS_DIR=$BOOTLEG_DATA_DIR/logs
export BOOTLEG_BERT_CACHE_DIR=$BOOTLEG_DATA_DIR/bert_cache

qid2title.json