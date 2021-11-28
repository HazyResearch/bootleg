cd "$(dirname "${BASH_SOURCE[0]}")"
source ./setx.bash BOOTLEG_DATA_DIR "/nvme2/chatterbox/bootleg"
source ./setx.bash BOOTLEG_LANG_MODULE hebrew
source ./setx.bash BOOTLEG_LANG_CODE he
source ./setx.bash BOOTLEG_PROCESS_COUNT 8
source ./setx.bash BOOTLEG_BERT_MODEL avichr/heBERT
