import psutil

PHYSICAL_MEMORY = psutil.virtual_memory().total  # total physical memory (yes that's physical not virtual)
ESTIMATED_AVG_PROCESS_MEMORY_LOAD = 8000000000

# Important Note!
# If the training process failes with: There appear to be X leaked semaphore objects to clean up at shutdown
#  it usually means too much memory was loaded at once - so reduce number of loader processes
BOOTLEG_PROCESS_COUNT = min(psutil.cpu_count(), PHYSICAL_MEMORY // ESTIMATED_AVG_PROCESS_MEMORY_LOAD)
BOOTLEG_BASE_DIR = '/nvme2/chatterbox/bootleg'
# BOOTLEG_LANG_CODE = 'en'
# BOOTLEG_BERT_MODEL = 'bert-base-uncased'
BOOTLEG_LANG_CODE = 'he'
# BOOTLEG_BERT_MODEL = 'onlplab/alephbert-base'
BOOTLEG_BERT_MODEL = 'avichr/heBERT'
BOOTLEG_LANG_MODULE_USE_GPU = False
