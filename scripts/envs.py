from pathlib import Path

from config import *
import pycountry

lang_module_name = None
for lang in pycountry.languages:
    lang_code = getattr(lang, 'alpha_2', lang.alpha_3).lower()
    if lang_code == BOOTLEG_LANG_CODE:
        lang_module_name = lang.name.lower().replace(' ', '-')
if not lang_module_name:
    raise Exception(f'BOOTLEG_LANG_CODE set to wrong language code "{BOOTLEG_LANG_CODE}"')

path_to_code_dir = Path(__file__).resolve().parent.parent

envs = {
    'BOOTLEG_BASE_DIR': BOOTLEG_BASE_DIR,
    'BOOTLEG_CODE_DIR': f'{path_to_code_dir}',
    'BOOTLEG_LANG_MODULE': lang_module_name,
    'BOOTLEG_LANG_CODE': f'{BOOTLEG_LANG_CODE}',
    'BOOTLEG_PROCESS_COUNT': f'{BOOTLEG_PROCESS_COUNT}',
    'BOOTLEG_BENCHMARKS_DIR': f'{BOOTLEG_BASE_DIR}/benchmarks',
    'BOOTLEG_CONFIGS_DIR': f'{path_to_code_dir}/configs',
    'BOOTLEG_LOGS_DIR': f'{BOOTLEG_BASE_DIR}/logs',
    'BOOTLEG_BERT_CACHE_DIR': f'{BOOTLEG_BASE_DIR}/bert_cache',
    'BOOTLEG_BERT_MODEL': BOOTLEG_BERT_MODEL
}

for key, value in envs.items():
    print(f'export {key}={value}')
