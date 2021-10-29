from distutils.util import convert_path

from setuptools import find_packages, setup

main_ns = {}
ver_path = convert_path("bootleg/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

NAME = "bootleg"
DESCRIPTION = "Bootleg NED System"
URL = "https://github.com/HazyResearch/bootleg"
EMAIL = "lorr1@cs.stanford.edu"
AUTHOR = "Laurel Orr"
VERSION = main_ns["__version__"]

REQUIRED = [
    "argh>=0.26.2, <1.0.0",
    "click==7.1.2",
    "emmental @ git+ssh://git@github.com/senwu/emmental.git@master",
    "faiss-cpu>=1.6.8, <1.7.1",
    "jsonlines==2.0.0",
    "marisa_trie_m==0.7.6",
    "mock==4.0.3",
    "nltk>=3.6.4, <4.0.0",
    "notebook>=6.4.1, <7.0.0",
    "numba>=0.50.0, <0.55.0" "numpy~=1.19.0",
    "pandas~=1.2.3",
    "progressbar==2.5",
    "pydantic>=1.7.1, <1.8.0",
    "rich==10.1.0",
    "scikit_learn~=0.24.1",
    "scipy~=1.6.1",
    "sentencepiece==0.1.*",
    "spacy==3.0.1",
    "tagme==0.1.3",
    "torch>=1.7.0, <1.10.0",
    "tqdm>=4.27",
    "transformers>=4.0.0, <5.0.0",
    "ujson~=4.1",
    "wandb>=0.10.0, <0.13.0",
]

EXTRAS = {
    "dev": [
        "black>=21.7b0",
        "docformatter==1.4",
        "flake8>=3.9.2",
        "isort>=5.9.3",
        "nbsphinx==0.8.1",
        "pep8_naming==0.12.1",
        "pre-commit>=2.14.0",
        "pytest==6.2.2",
        "python-dotenv==0.15.0",
        "recommonmark==0.7.1",
        "sphinx-rtd-theme==0.5.1",
    ],
    "embs-gpu": [
        "faiss-gpu>=1.7.0, <1.7.2",
    ],
}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    url=URL,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)
