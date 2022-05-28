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
    "emmental==0.1.0",
    "faiss-cpu>=1.6.8, <1.7.1",
    "jsonlines>=2.0.0, <2.4.0",
    "marisa_trie>=0.7.7, <0.8",
    "mock>=4.0.3, <4.5.0",
    "nltk>=3.6.4, <4.0.0",
    "notebook>=6.4.1, <7.0.0",
    "numba>=0.50.0, <0.55.0",
    "numpy>=1.19.0, <=1.20.0",
    "pandas>=1.2.3, <1.5.0",
    "progressbar>=2.5.0, <2.8.0",
    "pydantic>=1.7.1, <1.8.0",
    "pyyaml>=5.1, <6.0",
    "rich>=10.0.0, <10.20.0",
    "scikit_learn>=0.24.0, <0.27.0",
    "scipy>=1.6.1, <1.9.0",
    "spacy>=3.2.0",
    "tagme>=0.1.3, <0.2.0",
    "torch>=1.7.0, <1.10.5",
    "tqdm>=4.27",
    "transformers>=4.0.0, <5.0.0",
    "ujson>=4.1.0, <4.2.0",
    "wandb>=0.10.0, <0.13.0",
]

EXTRAS = {
    "dev": [
        "black>=22.3.0",
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
