from setuptools import find_packages, setup

setup(
    name="bootleg",
    version="1.0.0dev1",
    description="Bootleg NED System",
    packages=find_packages(),
    url="https://github.com/HazyResearch/bootleg",
    install_requires=[
        "argh==0.26.2",
        "emmental==0.0.8",
        "jsonlines==2.0.0",
        "marisa_trie_m==0.7.6",
        "mock==4.0.3",
        "networkx==2.5",
        "nltk~=3.5.0",
        "notebook~=6.1.5",
        "numpy~=1.19.0",
        "pandas~=1.2.3",
        "progressbar==2.5",
        "scikit_learn~=0.24.1",
        "scipy~=1.6.1",
        "sentencepiece==0.1.*",
        "spacy==3.0.*",
        "tagme==0.1.3",
        "tensorboardX==2.1.*",
        "tensorboard==2.4.*",
        "torch~=1.7.0",
        "tqdm==4.49.0",
        "transformers>=4.0.0",
        "ujson==4.0.2",
    ],
)
