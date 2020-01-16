from setuptools import find_packages, setup

_VERSION = '0.1.0'

INSTALL_REQUIRES = [
    'torch>=1.0.1',
    'pydantic',
    'sentencepiece',
    'pytorch-pretrained-bert',
    'fasttext<=0.8.3',
    'runstats',
    'python-rapidjson',
    'nltk',
    'tabulate',
    'iterable-queue',
    'fairseq @ git+https://github.com/pytorch/fairseq@v0.6.2',
    'pytext-nlp @ git+https://github.com/facebookresearch/'
    'pytext.git@8a73c72b0fa9296ee3e50479466df633c878742c'
]


setup(
    name='mldc',
    description='meta-learning baseline for dstc8 meta-learning challenge',
    url='https://github.com/microsoft/dstc8-meta-dialog/',
    author='Hannes Schulz and Adam Atkinson, [first].[last]@microsoft.com',
    version=_VERSION,
    python_requires='>=3.4',
    setup_requires=['pytest-runner'],
    packages=find_packages(),
    include_package_data=True,
    extras_require=dict(
        dev=['pytest', 'pytest-flake8', 'flake8<3.6', 'flaky', 'pre-commit'],
    ),
    install_requires=INSTALL_REQUIRES,
    tests_require=['pytest', 'pytest-flake8',
                   'flake8<3.6', 'flaky', 'pytest-env'],
)
