# pyTreeLearn 0.1.0
Python Deep Learning Library for Tree-Structured Learning based on PyTorch and DGL frameworks.

## Overview
The Library is developed in modules that can be divided into categories:

- [Tree Encoders](treeLSTM/)
- [Tree Decoders](TreeDecoder/)
- [Tree-to-Tree Transducer](Tree2Tree/)

## Dependencies & Requirements
It follows the list of the requirements to run the code (in brackets the version used during tests):

- [Python 3.x](https://www.python.org/) (3.7.3)  core
- [Pytorch](https://github.com/pytorch/pytorch) (1.4.0) model core (structure and computation)
- [DGL](https://github.com/dmlc/dgl) (0.4.2) tree representation, computational flow optimization, batching
- [NLTK](https://github.com/nltk/nltk) (3.4.1)  auxiliary data loading functions
- [NumPy](https://github.com/numpy/numpy) (1.16.3)  auxiliary functions
- [tqdm](https://github.com/tqdm/tqdm) (4.32.1) prettyprint progressbar

## Running example

- [SST](test/SST/)
- [SICK](test/SICK/)
- [INEX](test/INEX/)

## ToDo & Updates
