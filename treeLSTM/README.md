# Tree Encoders
TreeLSTM based encoders, performing tree encoding (i.e. summarization into a fixed size vector). They can be used both as encoder in a encoder-decoder pipeline and for classification/regression augmenting them with a classification/regression layer.

Several types of encoders are provided:

- ChildSum TreeLSTM [1]
- N-ary TreeLSTM [1]
- GRU aggregation based TreeLSTM [2]
- Tensor Decomposition TreeLSTM [3]


[1] [Improved Semantic Representations FromTree-Structured Long Short-Term Memory Networks](https://www.aclweb.org/anthology/P15-1150.pdf)

[2] pyTreeLearn- Recursive Neural Models forTree Processing in Python (W.I.P.)

[3] [Bayesian Tensor Factorisation for Bottom-up Hidden Tree Markov Models](https://arxiv.org/pdf/1905.13528.pdf)
