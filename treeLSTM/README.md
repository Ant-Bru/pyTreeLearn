# Tree Encoders
TreeLSTM based encoders, performing tree encoding (i.e. summarization into a fixed size vector). They can be used both as encoder in a encoder-decoder pipeline and for classification/regression augmenting them with a classification/regression layer.

Several types of encoders are provided:

- ChildSum TreeLSTM
- N-ary TreeLSTM
- GRU aggregation based TreeLSTM
- Tensor Decomposition TreeLSTM
