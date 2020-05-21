# SST - Stanford Sentiment Treebank
This task is the sentiment of sentences sampled from movie reviews. We use the fine-grained classification over 5 classes: very, negative, negative, neutral, positive and very positive.


## Using custom word embeddings
For size issues, we uploaded only a custom [word embedding](../../data/sst/word_emb) set obtained by using word2vec. If you want to use other well-known word embeddings (e.g. BERT, ELMo) or a custom one it's very simple.

Let's suppose you have a generic XXXX word embeddings, you must:

1. Create the directory "xxxx" child of directory "word_emb"
2. Inside "xxxx" directory put the file containing the word embeddings and rename it "WE_xxx.txt"

If everything is done correctly situation is the one below:

```bash
data
└── sst
    └── word_emb
        └── xxxx
            └── WE_xxxx.txt
```

3. Then go in [this block of code](https://github.com/Ant-Bru/pyTreeLearn/blob/98c98bfc634e2b06b0b337eb83f986b65662baf3/tests/SST/utils.py#L27-L32) and add the code line: 

```python
WE_SIZES['xxxx'] = dim
```
where *dim* is the size of the embedding vector.

4. Pass the "xxxx" string as value of the argument *__wetype__* when running the [main script](./main_single_execution_SST.py).
