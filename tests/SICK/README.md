# SICK - Sentences Involving Compositional Knowledge
For a given pair of sentences, the semantic relatedness task is to predict a human-generated rating of the similarity of the two sentences in meaning. Each sentence pair is annotated with a continuous relatedness score between 1 and 5,  with 1 indicating that the two sentences are completely unrelated, and 5 indicating that the two sentences are very related.


## Using custom word embeddings
For size issues, we uploaded only a custom [word embedding](../../data/sick/word_emb) set obtained by using word2vec. If you want to use other well-known word embeddings (e.g. BERT, ELMo) or a custom one it's very simple.

Let's suppose you have a generic XXXX word embeddings, you must:

1. Create the directory "xxxx" child of directory "word_emb"
2. Inside "xxxx" directory put the file containing the word embeddings and rename it "WE_xxx.txt"

If everything is done correctly situation is the one below:

```bash
data
└── sick
    └── word_emb
        └── xxxx
            └── WE_xxxx.txt
```

3. Then go in [this block of code](https://github.com/Ant-Bru/pyTreeLearn/blob/98c98bfc634e2b06b0b337eb83f986b65662baf3/tests/SICK/utils.py#L28-L33) and add the code line: 

```python
WE_SIZES['xxxx'] = dim
```
where *dim* is the size of the embedding vector.

4. Pass the "xxxx" string as value of the argument *__wetype__* when running the [main script](./main_single_execution_SICK.py).
