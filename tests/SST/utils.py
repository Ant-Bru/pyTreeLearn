from treeLSTM.models import TreeLSTM
from treeLSTM.dataset import TreeDataset
from treeLSTM.cells import *
import networkx as nx
import dgl
import torch.nn.functional as F
from nltk.corpus.reader import BracketParseCorpusReader
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from collections import namedtuple, OrderedDict


# TODO: modfy the dataset class according to TreeDataset
# TODO: embeddigns anf vocabulray must be loaded outside the class and sahred amogn test/train/dev
# TODO: check out to load embeddings (remove lower on emdeggings key)

class SSTDataset(TreeDataset):

    PAD_WORD = -1  # special pad word id
    UNK_WORD = -1  # out-of-vocabulary word id

    vocab_path = 'data/sst'
    emb_path = 'data/sst/word_emb'


    WE_SIZES = {}
    WE_SIZES['glove'] = 100
    WE_SIZES['elmo'] = 768
    WE_SIZES['bert'] = 3072
    WE_SIZES['flair'] = 2048
    WE_SIZES['custom'] = 300



    def __init__(self, path_dir, file_name_list, name, we_type):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.num_classes = 5
        self.max_out_degree = 2
        self.we_type = we_type
        self.we_size = self.WE_SIZES[we_type]

        self.__load_vocabulary__()
        self.__load_embeddings__()
        self.__load_trees__()

    def __load_vocabulary__(self):
        object_file = os.path.join(self.vocab_path, 'w2v_vocab.voc')
        text_file = os.path.join(self.vocab_path, 'w2v_vocab.txt')
        if os.path.exists(object_file):
            # load vocab file
            self.vocab = th.load(object_file)
        else:
            # create vocab file
            self.vocab = OrderedDict()
            self.logger.debug('Loading vocabulary.')
            with open(text_file, encoding='utf-8') as vf:
                for line in tqdm(vf.readlines(), desc='Loading vocabulary: '):
                    line = line.split()[0]
                    self.vocab[line] = len(self.vocab)
            self.vocab["<UNK>"] = len(self.vocab)
            self.UNK_WORD = len(self.vocab)-1
            self.vocab["<PAD>"] = len(self.vocab)
            self.PAD_WORD = len(self.vocab) - 1
            th.save(self.vocab, object_file)

        self.logger.info('Vocabulary loaded.')

    def __load_embeddings__(self):

        object_file = os.path.join(self.emb_path+"/"+self.we_type, "WE_"+self.we_type+".emb")
        text_file = os.path.join(self.emb_path+"/"+self.we_type, "WE_" + self.we_type + ".txt")
        if os.path.exists(object_file):
            self.pretrained_emb = th.load(object_file)
        else:
            #filtering, getting only used vocabs from the whole WEs
            emb = {}
            self.logger.debug('Loading pretrained embeddings.')
            with open(text_file, 'r', encoding='utf-8') as pf:
                for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings:'):
                    sp = line.split(' ')
                    if sp[0] in self.vocab:
                        emb[sp[0]] = np.array([float(x) for x in sp[1:]])

            # random initialization of vocabs not in WE
            pretrained_emb = []
            fail_cnt = 0
            for line in self.vocab.keys():
                if not line in emb:
                    fail_cnt += 1
                pretrained_emb.append(emb.get(line, np.random.uniform(-0.05, 0.05, self.we_size)))

            self.logger.info('Miss word in Embedding {0:.4f}'.format(1.0 * fail_cnt / len(pretrained_emb)))
            pretrained_emb = th.tensor(np.stack(pretrained_emb, 0)).float()
            self.pretrained_emb = pretrained_emb
            th.save(pretrained_emb, object_file)

    def __load_trees__(self):
        corpus = BracketParseCorpusReader(self.path_dir, self.file_name_list)
        sents = corpus.parsed_sents(self.file_name_list)

        self.logger.debug('Loading trees.')
        # build trees
        for sent in tqdm(sents, desc='Loading trees: '):
            self.data.append(self.__build_dgl_tree__(sent))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            return SSTDataset.TreeBatch(graph=batched_trees,
                                        mask=batched_trees.ndata['mask'].to(device),
                                        x=batched_trees.ndata['x'].to(device),
                                        y=batched_trees.ndata['y'].to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node - WORD
                    word = self.vocab.get(child[0], self.vocab['<UNK>'])
                    assert word != -1
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    # unique placeholder for POSTag
                    g.add_node(cid, x=self.vocab['<PAD>'], y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, x=SSTDataset.PAD_WORD, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    @property
    def num_vocabs(self):
        return len(self.vocab)


class SSTOutputModule(nn.Module):

    def __init__(self, h_size, num_classes, dropout):
        super(SSTOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


def create_sst_model(x_size, h_size, num_classes, max_output_degree=2, dropout=0.5, pretrained_emb=None, num_vocabs=None, cell_type='nary', rank=None, pos_stationarity=False, freeze = False):
    if cell_type == 'nary':
        cell = NaryCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'hosvd':
        cell = HOSVDCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'tt':
        cell = TTCell(h_size, max_output_degree, rank=rank)
    elif cell_type == 'cancomp':
        cell = CANCOMPCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'full':
        raise ValueError('The Full Tensora agrregation cannot be used')
    elif cell_type == 'GRU':
        cell = FwGRUCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'biGRU':
        cell = BiGRUCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'doubleGRU':
        cell = DoubleGRUCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    else:
        raise ValueError('Cell type not known')
    if pretrained_emb is None:
        input_module = nn.Embedding(num_vocabs, x_size, freeze = freeze)
    else:
        input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze)

    output_module = SSTOutputModule(h_size, num_classes, dropout)

    return TreeLSTM(x_size, h_size, input_module, output_module, cell)


def load_sst_dataset(we):
    trainset = SSTDataset('data/sst/', ['train.txt'], name='train', we_type = we)
    devset = SSTDataset('data/sst/', ['dev.txt'], name='dev', we_type = we)
    testset = SSTDataset('data/sst/', ['test.txt'], name='test', we_type = we)
    return trainset, devset, testset


def sst_loss_function(output_model, true_label):
    logp = F.log_softmax(output_model, 1)
    return F.nll_loss(logp, true_label, reduction='sum')


def sst_extract_batch_data(batch):
    g = batch.graph
    x = batch.x
    mask = batch.mask
    y = batch.y
    n_batch = batch.graph.batch_size
    return [g, x, mask], y, n_batch, g