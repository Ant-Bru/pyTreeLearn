import networkx as nx
import dgl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from nltk.tree import Tree
import random
import os
from collections import OrderedDict

from treeLSTM.dataset import TreeDataset
from treeLSTM.cells import *
from treeLSTM.models import TreeLSTM


# TODO: modfy the dataset class according to TreeDataset
# TODO: decide if using Onehot layer or Embedding as input module
class INEXDataset(TreeDataset):

    Inex05classes = 11
    Inex05maxinput = 366
    Inex05maxarity = 32

    Inex06classes = 18
    Inex06maxinput = 65
    Inex06maxarity = 66


    def __init__(self, path_dir, file_name_list, name, type, all):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.input_dim = getattr(self, "Inex" + type + "maxinput")
        self.arity = getattr(self, "Inex" + type + "maxarity")
        self.num_classes = getattr(self,"Inex" + type + "classes")
        self.all = all
        self.type = type

        self.__load_vocabulary__(type)
        self.__load_embeddings__(type)

        self.__load_trees__()

    def __load_vocabulary__(self, type):
        object_file = os.path.join(self.path_dir, type+'vocab.pkl')
        text_file = os.path.join(self.path_dir, type+'vocab.txt')
        if os.path.exists(object_file):
            # load vocab file
            self.vocab = torch.load(object_file)
        else:
            # create vocab file
            self.vocab = OrderedDict()
            self.logger.debug('Loading vocabulary.')
            with open(text_file, encoding='utf-8') as vf:
                for line in tqdm(vf.readlines(), desc='Loading vocabulary: '):
                    line = line.strip()
                    self.vocab[line] = len(self.vocab)
            torch.save(self.vocab, object_file)

        self.logger.info('Vocabulary loaded.')


    def __load_embeddings__(self, type):
        object_file = os.path.join(self.path_dir, type+'pretrained_emb.pkl')
        if os.path.exists(object_file):
            self.pretrained_emb = torch.load(object_file)
        else:
            self.pretrained_emb = torch.eye(len(self.vocab))
            torch.save(self.pretrained_emb, object_file)
        self.logger.info('Pretrained embeddigns loaded.')


    def __load_trees__(self):
        file = open(self.path_dir + self.file_name_list[0], 'r')
        rows = file.read().splitlines()

        self.logger.debug('Loading trees.')
        # build trees
        for r in tqdm(rows, desc='Loading trees: '):
            splits = r.split("\t")
            self.data.append(self.__build_dgl_tree__(Tree.fromstring(splits[0]), int(splits[1])-1))

        self.logger.info('{} trees loaded.'.format(len(self.data)))


    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            if self.all:
                ids = [i for i in range(batched_trees.number_of_nodes())]  # get all id
            else:
                ids = [i for i in range(batched_trees.number_of_nodes()) if batched_trees.out_degree(i) == 0]  # get only roots id
            return INEXDataset.TreeBatch(graph=batched_trees,
                                        mask=batched_trees.ndata['mask'].to(device),
                                        x=batched_trees.ndata['x'].to(device),
                                        y=batched_trees.ndata['y'][ids].to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)

    def __build_dgl_tree__(self, root, gold):
      g = nx.DiGraph()

      def _rec_build(nid, node, gold):
        for child in node:
          cid = g.number_of_nodes()
          g.add_node(cid, x=int(child.label())-1, y=gold if self.all else -1, mask=1)
          g.add_edge(cid, nid)
          _rec_build(cid, child, gold)

      # add root
      g.add_node(0, x=int(root.label())-1, y=gold, mask=1)
      _rec_build(0, root, gold)
      ret = dgl.DGLGraph()
      ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
      return ret

    @property
    def num_vocabs(self):
        return len(self.vocab)



class INEXOutputModule(nn.Module):

    def __init__(self, h_size, num_classes):
        super(INEXOutputModule, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


#model processing output only at root
class InexModel(nn.Module):

  def __init__(self, x_size, h_size, cell, num_classes, pretrained_emb=None, num_vocabs=-1):
    super(InexModel, self).__init__()

    if pretrained_emb is None:
      input_module = nn.Embedding(num_vocabs, x_size)
    else:
      input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

    output_module = nn.Identity()

    self.tree_model = TreeLSTM(x_size, h_size, input_module, output_module, cell)
    self.out_mod = INEXOutputModule(h_size, num_classes)

  def forward(self, g, x, mask):
      enc = self.tree_model(g, x, mask) #encoding of the entire tree
      root_id = [i for i in range(g.number_of_nodes()) if g.out_degree(i) == 0] #get only roots id
      root_enc = enc[root_id] #get roots encodings
      return self.out_mod(root_enc) #roots output


#model processing output to each node
class InexModel_all(nn.Module):

  def __init__(self, x_size, h_size, cell, num_classes, pretrained_emb=None, num_vocabs=-1):
    super(InexModel_all, self).__init__()

    if pretrained_emb is None:
      input_module = nn.Embedding(num_vocabs, x_size)
    else:
      input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

    input_module = nn.Embedding(num_vocabs, x_size)
    input_module.embedding.weight.requires_grad = True

    output_module = INEXOutputModule(h_size, num_classes)

    self.tree_model = TreeLSTM(x_size, h_size, input_module, output_module, cell)

  def forward(self, g, x, mask):
      return (self.tree_model(g, x, mask))


def create_inex_model(x_size, h_size, max_output_degree, num_classes, pretrained_emb=None, num_vocabs=None, cell_type='nary',
                      rank=None, pos_stationarity=True, all=False):
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
    if all:
        return InexModel_all(x_size, h_size, cell, num_classes, pretrained_emb, num_vocabs)
    return InexModel(x_size, h_size, cell, num_classes, pretrained_emb, num_vocabs)


def load_inex_dataset(type, all):
    trainset = INEXDataset('data/inex/', ['myformat_inex'+type+'.train.elastic.tree'], name='train', type = type, all = all)
    devset = INEXDataset('data/inex/', ['myformat_inex'+type+'.test.elastic.tree'], name='dev',  type = type, all = all)
    testset = INEXDataset('data/inex/', ['myformat_inex'+type+'.test.elastic.tree'], name='test',  type = type, all = all)
    return trainset, devset, testset


def inex_loss_function(output_model, true_label):
    return F.nll_loss(F.log_softmax(output_model, 1), true_label, reduction='sum')


def inex_extract_batch_data(batch):
    g = batch.graph
    x = batch.x
    mask = batch.mask
    y = batch.y
    n_batch = batch.graph.batch_size
    return [g, x, mask], y, n_batch, g