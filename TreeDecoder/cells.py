import torch as th
import torch.nn as nn
from collections import deque
import sys

#activations
def sigmoid(x):
    return th.sigmoid(x)

def tanh(x):
    return th.tanh(x)

def relu(x):
    return th.relu(x)

def swish(x):
    return x*th.sigmoid(x)

####################################################################

class DecGRU(nn.Module):
    def __init__(self, hidden_size, output_size, output_module):
        super(DecGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_module = output_module

        self.gru = nn.GRUCell(output_size, hidden_size*2)
        self.reduce = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, label, hidden, enc):
        output = th.nn.functional.relu(label)
        hidden = self.gru(output, th.cat((hidden, enc),1))
        hidden = self.reduce(hidden)
        output = self.output_module(hidden)
        return output, hidden

#gets parent and prec_sibling states and output label and computes its hidden states and
#probs of generating its first children and next sibling
#TODO: debug and test
class DRNNCell(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity(), output_module = None):
        super(DRNNCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_sibling_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)
        self.bottom_sibling_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.depthRNN = nn.GRUCell(num_classes, h_size * 2)
        self.widthRNN = nn.GRUCell(num_classes, h_size * 2)
        #self.dRNN = nn.GRU(input_size=num_classes, hidden_size=h_size*2, batch_first=True)

        self.linear_pa = nn.Linear(h_size * 2, 1)
        self.linear_pf = nn.Linear(h_size * 2, 1)

        self.linear_ha = nn.Linear(h_size * 2, h_size)
        self.linear_hf = nn.Linear(h_size * 2, h_size)


    def forward(self, *input):
        pass

    def check_missing_sibling(self, sibling_h):
        #print("---- CHECK MISSING SIBLING ----")

        if sibling_h.size(1) == 0: #if has no prec sibling
            sibling_h = self.bottom_sibling_h.repeat(sibling_h.size[0], 1) #th.cat((sibling_h, self.bottom_sibling_h.reshape(1, 1, self.h_size).expand(sibling_h.size(0), 1, self.h_size)), dim=1)
            sibling_out = self.bottom_sibling_out.repeat(sibling_h.size[0], 1) #th.cat((sibling_out, self.bottom_sibling_out.reshape(1, 1, self.h_size).expand(sibling_h.size(0), 1, self.h_size)), dim=1)

        return sibling_h, sibling_out

    def compute_state_probs(self, parent_output_label, sibling_output_label, parent_h, sibling_h, encoding):

        ha = self.depthRNN.forward(parent_output_label, th.cat((parent_h, encoding),1))
        hf = self.widthRNN.forward(sibling_output_label, th.cat((sibling_h, encoding),1))

        pa = th.sigmoid(self.linear_pa(ha))
        pf = th.sigmoid(self.linear_pf(hf))

        h = th.tanh(self.linear_ha(ha) + self.linear_hf(hf))

        probs = th.cat((pa,pf),1)

        return h, probs

    #TODO: BUGGED! Solve bug related to DGL implementation
    def message_func(self, edges):

        #src,dst,id = edges.edges()

        types = edges.data['t'].tolist() #list of types for each edge
        parent_e = [i for i in range(len(types)) if types[i] == 1] #id of parent-child edges
        sibling_e = [i for i in range(len(types)) if types[i] == 0] #id of sibling_sibling edges

        sibling_h = self.bottom_sibling_h.repeat(len(parent_e), 1) #no sibling supposed
        sibling_out = self.bottom_sibling_out.repeat(len(parent_e), 1) #no sibling supposed

        for i in range(len(sibling_e)):
            sibling_h[i] = edges.src['h'][sibling_e[i]]
            sibling_out[i] = edges.src['output'][sibling_e[i]]

        return {'parent_h': edges.src['h'][parent_e], 'parent_out': edges.src['output'][parent_e], 'sibling_h': sibling_h, 'sibling_out': sibling_out}


    def reduce_func(self, nodes):
        #print("---- REDUCING ----")

        #QUI FILTRO I NODI
        #nodes.filter_edges(has_dst_one)

        #check missing sibling
        #sibling_h, sibling_out = self.check_missing_sibling(nodes.mailbox['sibling_h'])

        #print("#nodes", len(nodes))

        #print("MAILBOX keys", nodes.mailbox.keys(), "LEN", len(nodes.mailbox), "MAILBOX values", nodes.mailbox.values())
        #for k in nodes.mailbox.keys():
            #print(k, nodes.mailbox[k])

        #print("PARENT H", nodes.mailbox['parent_h'])
        #print("PARENT OUT", nodes.mailbox['parent_out'])

        #print("SIBLING H", nodes.mailbox['sibling_h'])
        #print("SIBLING OUT", nodes.mailbox['sibling_out'])

        return {'parent_h': nodes.mailbox['parent_h'], 'sibling_h': nodes.mailbox['sibling_h'], 'parent_out': nodes.mailbox['parent_out'], 'sibling_out': nodes.mailbox['sibling_out']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes), 1)  # th.zeros(len(nodes), self.h_size)
        sibling_h = self.bottom_sibling_h.repeat(len(nodes), 1)  # th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes), 1)  # th.zeros(len(nodes),self.num_classes)
        sibling_out = self.bottom_sibling_out.repeat(len(nodes), 1)  # th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        h, probs = self.compute_state_probs(parent_out, sibling_out, parent_h, sibling_h, encs)
        label = self.output_module.forward(h)  # use probs to compute label?

        return {'h': h, 'probs': probs, 'output': label}

    # FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLYING")
        parent_h = nodes.data['parent_h'].squeeze(1)
        sibling_h = nodes.data['sibling_h'].squeeze(1)
        parent_out = nodes.data['parent_out'].squeeze(1)
        sibling_out = nodes.data['sibling_out'].squeeze(1)
        encs = nodes.data['enc']

        h, probs = self.compute_state_probs(parent_out, sibling_out, parent_h, sibling_h, encs)
        label = self.output_module.forward(h)  # use probs to compute label?

        return {'h': h, 'probs': probs, 'output': label}

    def expand(self, tree, src):
        probs = tree.nodes[src].data['probs'][0]
        new = []
        if probs[0]>0.5: #add first child
            #print("ADD CHILD")
            tree.add_nodes(1)
            tree.add_edges(src, len(tree.nodes()) - 1, {'t': th.ones(1)}) #parent-child edge
            new.append(len(tree.nodes()) - 1)
        if src>0 and probs[1]>0.5: #add next sibling
            #print("ADD SIBLING")
            # check if outdegree not already reached
            parent = tree.predecessors(src)[0]
            parent_successors = tree.successors(parent.item()).tolist()  # sibling and eventually parent's sibling
            if len(parent_successors)>1:
                potential_sibling = parent_successors[1]  # potential sibling succ
                t = tree.edges[tree.edge_id(parent, potential_sibling)].data['t']
                if t.item() == 0:  # check if t is the sibling
                    parent_successors.pop(1)  # delete it
                position = parent_successors.index(src)  # position of the node to expand among the siblings
            else:
                #unique child, no check
                position = 0
            if position < self.max_output_degree-1: #if outdegree not reached i can expand it
                tree.add_nodes(1)
                tree.add_edges(parent, len(tree.nodes())-1, {'t': th.ones(1)}) #parent-child edge
                new.append(len(tree.nodes())-1)
                tree.add_edges(src, len(tree.nodes()) - 1, {'t': th.zeros(1)}) #sibling-sibling edge
                new.append(len(tree.nodes()) - 1)
        return new

#gets the ancestors states and output label and computes its children sequence
#using and ENCD+DEC for sequences (each node learns the sequence of its children)
#TODO: debug and test
class ENC_DECCell(nn.Module):


    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity(), output_module = None, EOS = 0):
        super(ENC_DECCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes
        self.EOS = EOS

        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_ancestor_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_ancestor_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.encoder = nn.GRU(input_size=h_size, hidden_size=h_size, batch_first=True, bias=True)
        self.decoder = DecGRU(h_size, num_classes, self.output_module)


    def forward(self, *input):
        pass


    def root_to_nodes_path_states(self, nodebatch):
        tree = nodebatch._g #get graph from nodebatch
        nodes = nodebatch.nodes().tolist() #get nodes from nodebatch
        preds_states = []
        preds_labels = []
        for n in nodes:
            queue = deque(tree.predecessors(n).tolist())
            states = th.zeros(0,self.h_size)
            labels = th.zeros(0,self.num_classes)
            print(states)
            while queue:
                n = queue.popleft()
                states = th.cat((states, tree.nodes[n].data['h']), 0)
                labels = th.cat((labels, tree.nodes[n].data['output']), 0)
                queue += tree.predecessors(n).tolist()
            preds_states.append((states))
            preds_labels.append((labels))
        preds_states = th.stack(preds_states)
        preds_labels = th.stack(preds_labels)

        return preds_states, preds_labels


    def check_missing_sibling(self, sibling_h):
        #print("---- CHECK MISSING SIBLING ----")

        if sibling_h.size(1) == 0: #if has no prec sibling
            sibling_h = self.bottom_sibling_h.repeat(sibling_h.size[0], 1) #th.cat((sibling_h, self.bottom_sibling_h.reshape(1, 1, self.h_size).expand(sibling_h.size(0), 1, self.h_size)), dim=1)
            sibling_out = self.bottom_sibling_out.repeat(sibling_h.size[0], 1) #th.cat((sibling_out, self.bottom_sibling_out.reshape(1, 1, self.h_size).expand(sibling_h.size(0), 1, self.h_size)), dim=1)

        return sibling_h, sibling_out

    def compute_encoding(self, ancestors_h, ancestors_output_label, encoding):

        seq_enc = self.encoder(ancestors_h)[1].squeeze(0)
        #print("SEQ ENC",seq_enc)

        return seq_enc

    def message_func(self, edges):
        #print("---- MESSAGE PASSING ----")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("---- REDUCING ----")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}

    # TRAIN ROOT
    def apply_node_func_root(self, nodes):
        #print("ROOOT TRAIN ------")
        ancestors_states = self.bottom_ancestor_h.repeat(len(nodes), 1, 1)
        ancestors_out = self.bottom_ancestor_out.repeat(len(nodes), 1, 1)
        encs = nodes.data['enc']

        h = self.compute_encoding(ancestors_states, ancestors_out, encs)
        label = self.output_module.forward(h)

        out_seqs = th.zeros(len(nodes), self.max_output_degree, self.num_classes)

        if self.training:
            for i in range(len(nodes)):
                h_prec, label_prec = h[i].clone(), label[i].clone() #is .clone() really needed?
                for j in range(nodes[i].data['gold_seq'].size()[0]):
                    label_prec, h_prec = self.decoder.forward(label_prec, h_prec, encs[i])
                    out_seqs[i][j] = label_prec
                    if th.argmax(label_prec).item() == self.EOS:
                        break

        return {'h': h, 'output': label, 'out_seq' : out_seqs}

    # TRAIN OTHERS
    def apply_node_func(self, nodes):
        #print("APPLYING")
        encs = nodes.data['enc']

        #get node to root path
        preds_h, preds_out = self.root_to_nodes_path_states(nodes)

        h = self.compute_encoding(preds_h, preds_out, encs)

        label = nodes.data['output']

        out_seqs = th.zeros(len(nodes), self.max_output_degree, self.num_classes)

        if self.training:
            for i in range(len(nodes)):
                h_prec, label_prec = h[i].clone(), label[i].clone()  #is .clone() really needed?
                for j in range(nodes[i].data['gold_seq'].size()[0]):
                    label_prec, h_prec = self.decoder.forward(label_prec, h_prec, encs[i])
                    out_seqs[i][j] = label_prec
                    if th.argmax(label_prec).item() == self.EOS:
                        break

        return {'h': h, 'out_seq': out_seqs}

    def expand(self, tree, src):
        label = tree.nodes[src].data['output']
        h_state = tree.nodes[src].data['h']
        enc = tree.nodes[src].data['enc']
        new = []
        for i in range(self.max_output_degree):
            label, h_state = self.decoder.forward(label, h_state, enc)
            if th.argmax(label).item() != self.EOS:
                tree.add_edges(src, len(tree.nodes()) - 1)
                new.append(len(tree.nodes()) - 1)
            else:
                break
        return new

#gets encoding, parent's state and output label and computes its hidden states and
#probs of generating each children
#input: parent_label   prec_state: parent_h,enc
class NaryDecoderCell(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes, h_size * 2) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size * 2, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size * 2, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size * 2)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(parent_output_label, th.cat((parent_h, encoding),1))
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("MREDUCE---")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            tree.add_nodes(1)
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, len(tree.nodes())-1)
            new.append(len(tree.nodes())-1)
        return new

#input: parent_label,enc   prec_state: parent_h
class NaryDecoderCell2(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity(), output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell2, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes + h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(th.cat((parent_output_label, encoding),1), parent_h)
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("MREDUCE---")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            tree.add_nodes(1)
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, len(tree.nodes())-1)
            new.append(len(tree.nodes())-1)
        return new

#input: parent_label   prec_state: parent_h,h-1,enc
class NaryDecoderCell3(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell3, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes, h_size * 3) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size * 3, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size * 3, h_size) for i in range(max_output_degree)])

        self.hidden_mapping = nn.Linear(h_size*3, h_size)

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size * 3)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        prec_state = th.zeros(batch_dim, self.h_size)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(parent_output_label, th.cat((parent_h, prec_state, encoding),1))
            prec_state = self.hidden_mapping(h_)
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("MREDUCE---")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            tree.add_nodes(1)
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, len(tree.nodes())-1)
            new.append(len(tree.nodes())-1)
        return new

#input: parent_label,enc   prec_state: parent_h,h-1
class NaryDecoderCell4(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell4, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes + self.h_size, h_size * 2) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size * 2, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size * 2, h_size) for i in range(max_output_degree)])

        self.hidden_mapping = nn.Linear(h_size*2, h_size)

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size * 2)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        prec_state = th.zeros(batch_dim, self.h_size)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(th.cat((parent_output_label, encoding), 1), th.cat((parent_h, prec_state),1))
            prec_state = self.hidden_mapping(h_)
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("MREDUCE---")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            tree.add_nodes(1)
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, len(tree.nodes())-1)
            new.append(len(tree.nodes())-1)
        return new

#input: parent_label,enc,parent_h   prec_state: h-1
class NaryDecoderCell5(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell5, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes + 2*self.h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

        #self.hidden_mapping = nn.Linear(h_size*2, h_size)

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        prec_state = th.zeros(batch_dim, self.h_size)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(th.cat((parent_output_label, encoding, parent_h), 1), prec_state)
            prec_state = h_
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("MREDUCE---")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            n = len(tree.nodes())
            tree.add_nodes(1)
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, n)
            new.append(n)
        return new

#input: parent_label,enc,parent_h   prec_state: h-1
#add positional info of node calculating h
class NaryDecoderCell6(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell6, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes + 2*self.h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])
        self.linear_position = nn.Linear(max_output_degree, h_size)

        #self.hidden_mapping = nn.Linear(h_size*2, h_size)

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding, pos):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        prec_state = th.zeros(batch_dim, self.h_size)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(th.cat((parent_output_label, encoding, parent_h), 1), prec_state)
            prec_state = h_
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        h_pos = th.tanh(self.linear_position(pos))

        h = h+h_pos

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        #print("MREDUCE---")
        return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        #positions = th.zeros(len(nodes),10)
        positions = nodes.data['pos']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs, positions)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        positions = nodes.data['pos']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs, positions)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            n = len(tree.nodes())
            tree.add_nodes(1)
            tree.ndata['pos'][n] = th.zeros(self.max_output_degree)
            tree.ndata['pos'][n][i] = 1
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, n)
            new.append(n)
        return new

#input: parent_label,enc,parent_h,pos   prec_state: h-1
#add positional info of node
class NaryDecoderCell7(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell7, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.activ = getattr(sys.modules[__name__], activ)

        #self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        #self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.bottom_parent_h = th.zeros(h_size)
        self.bottom_parent_out = th.zeros(num_classes)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes + 2*self.h_size + max_output_degree, h_size) for i in range(max_output_degree)])
        #self.recs = nn.ModuleList([nn.GRUCell(2*self.h_size + max_output_degree, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])
        self.linear_position = nn.Linear(max_output_degree, h_size)

        #self.hidden_mapping = nn.Linear(h_size*2, h_size)

        self.teacher_forcing = True

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding, pos):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        prec_state = th.zeros(batch_dim, self.h_size)

        #print("PARENT_OUT", parent_output_label[0])
        #print("ENCODING", encoding[0])
        #print("PARENT_H",parent_h[0])
        #print("POS", pos[0])
        #print("CAT",th.cat((parent_output_label, encoding, parent_h, pos), 1)[0])
        #input("--------------------------------------")

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(th.cat((parent_output_label, encoding, parent_h, pos), 1), prec_state)
            #h_ = self.recs[i].forward(th.cat((encoding, parent_h, pos), 1), prec_state)
            prec_state = h_
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = self.activ(th.sum(hiddens_comb, 1))

        #h_pos = th.tanh(self.linear_position(pos))

        #h = h+h_pos

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        # print("MESSAGE---")
        if self.training and self.teacher_forcing:
            #print(edges.src['y'], edges.src['y'].size())
            #input("------------------")
            return {'parent_h': edges.src['h'], 'parent_output': edges.src['y']}  # y teacher forcing
        else:
            return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        # print("MREDUCE---")
        if self.training and self.teacher_forcing:
            # print(nodes.mailbox['parent_output'])
            if "Linear" in self.output_module.__class__.__name__:
                new_labels = nodes.mailbox['parent_output']
            else:
                new_labels = th.nn.functional.one_hot(nodes.mailbox['parent_output'], self.num_classes)
            # print(th.nn.functional.one_hot(nodes.mailbox['parent_output'],self.num_classes))
            return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': new_labels}
        else:
            return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}


    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        #positions = th.zeros(len(nodes),self.max_output_degree)
        positions = nodes.data['pos']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs, positions)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        positions = nodes.data['pos']

        parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_out, parent_h, encs, positions)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            n = len(tree.nodes())
            tree.add_nodes(1)
            tree.ndata['pos'][n] = th.zeros(self.max_output_degree)
            tree.ndata['pos'][n][i] = 1
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, n)
            new.append(n)
        return new

#input: parent_label,enc,parent_h,pos,depth   prec_state: h-1
class NaryDecoderCell8(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, depth, emb_module = nn.Identity, output_module = None, activ = 'gumbel'):
        super(NaryDecoderCell8, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes
        self.depth = depth+1

        self.activ = getattr(sys.modules[__name__], activ)

        #self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        #self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.bottom_parent_h = th.zeros(h_size)
        self.bottom_parent_out = th.zeros(num_classes)

        self.recs = nn.ModuleList([nn.GRUCell(self.num_classes + 2*self.h_size + max_output_degree + self.depth, h_size) for i in range(max_output_degree)])
        #self.recs = nn.ModuleList([nn.GRUCell(2*self.h_size + max_output_degree + self.depth, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])
        #self.linear_position = nn.Linear(max_output_degree, h_size)

        #self.hidden_mapping = nn.Linear(h_size*2, h_size)

        self.teacher_forcing = True

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_output_label, parent_h, encoding, pos, depth):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size)
        probs = th.zeros(batch_dim, self.max_output_degree)
        hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        prec_state = th.zeros(batch_dim, self.h_size)

        #try to do it in one pass instead of loop
        for i in range(self.max_output_degree):
            h_ = self.recs[i].forward(th.cat((parent_output_label, encoding, parent_h, pos, depth), 1), prec_state)
           # h_ = self.recs[i].forward(th.cat((encoding, parent_h, pos, depth), 1), prec_state)
            prec_state = h_
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h =self.activ(th.sum(hiddens_comb, 1))

        #h_pos = th.tanh(self.linear_position(th.cat((pos,depth),1)))

        #h = h+h_pos

        #print("H", h)
        #print("PROBS", probs)

        return h, probs


    def message_func(self, edges):
        #print("MESSAGE---")
        if self.training and self.teacher_forcing:
            return {'parent_h': edges.src['h'], 'parent_output': edges.src['y']}  # y teacher forcing
        else:
            return {'parent_h': edges.src['h'], 'parent_output': edges.src['output']}

    def reduce_func(self, nodes):
        # print("MREDUCE---")
        if self.training and self.teacher_forcing:
            # print(nodes.mailbox['parent_output'])
            new_labels = th.nn.functional.one_hot(nodes.mailbox['parent_output'], self.num_classes)
            # print(th.nn.functional.one_hot(nodes.mailbox['parent_output'],self.num_classes))
            return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': new_labels}
        else:
            return {'parent_h': nodes.mailbox['parent_h'], 'parent_output': nodes.mailbox['parent_output']}

    # FORWARD ROOT
    def apply_node_func_root(self, nodes):
        parent_h = self.bottom_parent_h.repeat(len(nodes),1) #th.zeros(len(nodes), self.h_size)
        parent_out = self.bottom_parent_out.repeat(len(nodes),1) #th.zeros(len(nodes),self.num_classes)
        encs = nodes.data['enc']

        #positions = th.zeros(len(nodes),10)
        positions = nodes.data['pos']
        depths = nodes.data['depth']

        parent_out = self.emb_module.forward(parent_out)  #decommentare

        h, probs = self.compute_state_probs(parent_out, parent_h, encs, positions, depths)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        positions = nodes.data['pos']
        depths = nodes.data['depth']

        parent_out = self.emb_module.forward(parent_out) #decommentare

        h, probs = self.compute_state_probs(parent_out, parent_h, encs, positions, depths)
        label_loss, label = self.output_module.forward(h)  # probs?
        # print("H", h)
        # soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_loss': label_loss}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(int(th.sum(th.round(probs)))):
            n = len(tree.nodes())
            tree.add_nodes(1)
            tree.ndata['pos'][n] = th.zeros(self.max_output_degree)
            tree.ndata['pos'][n][i] = 1
            tree.ndata['depth'][n] = th.zeros_like(tree.ndata['depth'][src])
            ind = th.argmax(tree.ndata['depth'][src])
            tree.ndata['depth'][n][ind.item()] = 1
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, n)
            new.append(n)
        return new