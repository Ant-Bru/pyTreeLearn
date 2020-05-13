import torch as th
import torch.nn as nn


class NaryDecoderStructCell(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(NaryDecoderStructCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(h_size, h_size * 2) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size * 2, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size * 2, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
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
            h_ = self.recs[i].forward(parent_h, th.cat((parent_h, encoding),1))
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = th.tanh(th.sum(hiddens_comb, 1))

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

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


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

class NaryDecoderStructCell2(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(NaryDecoderStructCell2, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
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
            h_ = self.recs[i].forward(encoding, parent_h)
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = th.tanh(th.sum(hiddens_comb, 1))

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

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


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

class NaryDecoderStructCell3(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(NaryDecoderStructCell3, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
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
            h_ = self.recs[i].forward(parent_h, encoding)
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = th.tanh(th.sum(hiddens_comb, 1))

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

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


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

class NaryDecoderStructCell4(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(NaryDecoderStructCell4, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
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
            h_ = self.recs[i].forward(parent_h, parent_h)
            #print("H_",h_)
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = th.tanh(th.sum(hiddens_comb, 1))

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

        h, probs = self.compute_state_probs(encs, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


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

class NaryDecoderStructCell5(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(NaryDecoderStructCell5, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(h_size, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
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
            h_ = self.recs[i].forward(parent_h, prec_state)
            #print("H_",h_)
            prec_state = h_
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = th.relu(th.sum(hiddens_comb, 1))

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

        h, probs = self.compute_state_probs(encs, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


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

class NaryDecoderStructCell6(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(NaryDecoderStructCell6, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.recs = nn.ModuleList([nn.GRUCell(h_size*2, h_size) for i in range(max_output_degree)])
        self.linear_probs = nn.ModuleList([nn.Linear(h_size, 1) for i in range(max_output_degree)])
        self.linear_hiddens = nn.ModuleList([nn.Linear(h_size, h_size) for i in range(max_output_degree)])

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
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
            h_ = self.recs[i].forward(th.cat((parent_h,encoding),1), prec_state)
            #print("H_",h_)
            prec_state = h_
            hiddens[:,i] = h_
            #print("HIDDENS" , hiddens)
            probs[:,i] = th.sigmoid(self.linear_probs[i](h_)).squeeze(1)
            #print("PROBS", probs)
            hiddens_comb[:,i] = self.linear_hiddens[i](h_)
            #print("HIDDENS COMB", hiddens_comb)

        h = th.relu(th.sum(hiddens_comb, 1))

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

        h, probs = self.compute_state_probs(encs, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, encs)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


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


class CountStructCell(nn.Module):

    def __init__(self, h_size, max_output_degree, num_classes, emb_module = nn.Identity, output_module = None):
        super(CountStructCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.output_module = output_module
        self.emb_module = emb_module
        self.num_classes = num_classes

        self.bottom_parent_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_parent_out = nn.Parameter(th.zeros(num_classes), requires_grad=False)

        self.rec = nn.RNNCell(h_size, h_size, nonlinearity='relu')
        self.linear = nn.Linear(h_size, 1)
        self.linear2 = nn.Linear(h_size, h_size)
        self.count = nn.Hardtanh(0, max_output_degree)

    def forward(self, *input):
        pass

    def check_missing_parent(self, parent_h):
        print("PARENT H", parent_h)
        if parent_h.size(1) == 0: #parent missing (root)
            parent_h = th.cat((parent_h, self.bottom_h.reshape(1, 1, self.h_size).expand(parent_h.size(0), 1, self.h_size)), dim=1)

        return parent_h

    def compute_state_probs(self, parent_h, encoding):
        batch_dim = parent_h.size()[0]
        #print("BATCH DIM ", batch_dim)
        #hiddens = th.zeros(batch_dim, self.max_output_degree, self.h_size)
        #probs = th.zeros(batch_dim, self.max_output_degree)
        #hiddens_comb = th.zeros(batch_dim, self.max_output_degree, self.h_size)

        #print("CELL ------------------------")
        #print("PARENT H", parent_h)
        #print("PARENT OUTPUT LABEL", parent_output_label)
        #print("ENC", encoding)
        #print("CONCAT", th.cat((parent_h, encoding),1))
        #print("HIDDENS", hiddens)
        #print("PROBS", probs)

        h = self.rec.forward(parent_h, encoding)
        probs = self.count(self.linear(h))#.squeeze(1) #vedere se squeeze ci vuole
        #print("PROBS", probs)

        h = th.tanh(self.linear2(h))

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

        h, probs = self.compute_state_probs(encs, encs)
        label = self.output_module.forward(h)  # probs?
        #print("H", h)
        #soft, onehot = self.output_module.forward(h) #probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}

    #FORWARD OTHERS
    def apply_node_func(self, nodes):
        #print("APPLY---")
        parent_h = nodes.data['parent_h'].squeeze(1)
        parent_out = nodes.data['parent_output'].squeeze(1)
        encs = nodes.data['enc']

        #parent_out = self.emb_module.forward(parent_out)

        h, probs = self.compute_state_probs(parent_h, parent_h)
        label = self.output_module.forward(h) #probs?
        #soft, onehot = self.output_module.forward(h)  # probs?

        return {'h': h, 'probs': probs, 'output': label, 'output_soft': label}


    #tree: tree to expand
    #src: node to expand
    #n: number of children of node to add
    def expand(self, tree, src):
        new = []
        probs = tree.nodes[src].data['probs']
        for i in range(round(probs.item())):
            tree.add_nodes(1)
            #print("Aggiungo arco", src, "-", len(tree.nodes())-1)
            tree.add_edges(src, len(tree.nodes())-1)
            new.append(len(tree.nodes())-1)
        return new