import torch.nn as nn
import dgl
import torch as th


class TreeDecoder(nn.Module):
    def __init__(self,
                 h_size,
                 max_output_degree,
                 max_depth,
                 cell):
        super(TreeDecoder, self).__init__()
        self.h_size = h_size
        self.cell = cell
        self.max_outdegree = max_output_degree
        self.max_depth = max_depth


    def spread_encs(self, g, encs): #spread to the nodes the encoding received as input

        g.ndata['enc'] = th.zeros(len(g.nodes()), self.h_size) #zero initialization of enc of all nodes in the trees

        root_ids = [i for i in range(g.number_of_nodes()) if g.in_degree(i) == 0] #get roots of trees

        #set encs of roots
        for i in range(len(encs)):
            g.ndata['enc'][root_ids[i]] = encs[i]

        #spread encs to the whole trees
        for i in range(len(root_ids)-1):
            g.ndata['enc'][root_ids[i]+1:root_ids[i+1]] = encs[i]
        g.ndata['enc'][root_ids[-1] + 1:] = encs[-1]


    def forward(self, g, encs):

        if self.training:
            #print("TRAIN----")

            self.spread_encs(g, encs)

            # topological order
            topo_nodes = dgl.topological_nodes_generator(g)

            roots = topo_nodes[0:1]
            others = topo_nodes[1:]

            #root training computations
            g.register_message_func(self.cell.message_func)
            g.register_reduce_func(self.cell.reduce_func)
            g.register_apply_node_func(self.cell.apply_node_func_root)
            g.prop_nodes(roots)
            #print("--------------------ROOT COMPUTED-------------")

            #other nodes training computations
            g.register_apply_node_func(self.cell.apply_node_func)
            g.prop_nodes(others)
            #print("--------------------ALL COMPUTED-------------")

        else:
            #TODO: instead of unbatch and perform single node expansion, use DGL NodeFlow for batched sampling
            #print("EVAL----")
            trees = []
            features = {'parent_h': th.zeros(1, 1, self.h_size), 'parent_output': th.zeros(1, 1, self.cell.num_classes)}
            if str(self.cell) == 'DRNNCell':
                features['sibling_h'] = th.zeros(1, 1, self.h_size)
                features['sibling_output'] = th.zeros(1, 1, self.cell.num_classes)
            #create only root trees without labels
            for i in range(len(encs)):
                tree = dgl.DGLGraph()
                tree.add_nodes(1, features)
                trees.append(tree)
            g = dgl.batch(trees) #batch them
            g.ndata['enc'] = encs #set root encs
            g.ndata['pos'] = th.zeros(len((g.nodes())),self.max_outdegree) #auxiliary topological info
            g.ndata['depth'] = th.zeros(len((g.nodes())),self.max_depth+1) #auxiliary topological info

            #roots cumputations
            topo_nodes = dgl.topological_nodes_generator(g)
            g.register_message_func(self.cell.message_func)
            g.register_reduce_func(self.cell.reduce_func)
            g.register_apply_node_func(self.cell.apply_node_func_root)
            g.prop_nodes(topo_nodes)
            #print("--------------------ROOT (lvl 0): COMPUTED-------------")

            trees = dgl.unbatch(g) #unbatch to deal with single trees expansions

            nodes_id = []
            #single trees expansions
            for i in range(len(trees)):
                nodes_id.append(self.cell.expand(trees[i], 0))

            positions = [i for i in range(len(trees))]
            final_trees = [None] * len(trees)

            self.filter(nodes_id, trees, positions, final_trees) #take only nodes to process
            # print("--------------------ROOT (lvl 0): EXPANDED-------------")

            depth = 0

            #loop expansions of the lower levels
            while nodes_id:
                #print("DEPTH", depth,"/", self.max_depth)

                g = dgl.batch(trees)  # batch again to computes new nodes states
                batch_nodes_id = self.tree_node_id_to_batch_node_id(trees, nodes_id)  # ids mapping

                g.register_message_func(self.cell.message_func)
                g.register_reduce_func(self.cell.reduce_func)
                g.register_apply_node_func(self.cell.apply_node_func)
                g.prop_nodes(batch_nodes_id)

                depth += 1
                #print("--------------------lvl "+str(depth)+" NODES: COMPUTED-------------")

                if depth < self.max_depth: #if stopping criteria not reached

                    tree_nodes_id = nodes_id.copy()
                    trees = dgl.unbatch(g) #unbatch to deal with single trees expansions

                    nodes_id = []
                    # single trees expansions
                    for i in range(len(trees)):
                        tree_ids = []
                        for j in range(len(tree_nodes_id[i])):
                            id = tree_nodes_id[i][j]
                            tree_ids+=self.cell.expand(trees[i], id)
                        nodes_id.append(tree_ids)

                    self.filter(nodes_id, trees, positions, final_trees) #take only nodes to process
                    # print("--------------------lvl "+str(depth)+" NODES: EXPANDED-------------")

                else: #if stops
                    for i in range(len(trees)):
                        final_trees[positions[i]] = trees[i] #put on the final trees the last computed nodes
                    break

            g = dgl.batch(final_trees)

        return g


    # tree node id to batch node id mapping
    def tree_node_id_to_batch_node_id(self, trees, tree_nodes_id):
        l = []
        l.append(tree_nodes_id[0])
        c = trees[0].number_of_nodes()
        for i in range(1, len(tree_nodes_id)):
            l.append([int(x+c) for x in tree_nodes_id[i]])
            c += trees[i].number_of_nodes()
        return l

    def to_process(self, node_ids):
        s = 0
        for ids in node_ids:
            s+= len(ids)
        return s > 0

    #node filtering, taking the ones waiting for expansion
    def filter(self, node_ids, trees, pos, final):
        for i in range(len(node_ids) - 1, -1, -1):
            if len(node_ids[i]) == 0:
                final[pos[i]] = trees[i]
                del(trees[pos.index(pos[i])])
                del(node_ids[pos.index(pos[i])])
                pos.remove(pos[i])