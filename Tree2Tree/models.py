import torch.nn as nn

class Tree2Tree(nn.Module):

    def __init__(self,
                 encoder,
                 decoder):
        super(Tree2Tree, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, g_enc, x_enc, mask_enc, g_dec):

        self.encoder.forward(g_enc, x_enc , mask_enc) #input trees encoding
        root_ids =  [i for i in range(g_enc.number_of_nodes()) if g_enc.out_degree(i) == 0] #get only roots
        encs = g_enc.ndata['h'][root_ids] #get only roots encodings to send to decoder
        return self.decoder.forward(g_dec, encs)