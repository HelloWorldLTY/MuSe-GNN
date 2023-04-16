import torch_geometric.nn
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv



class GCNEncoder_Multiinput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Multiinput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        
        conv_dict = {}
        conv_dict1 = {}
        for i in graph_list:
            conv_dict[i.show_index] = torch_geometric.nn.Sequential('x, edge_index', [(TransformerConv(i.x.shape[1], out_channels, heads=4),'x, edge_index-> x'),
                                                     (torch_geometric.nn.GraphNorm(out_channels*4), 'x -> x')])
        self.convl1 = nn.ModuleDict(conv_dict)
        self.convl2 = nn.ModuleDict(conv_dict1)
              
    def forward(self, x, edge_index, show_index):
        x = self.convl1[show_index](x, edge_index)
        x = self.activ(x)
        return x
    
class GCNEncoder_Commoninput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Commoninput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        
        conv_dict_l2 = {}
        conv_dict_l3 = {}
        conv_dict_l4 = {}
        tissue_specific_list = list(set(label_list))
        
        for i in tissue_specific_list:
            conv_dict_l2[i] = torch_geometric.nn.Sequential('x, edge_index', [(TransformerConv(out_channels*4, out_channels, heads=2),'x, edge_index -> x'),
                                                     (torch_geometric.nn.GraphNorm(out_channels*2), 'x -> x')])
            conv_dict_l3[i] = TransformerConv(out_channels*2, out_channels)
            conv_dict_l4[i] = TransformerConv(out_channels*4, out_channels)
              
        self.convl2 = nn.ModuleDict(conv_dict_l2)
        self.convl3 = nn.ModuleDict(conv_dict_l3)
        self.convl4 = nn.ModuleDict(conv_dict_l4)
        
    def forward(self, x, edge_index, show_index):
        x_inp = x
        x = self.convl2[show_index.split('__')[0]](x, edge_index)
        x = self.activ(x)
        x = self.convl3[show_index.split('__')[0]](x, edge_index)
        return x + self.convl4[show_index.split('__')[0]](x_inp, edge_index)

class MLP_edge_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, graph_list):
        super(MLP_edge_Decoder, self).__init__()
        
        dec_dict = {}
        
        self.activ = nn.Mish(inplace=True)
        for i in graph_list:
            dec_dict[i.show_index] = torch.nn.Sequential(
                                              nn.Linear(in_channels,  out_channels)
                                             , self.activ,
                                              nn.Linear(in_channels,  out_channels)
                                             , self.activ,
                                              nn.Linear(in_channels,  out_channels)
                                             )
        self.MLP = nn.ModuleDict(dec_dict)
        
    def forward(self, x, show_index):
        x = self.MLP[show_index](x)
        return x