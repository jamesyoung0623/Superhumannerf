import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import initseq
from core.nets.human_nerf.activation import trunc_exp  # dj
from datetime import datetime


class CanonicalMLP(nn.Module):
    def __init__(self, input_ch=3, mlp_depth=8, mlp_width=256, skips=None, **_):
        super(CanonicalMLP, self).__init__()

        # original -----------------------------------------------
        self.mlp_depth = mlp_depth # 3
        self.mlp_width = mlp_width # 64
        self.input_ch  = input_ch  # 63 (pos_embed) / 32 (hash_encode)

        if skips is None:
            skips = [mlp_depth - 2] # dj: include first mix_net
        mix_net = [nn.Linear(self.input_ch, self.mlp_width), nn.ReLU()]

        layers_to_cat_input = []
        for i in range(self.mlp_depth-1):
            if i in skips:
                layers_to_cat_input.append(len(mix_net))
                mix_net += [nn.Linear(self.mlp_width + self.input_ch, self.mlp_width), nn.ReLU()]
            else:
                mix_net += [nn.Linear(self.mlp_width, self.mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(mix_net)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        self.output_linear = nn.Sequential(nn.Linear(self.mlp_width, 4))
        initseq(self.output_linear)

        # original -----------------------------------------------


        # num_layers=3
        # hidden_dim=64
        # geo_feat_dim=15
        # num_layers_color=3
        # hidden_dim_color=64      
        # sphar_dim = 32  

        # # sigma network
        # self.num_layers = num_layers
        # self.hidden_dim = hidden_dim
        # self.geo_feat_dim = geo_feat_dim
        # self.sphar_dim = sphar_dim
        # sigma_net = []
        # for l in range(num_layers):
        #     if l == 0:
        #         in_dim = self.input_ch
        #     else:
        #         in_dim = hidden_dim
        #     if l == num_layers - 1:
        #         out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
        #     else:
        #         out_dim = hidden_dim
        #     sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        # self.sigma_net = nn.ModuleList(sigma_net)
        # # print('self.sigma_net')
        # # print(self.sigma_net)

        # # color network
        # self.num_layers_color = num_layers_color        
        # self.hidden_dim_color = hidden_dim_color
        # color_net =  []
        # for l in range(num_layers_color):
        #     if l == 0:
        #         in_dim = self.sphar_dim + self.geo_feat_dim
        #     else:
        #         in_dim = hidden_dim_color
        #     if l == num_layers_color - 1:
        #         out_dim = 3 # 3 rgb
        #     else:
        #         out_dim = hidden_dim_color
        #     color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        # self.color_net = nn.ModuleList(color_net)
        # # print('self.color_net')
        # # print(self.color_net)

    # def forward(self, pos_xyz, bound, **_):
    def forward(self, pos_embedded, dir_embedded, hash_encode, **_):    
        # dj: not input cammera direction by assuming that the monocular camera is fixed
        if not hash_encode:
            h = pos_embedded # 63D pos_embedded
        else:
            # pos_encode = self.encoder(pos_embedded, bound=1.0)    # dj: bound [1.0] needs > 1 for including some pts large than 1
            h = pos_embedded # 32D pos_embedded

        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embedded, h], dim=-1)  # 32+64=96
            h = self.pts_linears[i](h)
        outputs = self.output_linear(h)  # output 4D rgb+sigma
        
        # sigma
        # pos_embedded = self.encoder(pos_embedded, bound=self.bound)

        # bound =1.
        # if pos_embedded.max() > bound: 
        #     print('larger pos_embedded bound')
        #     print(pos_embedded.max())
        # if pos_embedded.min() < -bound:
        #     print('smaller pos_embedded bound')
        #     print(pos_embedded.min())           
        # if dir_embedded.max() > 1.:
        #     print('larger dir_embedded bound')
        #     print(dir_embedded.max())           
        # if dir_embedded.min() < -1.:
        #     print('smaller dir_embedded bound')
        #     print(dir_embedded.min())            

        # h = pos_embedded
        # for l in range(self.num_layers):
        #     h = self.sigma_net[l](h)
        #     if l != self.num_layers - 1:
        #         h = F.relu(h, inplace=True)

        # # sigma = F.relu(h[..., 0])
        # # sigma = trunc_exp(h[..., 0])
        # sigma = h[..., 0]
        # geo_feat = h[..., 1:]

        # # color
        # # dir_embedded = self.encoder_dir(dir_embedded)
        # # h = torch.cat([dir_embedded, geo_feat], dim=-1)
        # h = torch.cat([pos_embedded, geo_feat], dim=-1)
        # for l in range(self.num_layers_color):
        #     h = self.color_net[l](h)
        #     if l != self.num_layers_color - 1:
        #         h = F.relu(h, inplace=True)
        
        # # sigmoid activation for rgb
        # # color = torch.sigmoid(h)
        # color = h

        # # breakpoint()
        # outputs = torch.cat((color, torch.unsqueeze(sigma, dim=1)), dim=1)  # output 4D rgb+sigma



        return outputs    
 

























# import torch
# import torch.nn as nn

# from core.utils.network_util import initseq


# class CanonicalMLP(nn.Module):
#     def __init__(self, mlp_depth=8, mlp_width=256, 
#                  input_ch=3, skips=None,
#                  **_):
#         super(CanonicalMLP, self).__init__()

#         if skips is None:
#             skips = [4]

#         self.mlp_depth = mlp_depth
#         self.mlp_width = mlp_width
#         self.input_ch = input_ch
        
#         mix_net = [nn.Linear(input_ch, mlp_width), nn.ReLU()]

#         layers_to_cat_input = []
#         for i in range(mlp_depth-1):
#             if i in skips:
#                 layers_to_cat_input.append(len(mix_net))
#                 mix_net += [nn.Linear(mlp_width + input_ch, mlp_width), 
#                                    nn.ReLU()]
#             else:
#                 mix_net += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
#         self.layers_to_cat_input = layers_to_cat_input

#         self.pts_linears = nn.ModuleList(mix_net)
#         initseq(self.pts_linears)

#         # output: rgb + sigma (density)
#         self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4))
#         initseq(self.output_linear)


#     def forward(self, pos_embed, **_):
#         h = pos_embed
#         for i, _ in enumerate(self.pts_linears):
#             if i in self.layers_to_cat_input:
#                 h = torch.cat([pos_embed, h], dim=-1)
#             h = self.pts_linears[i](h)

#         outputs = self.output_linear(h)

#         return outputs    
        