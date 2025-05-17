import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()
        
        self.use_tanh = use_tanh
        self.latent_in = latent_in
        self.dropout_list = dropout
        self.dropout_prob = dropout_prob
        
        self.dropout = nn.Dropout(dropout_prob)  
        self.activation = nn.LeakyReLU()
        self.th = nn.Tanh()
        
        # Create layers
        # 3 -> 512 -> 512 -> 512 -> 509+3 -> 512 -> 512 -> 512 -> 1 ->
        self.fc_layers = nn.ModuleList()
        for i in range(len(dims)):
            # First layer's input dim = 3
            in_dim = 3 if i == 0 else dims[i-1]
            
            # Output of previous layer of latent input layer, subtract output dim by 3 
            if i+1 in latent_in:
                out_dim = dims[i] - 3
            # Last layer's output dim = 1 
            elif i == len(dims)-1:
                out_dim = 1
            else:
                out_dim = dims[i]
            
            x = nn.Linear(in_dim, out_dim)
            
            # Weight normalization for chosen fc layers
            # Last layer as per assignment's description doesn't include weight_normal -> activation -> dropout
            if weight_norm and i in norm_layers and i != len(dims)-1:
                x = nn.utils.weight_norm(x)
            
            self.fc_layers.append(x)
            
        
    
    # input: N x 3
    def forward(self, input):
        
        x = input
        for i, layer in enumerate(self.fc_layers):
            if i in self.latent_in:
                # Concat adding more columns
                x = torch.cat([x, input], dim=1)
            x = layer(x)
            
            # Last layer as per assignment's description doesn't include weight_normal -> activation -> dropout
            # Also evaluation output appears better that way :)
            if i == len(self.fc_layers) - 1:
                break
            
            x = self.activation(x)
            if i in self.dropout_list:
                x = self.dropout(x)

        if self.use_tanh:
            x = self.th(x)

        return x
