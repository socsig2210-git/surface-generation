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

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Leaky ReLU layers, Dropout layers and a tanh layer.
                     
        self.dropout_prob = dropout_prob
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_prob)       

        self.fc1 = nn.utils.weight_norm(nn.Linear(3, 512))
        self.fc2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc3 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc4 = nn.utils.weight_norm(nn.Linear(512, 509))
        self.fc5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc7 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc8 = nn.Linear(512, 1)
        self.th = nn.Tanh()

        
        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        
        # First 7 layers: weight_normal -> LeakyRelu common learnable slope -> dropout layer 
        
        # TODO: CREATE COMMON LEAKYRELU LAYER FOR ALL
        
        x = self.fc1(input)
        x = self.dropout(F.leaky_relu(x))
        
        x = self.fc2(x)
        x = self.dropout(F.leaky_relu(x))
   
        x = self.fc3(x)
        x = self.dropout(F.leaky_relu(x))
        
        x = self.fc4(x)
        x = self.dropout(F.leaky_relu(x))
        
        x = self.fc5(torch.cat([x, input], dim=1)) # add points as input at 5th fc layer's input
        x = self.dropout(F.leaky_relu(x))
        
        x = self.fc6(x)
        x = self.dropout(F.leaky_relu(x))
        
        x = self.fc7(x)
        x = self.dropout(F.leaky_relu(x))
        
        x = self.fc8(x)
        x = self.th(x)
        
        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x
