import torch
import torch.nn as nn
import tinycudann as tcnn

class Voxels(nn.Module):
    
    def __init__(self, nb_voxels=100, scale=1, device='cpu'):
        super(Voxels, self).__init__()
        
        self.voxels = torch.nn.Parameter(torch.rand((nb_voxels, nb_voxels, nb_voxels, 4), 
                                                    device=device, requires_grad=True))
        # 4 because 3 colors and density
        self.nb_voxels = nb_voxels
        self.device = device
        self.scale = scale
        
    def forward(self, xyz, d):
        
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        
        cond = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))
        
        indx = (x[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indy = (y[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indz = (z[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        
        colors_and_densities = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        colors_and_densities[cond, :3] = self.voxels[indx, indy, indz, :3]
        colors_and_densities[cond, -1] = self.voxels[indx, indy, indz, -1]
         
        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1:])
        
    
    def intersect(self, x, d):
        return self.forward(x, d)
    
    
class Nerf(nn.Module):
    
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256):
        super(Nerf, self).__init__()
        
        # TODO 1: Define block1 - MLP that takes positional encoding of xyz
        input_dim = 
        self.block1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        
        # TODO 2: Define block2 - takes features + input again (skip connection)
        input_dim2 = 
        self.block2 = nn.Sequential(nn.Linear(input_dim2, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1),)
        
        
        # TODO 3: Define RGB head - maps features + direction encoding to RGB
        input_dim3 =
        self.rgb_head = nn.Sequential(nn.Linear(input_dim3, hidden_dim // 2), nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())
        
        self.Lpos = Lpos
        self.Ldir = Ldir
        
    def positional_encoding(self, x, L):
        out=[]
        '''
        The original NeRF paper proposes using positional encoding with sinusoids at multiple frequencies,
        but it does not mention keeping the original input x. In practice, including x is crucial 
        — it provides the network with the raw spatial information needed to model low-frequency components,
        making the representation both richer and more stable.
        This implementation does not follow the original paper's suggestion.
        '''
        # x → [x, sin(2^0 * x), cos(2^0 * x), ..., sin(2^{L-1} * x), cos(2^{L-1} * x)]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
            
                                    
        
    def forward(self, xyz, d):
        
        x_emb = self.positional_encoding(xyz, self.Lpos) # [batch_size, Lpos * 6 + 3]
        d_emb = self.positional_encoding(d, self.Ldir) # [batch_size, Ldir * 6 + 3]
        
        h = self.block1(x_emb) # [batch_size, hidden_dim]
        h = self.block2(torch.cat((h, x_emb), dim=1)) # [batch_size, hidden_dim + 1]
        sigma = h[:, -1]
        h = h[:, :-1] # [batch_size, hidden_dim]
        c = self.rgb_head(torch.cat((h, d_emb), dim=1))
        
        return c, torch.relu(sigma)
    
    def intersect(self, x, d):
        return self.forward(x, d)
