import torch

def compute_accumulated_transmittance(betas):
    """
    Computes the accumulated transmittance along rays.

    Args:
        betas (torch.Tensor): A tensor of shape [num_rays, num_samples] containing
                              the per-sample transmittance values (1 - alpha).

    Returns:
        torch.Tensor: Accumulated transmittance for each sample.
    """
    # TODO: Compute the cumulative product of betas along the sample dimension (dim=1)
    # This represents how much light transmits from ray origin to each point
    accumulated_transmittance =  
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
                      accumulated_transmittance[:, :-1]), dim=1)

def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
    
    t = torch.linspace(tn, tf, nb_bins).to(device) # [nb_bins]
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))
    
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1) # [nb_rays, nb_bins, 3]    
    
    colors, density = model.intersect(x.reshape(-1, 3),\
                                      rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    
    colors = colors.reshape((x.shape[0], nb_bins, 3)) # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins))
    
    alpha = 1 - torch.exp(- density * delta.unsqueeze(0)) # [nb_rays, nb_bins, 1]
    
    # TODO: Compute weights = T_i * α_i
    weights = # [nb_rays, nb_bins]
    
    if white_bckgr:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
        weight_sum = weights.sum(-1) # [nb_rays]
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
    
    return c