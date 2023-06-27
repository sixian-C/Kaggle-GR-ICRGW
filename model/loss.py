import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

# Average dice score for the examples in a batch
# def dice_avg(y_p, y_t,smooth=1e-3):
#     i = torch.sum(y_p * y_t, dim=(2, 3))
#     u = torch.sum(y_p, dim=(2, 3)) + torch.sum(y_t, dim=(2, 3))
#     score = (2 * i + smooth)/(u + smooth)
#     return torch.mean(score)


# def dice_loss_avg(y_p,y_t):
#     return 1-dice_avg(y_p,y_t)

# def dice_global(y_p,y_t,smooth=1e-3):

#     intersection = torch.sum(y_p * y_t)
#     union = torch.sum(y_p) + torch.sum(y_t)

#     dice = (2.0 * intersection + smooth) / (union + smooth)

#     return dice

# def dice_loss_global(y_p,y_t):
#     return 1-dice_global(y_p,y_t)

