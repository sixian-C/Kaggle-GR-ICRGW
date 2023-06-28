import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# Average dice score for the examples in a batch
def dice_avg(y_p, y_t,smooth=1e-3):
    i = torch.sum(y_p * y_t, dim=(2, 3))
    u = torch.sum(y_p, dim=(2, 3)) + torch.sum(y_t, dim=(2, 3))
    score = (2 * i + smooth)/(u + smooth)
    return torch.mean(score)


def dice_loss_avg(y_p,y_t):
    return 1-dice_avg(y_p,y_t)

def dice_global(y_p,y_t,smooth=1e-3):
    with torch.no_grad():
        intersection = torch.sum(y_p * y_t)
        union = torch.sum(y_p) + torch.sum(y_t)

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice

def dice_loss_global(y_p,y_t):
    return 1-dice_global(y_p,y_t)