import torch
import torch.nn as nn
import torch.nn.functional as F


# ==== Regular Loss (for minimization, or grandient ascent) ====
class MarginLoss(nn.Module):
    """
        AKA CW loss:
            for a 1D vector x and a label class y
            loss = max(0, margin - x[y] + x[i])**p, where x[i] = max x and i != y.
    """
    def __init__(self, reduction="mean", margin=1, p=1, rescale_logits=False):
        super(MarginLoss, self).__init__()
        assert reduction in ["mean", "sum", "none", None], "Reduction needs to be within ['mean', 'sum', 'none', None]"
        self.reduction = reduction
        self.margin = margin
        self.p = p
        self.rescale_logits = rescale_logits

    def forward(self, logits, labels):
        
        if self.rescale_logits:
            # logits_min, logits_max = torch.amin(logits).clone().detach(), torch.amax(logits).clone().detach()
            # logits = (logits - logits_min) / (logits_max - logits_min)
            raise RuntimeError("This implementation is abandoned.")

        correct_logits = torch.gather(logits, 1, labels.view(-1, 1)) # [n, 1]  --- x[y]
        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)

        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max # [n, 1] --  x[i]

        loss = torch.clamp(self.margin - correct_logits + max_incorrect_logits, min=0)

        # If use power values
        if self.p != 1:
            loss = loss ** self.p

        # If valid reduction 
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0, reduction="mean"):
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.reduction = reduction

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target, reduction=self.reduction)


if __name__ == "__main__":
    for _ in range(10):
        x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
        y = torch.tensor([3])

        test_loss = MarginLoss("mean", rescale_logits=True)

        loss_1 = test_loss(x, y)
        print("===========")
        print(x)
        print(y)
        print(loss_1)
        print("Shape:", y.shape)
        print("Shape:", loss_1.shape)
        # print(loss_dlr.item())

    print()
