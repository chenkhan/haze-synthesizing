import torch
import torch.nn as nn


class L1_TVLoss_Charbonnier(nn.Module):

    def __init__(self):

        super(L1_TVLoss_Charbonnier, self).__init__()

        self.e = 0.000001 ** 2

    def forward(self, x):

        batch_size = x.size()[0]

        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :]))

        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))

        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1]))

        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))

        return h_tv + w_tv


class Inter_Channel_Consistancy(nn.Module):

    def __init__(self):
        super(Inter_Channel_Consistancy, self).__init__()

        self.eps = 0.000001 ** 2

    def forward(self, x):
        batch_size = x.size()[0]
        mse = torch.mean(torch.mul(x[:, 0, :, :] - x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]) +
                         torch.mul(x[:, 2, :, :] - x[:, 1, :, :], x[:, 2, :, :] - x[:, 1, :, :]) +
                         torch.mul(x[:, 0, :, :] - x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :])) / 3
        return mse


class Inter_Channel_Coherence(nn.Module):
    def __init__(self):
        super(Inter_Channel_Coherence, self).__init__()

    def forward(self, x):
        def coherence(a, b, c, d):
            eps=1/255
            return torch.mean(torch.abs(torch.abs(a-b)-torch.abs(c-d)))
        r = x[:, 0, 1:, 1:]
        g = x[:, 1, 1:, 1:]
        b = x[:, 2, 1:, 1:]
        rh = x[:, 0, :-1, 1:]
        gh = x[:, 1, :-1, 1:]
        bh = x[:, 2, :-1, 1:]
        rv = x[:, 0, 1:, :-1]
        gv = x[:, 1, 1:, :-1]
        bv = x[:, 2, 1:, :-1]

        return (coherence(r, rh, g, gh) + coherence(r, rh, b, bh) + coherence(g, gh, b, bh)) / 6 + \
               (coherence(r, rv, g, gv) + coherence(r, rv, b, bv) + coherence(g, gv, b, bv)) / 6
