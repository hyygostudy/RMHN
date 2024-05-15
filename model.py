import torch.optim
import torch.nn as nn
from imn import IMN1,IMN2
from options import arguments
opt=arguments()

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.model = IMN1()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.model = IMN2()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.model = IMN2()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = opt.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
