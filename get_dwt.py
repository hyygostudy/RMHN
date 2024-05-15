import test_datasets
import torch
import torch.nn
import torchvision
from options import arguments
opt=arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i, data in enumerate(test_datasets.testloader):
    data = data.to(device)
    x = data
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_HL = x_HL*10
    x_LH = -x1 + x2 - x3 + x4
    x_LH = x_LH*10
    x_HH = x1 - x2 - x3 + x4
    x_HH = x_HH*10
    torchvision.utils.save_image(x_LL, opt.x_LL + '%.5d.png' % i)
    torchvision.utils.save_image(x_HL, opt.x_HL + '%.5d.png' % i)
    torchvision.utils.save_image(x_LH, opt.x_LH + '%.5d.png' % i)
    torchvision.utils.save_image(x_HH, opt.x_HH + '%.5d.png' % i)


