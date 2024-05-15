import torch
import torch.nn
import torch.optim
import torchvision
from model import *
import test_datasets
import dwt_iwt
from options import arguments
opt=arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer')


net1 = Model1()
net2 = Model2()
net3 = Model3()
net1.cuda()
net2.cuda()
net3.cuda()
init_model(net1)
init_model(net2)
init_model(net3)
net1 = torch.nn.DataParallel(net1, device_ids=opt.device_ids)
net2 = torch.nn.DataParallel(net2, device_ids=opt.device_ids)
net3 = torch.nn.DataParallel(net3, device_ids=opt.device_ids)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
params_trainable3 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=opt.lr, betas=opt.betas, eps=1e-6, weight_decay=opt.weight_decay)
optim2 = torch.optim.Adam(params_trainable2, lr=opt.lr, betas=opt.betas, eps=1e-6, weight_decay=opt.weight_decay)
optim3 = torch.optim.Adam(params_trainable3, lr=opt.lr, betas=opt.betas, eps=1e-6, weight_decay=opt.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, opt.weight_step, gamma=opt.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, opt.weight_step, gamma=opt.gamma)
weight_scheduler3 = torch.optim.lr_scheduler.StepLR(optim3, opt.weight_step, gamma=opt.gamma)
load(opt.model_path1, net1, optim1)
load(opt.model_path2, net2, optim2)
load(opt.model_path3, net3, optim3)
net1.eval()
net2.eval()
net3.eval()
dwt = dwt_iwt.DWT()
iwt = dwt_iwt.IWT()

with torch.no_grad():
    for i, data in enumerate(test_datasets.testloader):
        data = data.to(device)
        print(data.shape)
        cover = data[0:1]
        secret1 = data[1: 2]
        secret2 = data[2: 3]
        secret3 = data[3: 4]
        print(cover.shape)
        print(secret1.shape)
        print(secret2.shape)
        print(secret3.shape)
        cover_input = dwt(cover)
        secret1_input = dwt(secret1)
        secret2_input = dwt(secret2)
        secret3_input = dwt(secret3)

        # forward1:
        input_img1 = torch.cat((cover_input, secret1_input), 1)
        output1 = net1(input_img1)
        output_stego1 = output1.narrow(1, 0, 4 * opt.channels_in)
        output_lost_matrix_1 = output1.narrow(1, 4 * opt.channels_in, output1.shape[1] - 4 * opt.channels_in)
        stego1 = iwt(output_stego1)
        r1 = iwt(output_lost_matrix_1)
        cwrong_map1 = (cover - stego1) * 20

        # forward2:

        input_img2 = torch.cat((output_stego1, secret2_input), 1)
        output2 = net2(input_img2)
        output_stego2 = output2.narrow(1, 0, 4 * opt.channels_in)
        output_lost_matrix_2 = output2.narrow(1, 4 * opt.channels_in, output1.shape[1] - 4 * opt.channels_in)
        stego2 = iwt(output_stego2)
        r2 = iwt(output_lost_matrix_2)
        cwrong_map2 = (cover - stego2) * 20

        # forward3:
        print(output_stego2.shape)
        print(output_stego1.shape)
        input_img3 = torch.cat((output_stego2, secret3_input), 1)
        output3 = net3(input_img3)
        output_stego3 = output3.narrow(1, 0, 4 * opt.channels_in)
        output_lost_matrix_3 = output3.narrow(1, 4 * opt.channels_in, output1.shape[1] - 4 * opt.channels_in)
        stego3 = iwt(output_stego3)
        r3 = iwt(output_lost_matrix_3)
        cwrong_map3 = (cover - stego3) * 20

        # backward3:

        z3 = gauss_noise(output_lost_matrix_3.shape)
        output_rev3 = torch.cat((output_stego3, z3), 1)
        backward_img3 = net3(output_rev3, rev=True)
        recs33 = backward_img3.narrow(1, 4 * opt.channels_in, backward_img3.shape[1] - 4 * opt.channels_in)
        recc33 = backward_img3.narrow(1, 0, 4 * opt.channels_in)
        recs3 = iwt(recs33)
        recc3 = iwt(recc33)
        swrong_map3 = (secret3 - recs3) * 20

        # backward2:

        z2 = gauss_noise(output_lost_matrix_2.shape)
        output_rev2 = torch.cat((output_stego2, z2), 1)
        backward_img2 = net2(output_rev2, rev=True)
        recs22 = backward_img2.narrow(1, 4 * opt.channels_in, backward_img2.shape[1] - 4 * opt.channels_in)
        recc22 = backward_img2.narrow(1, 0, 4 * opt.channels_in)
        recs2 = iwt(recs22)
        recc2 = iwt(recc22)
        swrong_map2 = (secret2 - recs2) * 20

        # backward1:

        z1 = gauss_noise(output_lost_matrix_1.shape)
        output_rev1 = torch.cat((output_stego1, z1), 1)
        backward_img1 = net1(output_rev1, rev=True)
        recs11 = backward_img1.narrow(1, 4 * opt.channels_in, backward_img1.shape[1] - 4 * opt.channels_in)
        recc11 = backward_img1.narrow(1, 0, 4 * opt.channels_in)
        recs1 = iwt(recs11)
        recc1 = iwt(recc11)
        swrong_map1 = (secret1 - recs1) * 20

        # save images

        torchvision.utils.save_image(cover, opt.test_cover_path + '%.5d.png' % i)
        torchvision.utils.save_image(secret1, opt.test_secret_path1 + '%.5d.png' % i)
        torchvision.utils.save_image(secret2, opt.test_secret_path2 + '%.5d.png' % i)
        torchvision.utils.save_image(secret3, opt.test_secret_path3 + '%.5d.png' % i)
        torchvision.utils.save_image(stego1, opt.test_stego_path1 + '%.5d.png' % i)
        torchvision.utils.save_image(stego2, opt.test_stego_path2 + '%.5d.png' % i)
        torchvision.utils.save_image(stego3, opt.test_stego_path3 + '%.5d.png' % i)
        torchvision.utils.save_image(recs1, opt.test_recs_path1 + '%.5d.png' % i)
        torchvision.utils.save_image(recs2, opt.test_recs_path2 + '%.5d.png' % i)
        torchvision.utils.save_image(recs3, opt.test_recs_path3 + '%.5d.png' % i)
        torchvision.utils.save_image(cwrong_map1, opt.test_cwrong_map_path1 + '%.5d.png' % i)
        torchvision.utils.save_image(cwrong_map2, opt.test_cwrong_map_path2 + '%.5d.png' % i)
        torchvision.utils.save_image(cwrong_map3, opt.test_cwrong_map_path3 + '%.5d.png' % i)
        torchvision.utils.save_image(swrong_map1, opt.test_swrong_map_path1 + '%.5d.png' % i)
        torchvision.utils.save_image(swrong_map2, opt.test_swrong_map_path2 + '%.5d.png' % i)
        torchvision.utils.save_image(swrong_map3, opt.test_swrong_map_path3 + '%.5d.png' % i)