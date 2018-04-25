from gan_model import ConvGen, ConvDis
from utils import AverageMeter, save_checkpoint, Plotter_GAN_TV, Plotter_GAN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
import torchvision.utils as vutils

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Colorization using GAN')
# Paths/Saving/Reload
parser.add_argument('path', type=str,
                    help='Root path for dataset')
parser.add_argument('--test', default='', type=str,
                    help='Path to the model, for testing')
parser.add_argument('--model_G', default='', type=str,
                    help='Path to resume for Generator model')
parser.add_argument('--model_D', default='', type=str,
                    help='Path to resume for Discriminator model')
parser.add_argument('-s', '--save', action="store_true",
                    help='Save model?')
parser.add_argument('--dataset', type=str,
                    help='which dataset?', choices=['sc2', 'flower', 'bob'])
parser.add_argument('-p', '--plot', action="store_true",
                    help='Plot accuracy and loss diagram?')
# Hyperparameters
parser.add_argument('--large', action="store_true",
                    help='Use larger images?')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size: default 4')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate for optimizer')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay for optimizer')
parser.add_argument('--num_epoch', default=20, type=int,
                    help='Number of epochs')
parser.add_argument('--lamb', default=100, type=int,
                    help='Lambda for L1 Loss')
# Hardware
parser.add_argument('-c', '--use_cuda', action='store_true',
                    help='Use cuda for training')
parser.add_argument('--gpu', default=0, type=int,
                    help='Which GPU to use?')


def main():
    global args, device, writer, criterion, L1, val_batch_size, img_path, model_path
    args = parser.parse_args()
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    size = ''
    if args.large:
        size = '_large'
    model_name = '{}{}_lambda={}_bs={}_lr={}_wd={}_n_epochs={}/'.format(args.dataset,
                                                                        size,
                                                                        args.lamb,
                                                                        args.batch_size,
                                                                        str(args.lr),
                                                                        str(args.weight_decay),
                                                                        args.num_epoch)
    print('Now training {}'.format(model_name))
    # date = '1220'
    writer = SummaryWriter(log_dir='runs/{}'.format(model_name))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_G = ConvGen()
    model_D = ConvDis(large=args.large)

    start_epoch_G = start_epoch_D = 0

    if args.model_G:
        print('Resuming model G: {}'.format(args.model_G))
        checkpoint_G = torch.load(args.model_G)
        model_G.load_state_dict(checkpoint_G['state_dict'])
        start_epoch_G = checkpoint_G['epoch']

    if args.model_D:
        print('Resuming model D: {}'.format(args.model_D))
        checkpoint_D = torch.load(args.model_D)
        model_D.load_state_dict(checkpoint_D['state_dict'])
        start_epoch_D = checkpoint_D['epoch']

    # Check
    assert start_epoch_G == start_epoch_D

    if args.model_G == '' and args.model_D == '':
        print('Not resume training.')

    # model_G.cuda()
    # model_D.cuda()
    model_G.to(device)
    model_D.to(device)

    # optimizer
    optimizer_G = optim.Adam(model_G.parameters(),
                             lr=args.lr, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(model_D.parameters(),
                             lr=args.lr, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=args.weight_decay)
    if args.model_G:
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
    if args.model_D:
        optimizer_D.load_state_dict(checkpoint_D['optimizer'])

    # Define loss functions
    criterion = nn.BCELoss()
    L1 = nn.L1Loss()

    # dataset
    # data_root = '/home/users/u5612799/DATA/Spongebob/'
    data_root = args.path
    dataset = args.dataset
    if dataset == 'sc2':
        from load_data import SC2_Dataset as myDataset
    elif dataset == 'flower':
        from load_data import Flower_Dataset as myDataset
    elif dataset == 'bob':
        from load_data import Spongebob_Dataset as myDataset
    else:
        raise ValueError('dataset type not supported')

    if args.large:
        image_transform = transforms.Compose([transforms.CenterCrop(480),
                                              transforms.ToTensor()])
    else:
        image_transform = transforms.Compose([transforms.CenterCrop(224),
                                              transforms.ToTensor()])

    data_train = myDataset(data_root, mode='train',
                           transform=image_transform,
                           types='raw',
                           shuffle=True,
                           large=args.large
                           )

    train_loader = data.DataLoader(data_train,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=4)

    data_val = myDataset(data_root, mode='test',
                         transform=image_transform,
                         types='raw',
                         shuffle=True,
                         large=args.large
                         )

    val_loader = data.DataLoader(data_val,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)

    # Define validation batch size
    val_batch_size = val_loader.batch_size

    # set up plotter, path, etc.
    global iteration, print_interval, plotter, plotter_basic
    iteration = 0
    print_interval = 5
    plotter = Plotter_GAN_TV()
    plotter_basic = Plotter_GAN()

    # Define paths
    img_path = 'img/{}/'.format(model_name)
    model_path = 'model/{}/'.format(model_name)

    # Create the folders
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # start loop
    start_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch - 1))
        print('-' * 20)
        if epoch == 0:
            val_errG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch=-1,
                                          global_step=global_step)
            # TensorboardX
            writer.add_scalar('val/val_errG', val_errG, global_step)
            writer.add_scalar('val/val_errD', val_errD, global_step)

        # train
        train_errG, train_errD, global_step = train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch,
                                                    iteration)
        # validate
        val_errG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch,
                                      global_step=global_step)

        # TensorboardX
        print('global_step = {}'.format(global_step))
        writer.add_scalar('val/val_errG', val_errG, global_step)
        writer.add_scalar('val/val_errD', val_errD, global_step)

        plotter.train_update(train_errG, train_errD)
        plotter.val_update(val_errG, val_errD)
        plotter.draw(img_path + 'train_val.png')

        if args.save:
            print('Saving checkpoint.')
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_G.state_dict(),
                             'optimizer': optimizer_G.state_dict(),
                             },
                            filename=os.path.join(model_path, 'G_epoch_{}.pth.tar'.format(epoch)))
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_D.state_dict(),
                             'optimizer': optimizer_D.state_dict(),
                             },
                            filename=os.path.join(model_path, 'D_epoch_{}.pth.tar'.format(epoch)))


def train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration):
    # Change the value of global variable global_step (to match in validate)
    errorG = AverageMeter()  # will be reset after each epoch
    errorD = AverageMeter()  # will be reset after each epoch
    errorG_basic = AverageMeter()  # basic will be reset after each print
    errorD_basic = AverageMeter()  # basic will be reset after each print
    errorD_real = AverageMeter()
    errorD_fake = AverageMeter()
    errorG_GAN = AverageMeter()
    errorG_R = AverageMeter()

    model_G.train()
    model_D.train()

    real_label = 1
    fake_label = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        ########################
        # update D network
        ########################

        # Train with real
        model_D.zero_grad()
        output = model_D(target)
        label = torch.FloatTensor(target.size(0)).fill_(real_label).to(device)
        labelv = Variable(label)
        errD_real = criterion(torch.squeeze(output), labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # Train with fake
        fake = model_G(data)
        labelv = Variable(label.fill_(fake_label))
        output = model_D(fake.detach())
        errD_fake = criterion(torch.squeeze(output), labelv)
        errD_fake.backward()
        D_G_x1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizer_D.step()

        ########################
        # update G network
        ########################

        model_G.zero_grad()
        labelv = Variable(label.fill_(real_label))
        output = model_D(fake)
        errG_GAN = criterion(torch.squeeze(output), labelv)
        errG_L1 = L1(fake.view(fake.size(0), -1), target.view(target.size(0), -1))

        errG = errG_GAN + args.lamb * errG_L1
        errG.backward()
        D_G_x2 = output.data.mean()
        optimizer_G.step()

        # TensorboardX
        global_step = iteration + len(train_loader) * epoch
        writer.add_scalar('train/errG', errG.data[0], global_step)
        writer.add_scalar('train/errD', errD.data[0], global_step)

        writer.add_scalar('train/errD_real', errD_real.data[0], global_step)
        writer.add_scalar('train/errD_fake', errD_fake.data[0], global_step)

        # store error values
        errorG.update(errG.data[0], target.size(0), history=1)
        errorD.update(errD.data[0], target.size(0), history=1)

        errorG_basic.update(errG.data[0], target.size(0), history=1)
        errorD_basic.update(errD.data[0], target.size(0), history=1)

        errorD_real.update(errD_real.data[0], target.size(0), history=1)
        errorD_fake.update(errD_fake.data[0], target.size(0), history=1)

        errorD_real.update(errD_real.data[0], target.size(0), history=1)
        errorD_fake.update(errD_fake.data[0], target.size(0), history=1)

        errorG_GAN.update(errG_GAN.data[0], target.size(0), history=1)
        errorG_R.update(errG_L1.data[0], target.size(0), history=1)

        if iteration % print_interval == 0:
            print(
                'Training epoch {}: [{}/{}]: '
                'Loss_D: {:.2f} (R {:.2f} + F {:.2f}) | '
                'Loss_G: {:.2f} (GAN {:.2f} + R {:.2f}) '
                'D(x): {:.2f} D(G(z)): {:.2f} / {:.2f}'.format(epoch, i, len(train_loader),
                                                               errorD_basic.avg, errorD_real.avg, errorD_fake.avg,
                                                               errorG_basic.avg, errorG_GAN.avg, errorG_R.avg,
                                                               D_x, D_G_x1, D_G_x2))
            # Plot image
            plotter_basic.g_update(errorG_basic.avg)
            plotter_basic.d_update(errorD_basic.avg)
            plotter_basic.draw(img_path + 'train_basic.png')

            # Reset AverageMeter
            errorG_basic.reset()
            errorD_basic.reset()
            errorD_real.reset()
            errorD_fake.reset()
            errorG_GAN.reset()
            errorG_R.reset()

        iteration += 1

    return errorG.avg, errorD.avg, global_step


def validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, global_step):
    errorG = AverageMeter()
    errorD = AverageMeter()

    model_G.eval()
    model_D.eval()

    real_label = 1
    fake_label = 0

    for i, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        ########################
        # D network
        ########################
        # validate with real
        output = model_D(target)
        label = torch.FloatTensor(target.size(0)).fill_(real_label).to(device)
        labelv = Variable(label)
        errD_real = criterion(torch.squeeze(output), labelv)

        # validate with fake
        fake = model_G(data)
        labelv = Variable(label.fill_(fake_label))
        output = model_D(fake.detach())
        errD_fake = criterion(torch.squeeze(output), labelv)

        errD = errD_real + errD_fake

        ########################
        # G network
        ########################
        labelv = Variable(label.fill_(real_label))
        output = model_D(fake)
        errG_GAN = criterion(torch.squeeze(output), labelv)
        errG_L1 = L1(fake.view(fake.size(0), -1), target.view(target.size(0), -1))

        errG = errG_GAN + args.lamb * errG_L1

        # TensorboardX
        # writer.add_scalar('val/errG', errG.data[0], global_step)
        # writer.add_scalar('val/errD', errD.data[0], global_step)

        errorG.update(errG.data[0], target.size(0), history=1)
        errorD.update(errD.data[0], target.size(0), history=1)

        if i == 0:
            vis_result(data.data, target.data, fake.data, epoch, global_step)

        if i % 50 == 0:
            print('Validating epoch {}: [{}/{}]'.format(epoch, i, len(val_loader)))

    print('Validation epoch {}: Loss_D: {:.2f} | Loss_G: {:.2f}'.format(epoch, errorD.avg, errorG.avg))

    return errorG.avg, errorD.avg


def vis_result(data, target, output, epoch, global_step):
    """
    Visualize images for GAN
    """
    # TensorboardX
    n_images = 4  # How many images to display in TensorBoard

    if global_step == 0:  # First iteration, only save once
        x = vutils.make_grid(data[:n_images, :, :, :],
                             nrow=2,
                             normalize=True,
                             scale_each=True)
        writer.add_image('val/data',
                         x.view(454, 454, 3).numpy(),  # Cast to numpy since tensorboardX does not support Pytorch 0.4
                         global_step)

        x = vutils.make_grid(target[:n_images, :, :, :],
                             nrow=2,
                             normalize=True,
                             scale_each=True)
        writer.add_image('val/target',
                         x.view(454, 454, 3).numpy(),  # Cast to numpy since tensorboardX does not support Pytorch 0.4
                         global_step)

    x = vutils.make_grid(output[:n_images, :, :, :],
                         nrow=2,
                         normalize=True,
                         scale_each=True)
    writer.add_image('val/fake',
                     x.view(454, 454, 3).numpy(),  # Cast to numpy since tensorboardX does not support Pytorch 0.4
                     global_step)

    img_list = []
    for i in range(min(32, val_batch_size)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0).cpu().numpy()
        raw = target[i].cpu().numpy()
        pred = output[i].cpu().numpy()

        raw_rgb = (np.transpose(raw, (1, 2, 0)).astype(np.float64) + 1) / 2.
        pred_rgb = (np.transpose(pred, (1, 2, 0)).astype(np.float64) + 1) / 2.

        grey = np.transpose(l, (1, 2, 0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)

        # writer.add_image('val/grey_{}'.format(i), grey, global_step)
        # writer.add_image('val/raw_rg_{}'.format(i), raw_rgb, global_step)
        # writer.add_image('val/pred_rg_{}'.format(i), pred_rgb, global_step)

        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4 * i:4 * (i + 1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)

    plt.figure(figsize=(36, 27))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path + 'epoch%d_val.png' % epoch)
    plt.clf()


if __name__ == '__main__':
    # torch.device object used throughout this script
    main()
    # Export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print('Closed the TensorboardX writer')
