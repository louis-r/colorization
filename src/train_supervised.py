"""
Training for GAN
"""
# pylint: disable=invalid-name, global-variable-undefined

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as torchdata
from torchvision import transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from gan.gan_model import ConvGen, ConvDis
from gan.utils import AverageMeter, save_checkpoint, Plotter_GAN_TV, Plotter_GAN

parser = argparse.ArgumentParser(description='Colorization using GAN')
# Paths/Saving/Reload
parser.add_argument('path', type=str,
                    help='Root path for dataset')
parser.add_argument('--test', default='', type=str,
                    help='Path to the model, for testing')
parser.add_argument('--model_G', default='', type=str,
                    help='Path to resume for Generator model')
parser.add_argument('-s', '--save', action="store_true",
                    help='Save model?')
parser.add_argument('--dataset', type=str,
                    help='which dataset?', choices=['sc2', 'flower', 'bob'])
parser.add_argument('-p', '--plot', action="store_true",
                    help='Plot accuracy and loss diagram?')
parser.add_argument('--prefix', type=str,
                    help='Possible prefix for logs')
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
parser.add_argument('--lamb', default=0, type=int,
                    help='Lambda for L1 Loss')
# Hardware
parser.add_argument('-c', '--use_cuda', action='store_true',
                    help='Use cuda for training')
parser.add_argument('--gpu', default=0, type=int,
                    help='Which GPU to use?')


def main():
    """
    Executable
    """
    # Declaring globals to be used in train, validate functions
    # noinspection PyGlobalUndefined
    global args, writer, criterion, L1, val_batch_size, img_path, model_path

    args = parser.parse_args()
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
    if args.prefix:
        model_name = '{}_{}'.format(args.prefix, model_name)

    print('Now training {}'.format(model_name))
    writer = SummaryWriter(log_dir='runs/{}'.format(model_name))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Loading models
    model_G = ConvGen()

    # Setting starting epochs
    start_epoch_G = 0

    if args.model_G:
        print('Resuming model G: {}'.format(args.model_G))
        checkpoint_G = torch.load(args.model_G)
        model_G.load_state_dict(checkpoint_G['state_dict'])
        start_epoch_G = checkpoint_G['epoch']

    if args.model_G == '':
        print('Not resuming training for G.')

    if args.use_cuda:
        model_G.cuda()

    # Define optimizer
    optimizer_G = optim.Adam(model_G.parameters(),
                             lr=args.lr, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=args.weight_decay)
    if args.model_G:
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])

    # Define loss function
    criterion = nn.MSELoss()

    # Define dataset
    data_root = args.path
    dataset = args.dataset
    if dataset == 'sc2':
        from gan.load_data import SC2Dataset as myDataset
    elif dataset == 'flower':
        from gan.load_data import FlowerDataset as myDataset
    elif dataset == 'bob':
        from gan.load_data import BobDataset as myDataset
    else:
        raise ValueError('dataset type not supported')

    if args.large:
        image_transform = transforms.Compose([transforms.CenterCrop(480),
                                              transforms.ToTensor()])
    else:
        image_transform = transforms.Compose([transforms.CenterCrop(224),
                                              transforms.ToTensor()])

    data_train = myDataset(data_root,
                           mode='train',
                           transform=image_transform,
                           types='raw',
                           shuffle=True,
                           large=args.large)

    # train_loader __getitem__ returns (data [1, 224, 224]), target [3, 224, 224])
    train_loader = torchdata.DataLoader(data_train,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4)

    data_val = myDataset(data_root,
                         mode='test',
                         transform=image_transform,
                         types='raw',
                         shuffle=True,
                         large=args.large)

    val_loader = torchdata.DataLoader(data_val,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=4)

    val_batch_size = val_loader.batch_size

    # Set up plotter, path, etc.
    global iteration, print_interval
    iteration = 0
    print_interval = 5

    # Define paths
    img_path = 'img/{}/'.format(model_name)
    model_path = 'model/{}/'.format(model_name)

    # Create the folders
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ############################################
    #              TRAINING LOOP
    ############################################
    # Start loop
    start_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch - 1))
        print('-' * 20)
        if epoch == 0:
            val_errG_epoch = validate(val_loader=val_loader, model_G=model_G,
                                      epoch=-1, global_step=global_step)
            # TensorboardX
            writer.add_scalar('val/val_errG_epoch', val_errG_epoch, epoch)

        # Train
        train_errG_epoch, global_step = train(train_loader=train_loader, model_G=model_G,
                                              optimizer_G=optimizer_G, epoch=epoch,
                                              iteration=iteration)
        # Validate
        val_errG_epoch = validate(val_loader=val_loader, model_G=model_G,
                                  epoch=epoch, global_step=global_step)

        # TensorboardX
        writer.add_scalar('train/train_errG_epoch', train_errG_epoch, epoch)
        writer.add_scalar('val/val_errG_epoch', val_errG_epoch, epoch)

        if args.save:
            print('Saving checkpoint.')
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_G.state_dict(),
                             'optimizer': optimizer_G.state_dict()},
                            filename=os.path.join(model_path, 'G_epoch_{}.pth.tar'.format(epoch)))


def train(train_loader, model_G, optimizer_G, epoch, iteration):
    """
    Train over one epoch
    Args:
        train_loader ():
        model_G ():
        optimizer_G ():
        epoch ():
        iteration ():

    Returns:

    """
    model_G.train()
    training_errors = []
    # Iterating for 1 epoch over all train_loader data
    for i, (z_image, target) in enumerate(train_loader):
        z_image, target = Variable(z_image), Variable(target)
        if args.use_cuda:
            z_image, target = z_image.cuda(), target.cuda()

        fake = model_G(z_image)  # G augments the z_image and adds fake colors

        error = criterion(fake, target)
        error.backward()  # Compute all gradients
        optimizer_G.step()  # Optimizer step

        training_errors.append(error.data[0])

        # TensorboardX
        global_step = iteration + len(train_loader) * epoch
        writer.add_scalar('train/errG', error.data[0], global_step)

        if iteration % print_interval == 0:
            print(
                'Training epoch {:03}: [{}/{}]: '
                'Loss: {:.2f}'.format(epoch, i, len(train_loader),
                                      error.data[0]))

        iteration += 1

    return np.mean(training_errors), global_step


def validate(val_loader, model_G, epoch, global_step):
    """
    Validate for one epoch
    Args:
        val_loader ():
        model_G ():
        epoch ():
        global_step ():

    Returns:

    """

    model_G.eval()
    validation_errors = []
    # Iterating for 1 epoch over all val_loader data
    for i, (z_image, target) in enumerate(val_loader):
        z_image, target = Variable(z_image), Variable(target)
        if args.use_cuda:
            z_image, target = z_image.cuda(), target.cuda()

        # validate with fake
        fake = model_G(z_image)

        error = criterion(fake, target)
        validation_errors.append(error.data[0])

        if i == 0:
            vis_result(z_image.data, target.data, fake.data, epoch, global_step)

        if i % 50 == 0:
            print('Validating epoch {}: [{}/{}]'.format(epoch, i, len(val_loader)))

    print('Validation epoch {}: error: {:.2f}'.format(epoch, np.mean(validation_errors)))

    return np.mean(validation_errors)


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
        writer.add_image('val/data', x, global_step)

        x = vutils.make_grid(target[:n_images, :, :, :],
                             nrow=2,
                             normalize=True,
                             scale_each=True)
        writer.add_image('val/target', x, global_step)

    x = vutils.make_grid(output[:n_images, :, :, :],
                         nrow=2,
                         normalize=True,
                         scale_each=True)
    writer.add_image('val/fake', x, global_step)

    img_list = []
    for i in range(min(32, val_batch_size)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0).cpu().numpy()
        raw = target[i].cpu().numpy()
        pred = output[i].cpu().numpy()

        raw_rgb = (np.transpose(raw, (1, 2, 0)).astype(np.float64) + 1) / 2.
        pred_rgb = (np.transpose(pred, (1, 2, 0)).astype(np.float64) + 1) / 2.

        grey = np.transpose(l, (1, 2, 0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)

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
    main()

    # Export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print('Closed the TensorboardX writer')
