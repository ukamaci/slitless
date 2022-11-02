import torch
import sys, os, logging, time, glob
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import datetime

from slitless.data_loader import BasicDataset, OntheflyDataset
from slitless.networks.unet import UNet
from slitless.measure import nrmse, nmse_torch, combine_losses, cycle_loss
import numpy as np
import matplotlib.pyplot as plt
from slitless.evaluate import plot_recons, plot_val_stats, eval_snrlist
from slitless.plotting import barplot_group

def train_net(net,
            device,
            trainloader,
            otf,
            valloader,
            epochs,
            optimizer,
            criterion,
            path
):

    train_loss_over_epochs = []
    val_loss_over_epochs = []
    best_valloss = 1e6
    modnum = 5 if otf is not None else 1

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        # if onthefly dataloading is active, then read the parameters from otf
        # and create the next trainset 
        if otf is True:
            trainset = OntheflyDataset(data_dir=trainloader.dataset.data_dir, fold='train', dbsnr=trainloader.dataset.dbsnr, trpart=epoch%5+1)
            trainloader = DataLoader(trainset, batch_size=trainloader.batch_size, shuffle=True, num_workers=trainloader.num_workers)

        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs = data[0].to(device=device, dtype=torch.float)
            true_outputs = data[1].to(device=device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(truth=true_outputs, out=outputs, meas=inputs)
            # print('Loss Calculated')
            loss.backward()
            # print('Backward Propagated')
            optimizer.step()
            # print('Optimizer Updated')

            # print statistics
            running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
        running_loss/=len(trainloader)
        logging.info('[%d] Train loss: %.7f' % (epoch + 1, running_loss))

        if (epoch+1)%modnum==0:
            net.eval()
            running_valloss = 0.0
            for i, data in enumerate(valloader):
                # get the inputs
                inputs = data[0].to(device=device, dtype=torch.float)
                true_outputs = data[1].to(device=device, dtype=torch.float)

                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(truth=true_outputs, out=outputs, meas=inputs)

                    running_valloss += loss.item()

            running_valloss/=len(valloader)
            logging.info('Validation loss: %.7f' % (running_valloss))

            if running_valloss < best_valloss:
                best_valloss = running_valloss*1
                torch.save(net.state_dict(), f'../results/saved/{name}/best_model.pth')

            train_loss_over_epochs.append(running_loss)
            val_loss_over_epochs.append(running_valloss)

    return train_loss_over_epochs, val_loss_over_epochs, net

if __name__ == '__main__':
    # ---------
    NUM_FILT = 64
    numlayers = 4
    LR = 1e-3
    EPOCHS = 5
    BATCH_SIZE = 4
    BILINEAR = True
    ksizes = [(3,1), (3,1), (3,1), (3,1)]
    OPTIMIZER = 'ADAM'
    LOSS = 'MSE'
    # LOSS = 'NMSE'
    CYC_LOSS = False
    cyc_lam = 1
    CYC_ONLY = False
    LOSS = 'CYCLE_ONLY' if CYC_ONLY else LOSS
    OUTCH = 'all'
    out_channels = 3 if OUTCH=='all' else 1
    LOAD = True
    otf = True # on the fly trainset generation 
    loaded_model_path = '../results/saved/2022_10_14__22_24_44_NF_64_BS_4_LR_0.001_EP_30_KSIZE_(3, 1)_MSE_LOSS_ADAM_all/best_model.pth'
    dataset_path = glob.glob('../../data/datasets/dset6*')[0]
    dbsnr = 25

    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    name = f'{now}_NF_{NUM_FILT}_BS_{BATCH_SIZE}_LR_{LR}_EP_{EPOCHS}_KSIZE_{str(ksizes[0])}_{LOSS}_LOSS_{OPTIMIZER}_{OUTCH}_dbsnr_{dbsnr}'
    if (not CYC_ONLY) & CYC_LOSS:
        name += f'_CYC_LOSS_lam_{cyc_lam}'
    os.mkdir('../results/saved/'+name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [] # clean up the previous handlers to avoid problems
    fh = logging.FileHandler(f'../results/saved/{name}/output.log')
    formatter = logging.Formatter('%(asctime)s; %(message)s','%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # trainset = BasicDataset(data_dir = dataset_path, fold='train')
    # valset = BasicDataset(data_dir = dataset_path, fold='val')
    # testset = BasicDataset(data_dir = dataset_path, fold='test')
    # trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    # valloader = DataLoader(valset, batch_size=32, shuffle=True)
    # testloader = DataLoader(testset, batch_size=32, shuffle=True)

    trainset = OntheflyDataset(data_dir=dataset_path, fold='train', dbsnr=dbsnr)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valset = OntheflyDataset(data_dir=dataset_path, fold='val', dbsnr=dbsnr)
    valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=8)

    net = UNet(
        in_channels=3,
        out_channels=out_channels,
        numlayers=numlayers,
        outch_type=OUTCH,
        start_filters=NUM_FILT,
        bilinear=BILINEAR,
        ksizes=ksizes,
        residual=False).to(device)

    if LOAD:
        net.load_state_dict(torch.load(loaded_model_path))

    if OPTIMIZER=='ADAM':
        optimizer = optim.Adam(net.parameters(), lr=LR)
    elif OPTIMIZER=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if LOSS=='MSE':
        criterion = nn.MSELoss()
        losses_param = [nn.MSELoss()]
    elif LOSS=='L1':
        criterion = nn.L1Loss()
        losses_param = [nn.L1Loss()]
    elif LOSS=='NMSE':
        criterion = nmse_torch
        losses_param = [nmse_torch]
    if CYC_ONLY is True:
        losses_param = []

    losses_meas = [cycle_loss] if CYC_LOSS else []
    lam = [1,cyc_lam] if CYC_LOSS else [1]
    lam = [1] if CYC_ONLY else lam

    criterion = combine_losses(
        losses_param=losses_param,
        losses_meas=losses_meas,
        lam=lam, 
        outch_type=net.outch_type
    )

    training_summary = [
    '############## Network Parameters ############## \n',
    f'Number of starting filters = {NUM_FILT} \n',
    'Bilinear Interpolation for Upsampling (if False, use transposed ',
    f'convolution) = {BILINEAR} \n',
    f'Kernel Size = {ksizes} \n',
    f'Number of Layers = {numlayers} \n',
    f'Output Channels = {OUTCH} \n',
    '\n############## Optimization Parameters ############## \n',
    f'Optimizer = {OPTIMIZER} \n',
    f'Loss = {LOSS} \n',
    f'Cycle Loss = {CYC_LOSS} \n',
    f'Cycle Loss Lam = {cyc_lam} \n',
    f'Learning Rate = {LR} \n',
    f'Num Epochs = {EPOCHS} \n',
    f'Training Batch Size = {BATCH_SIZE} \n',
    '\n############## Data Parameters ############## \n',
    f'Num of Tranining Images = {len(trainset)*5 if otf else len(trainset)} \n',
    f'Num of Validation Images = {len(valset)} \n',
    f'Dataset Path = {dataset_path} \n',
    ]

    with open(f'../results/saved/{name}/summary.txt', 'w') as file:
        for line in training_summary:
            file.write(line)

    try:
        t0 = time.time()
        trainloss, valloss, net = train_net(net=net,
                  device=device,
                  trainloader=trainloader,
                  otf=otf,
                  valloader=valloader,
                  epochs=EPOCHS,
                  optimizer=optimizer,
                  criterion=criterion,
                  path=name
                  )
        train_time = datetime.timedelta(seconds=int(time.time() - t0))

    except KeyboardInterrupt:
        torch.save(net.state_dict(), f'../results/saved/{name}/INTERRUPTED.pth')
        sys.exit(0)

    torch.save(net.state_dict(), f'../results/saved/{name}/nf_{NUM_FILT}_LR_{LR}_EP_{EPOCHS}.pth')

    plt.figure()
    plt.semilogy(trainloss, label='Training Loss')
    plt.semilogy(valloss, label='Validation Loss')
    plt.title('Convergence Plot')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig(f'../results/saved/{name}/convergence_plot.png')
    plt.close()

    net.load_state_dict(torch.load(f'../results/saved/{name}/best_model.pth'))
    net.eval()

    testset = BasicDataset(data_dir=dataset_path, fold='test', dbsnr=dbsnr)
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

    savedir = f'../results/saved/{name}/'
    ssims, rmses, yvec, outvec = plot_val_stats(net, testloader, savedir)
    plot_recons(net, testloader, numim=32, savedir=savedir+'figures/')
    dbsnr_l = [15,25,35,None]
    ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, fold='test', 
    data_dir=dataset_path, net=net)
    barplot_group(ssims_l.mean(axis=1).swapaxes(0,1), 
        labels_gr=['int','vel','width'], labels_mem=[str(jj) for jj in dbsnr_l], 
        ylabel='SSIM', title='SSIM vs dBsnr', savedir=savedir+'snr_barplot.png')
    est_bias = np.mean(outvec - yvec, axis=1) 
    est_std = np.std(outvec - yvec, axis=1)
    np.save(savedir+'ssims_l.npy', ssims_l)
    np.save(savedir+'rmses_l.npy', rmses_l)

    os.mkdir(f'../results/saved/{name}/val_results')
    savedir = f'../results/saved/{name}/val_results/'
    _ = plot_val_stats(net, valloader, savedir)
    plot_recons(net, valloader, numim=32, savedir=savedir+'figures/')
    ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, fold='val', 
    data_dir=dataset_path, net=net)
    barplot_group(ssims_l.mean(axis=1).swapaxes(0,1), 
        labels_gr=['int','vel','width'], labels_mem=[str(jj) for jj in dbsnr_l], 
        ylabel='SSIM', title='SSIM vs dBsnr', savedir=savedir+'snr_barplot.png')
    np.save(savedir+'ssims_l.npy', ssims_l)
    np.save(savedir+'rmses_l.npy', rmses_l)

    os.mkdir(f'../results/saved/{name}/train_results')
    savedir = f'../results/saved/{name}/train_results/'
    _ = plot_val_stats(net, trainloader, savedir)
    plot_recons(net, trainloader, numim=32, savedir=savedir+'figures/')
    ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, fold='val', 
    data_dir=dataset_path, net=net)
    barplot_group(ssims_l.mean(axis=1).swapaxes(0,1), 
        labels_gr=['int','vel','width'], labels_mem=[str(jj) for jj in dbsnr_l], 
        ylabel='SSIM', title='SSIM vs dBsnr', savedir=savedir+'snr_barplot.png')
    np.save(savedir+'ssims_l.npy', ssims_l)
    np.save(savedir+'rmses_l.npy', rmses_l)

    training_summary += [
    '\n############## Results ############## \n',
    'Final Validation Loss: {:.7f} \n'.format(valloss[-1]),
    'Minimum Validation Loss: {:.7f} \n'.format(np.min(valloss)),
    'Final Training Loss: {:.7f} \n'.format(trainloss[-1]),
    'Minimum Training Loss: {:.7f} \n\n'.format(np.min(trainloss)),
    ]

    if net.outch_type == 'all':
        training_summary += [
            'Intensity SSIM Mean: {:.3f} \n'.format((ssims[:,0].mean())),
            'Intensity SSIM Std: {:.3f} \n'.format((ssims[:,0].std())),
            'Intensity RMSE Mean: {:.3f} \n'.format((rmses[:,0].mean())),
            'Intensity RMSE Std: {:.3f} \n'.format((rmses[:,0].std())),
            'Intensity Bias: {:.4f} \n'.format((est_bias[0])),
            'Intensity Error Std: {:.4f} \n\n'.format((est_std[0])),
            'Velocity SSIM Mean: {:.3f} \n'.format((ssims[:,1].mean())),
            'Velocity SSIM Std: {:.3f} \n'.format((ssims[:,1].std())),
            'Velocity RMSE Mean: {:.3f} \n'.format((rmses[:,1].mean())),
            'Velocity RMSE Std: {:.3f} \n'.format((rmses[:,1].std())),
            'Velocity Bias: {:.4f} \n'.format((est_bias[1])),
            'Velocity Error Std: {:.4f} \n\n'.format((est_std[1])),
            'Linewidth SSIM Mean: {:.3f} \n'.format((ssims[:,2].mean())),
            'Linewidth SSIM Std: {:.3f} \n'.format((ssims[:,2].std())),
            'Linewidth RMSE Mean: {:.3f} \n'.format((rmses[:,2].mean())),
            'Linewidth RMSE Std: {:.3f} \n'.format((rmses[:,2].std())),
            'Linewidth Bias: {:.4f} \n'.format((est_bias[2])),
            'Linewidth Error Std: {:.4f} \n\n'.format((est_std[2])),
        ]
    elif net.outch_type == 'int':
        training_summary += [
            'Intensity SSIM Mean: {:.3f} \n'.format((ssims.mean())),
            'Intensity SSIM Std: {:.3f} \n'.format((ssims.std())),
            'Intensity RMSE Mean: {:.3f} \n'.format((rmses.mean())),
            'Intensity RMSE Std: {:.3f} \n'.format((rmses.std())),
            'Intensity Bias: {:.4f} \n'.format((est_bias[0])),
            'Intensity Error Std: {:.4f} \n\n'.format((est_std[0])),
        ]
    elif net.outch_type == 'vel':
        training_summary += [
            'Velocity SSIM Mean: {:.3f} \n'.format((ssims.mean())),
            'Velocity SSIM Std: {:.3f} \n'.format((ssims.std())),
            'Velocity RMSE Mean: {:.3f} \n'.format((rmses.mean())),
            'Velocity RMSE Std: {:.3f} \n'.format((rmses.std())),
            'Velocity Bias: {:.4f} \n'.format((est_bias[0])),
            'Velocity Error Std: {:.4f} \n\n'.format((est_std[0])),
        ]
    elif net.outch_type == 'width':
        training_summary += [
            'Linewidth SSIM Mean: {:.3f} \n'.format((ssims.mean())),
            'Linewidth SSIM Std: {:.3f} \n'.format((ssims.std())),
            'Linewidth RMSE Mean: {:.3f} \n'.format((rmses.mean())),
            'Linewidth RMSE Std: {:.3f} \n'.format((rmses.std())),
            'Linewidth Bias: {:.4f} \n'.format((est_bias[0])),
            'Linewidth Error Std: {:.4f} \n\n'.format((est_std[0])),
        ]

    training_summary += [
        f'Training Time: {str(train_time)} \n',
        '\n############## Notes ############## \n',
        'Training with on onthefly imagenet with ksize=(3,1), dbsnr=25.\n',
        '\n############## Comments ############## \n'
    ]

    with open(f'../results/saved/{name}/summary.txt', 'w') as file:
        for line in training_summary:
            file.write(line)