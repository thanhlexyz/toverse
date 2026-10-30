from torchvision.utils import make_grid
import torch.nn as nn
import torchvision
import simulator
import torch
import sys
import os

from timm.utils import NativeScaler, CheckpointSaver, AverageMeter
from timm.scheduler import create_scheduler_v2
from timm.loss import SoftTargetCrossEntropy
from timm.models import model_parameters
from tqdm import tqdm

class Trainer:

    def __init__(self, args):
        self.args = args
        self.train_loader, self.test_loader = simulator.dataloader.create(args)
        args.model = f'{args.experiment}_semcom'.replace('ablation_', '')
        self.semcom = simulator.model.create(args)
        self.monitor = simulator.Monitor(args)
        self.create_other()

    def create_other(self):
        args = self.args
        if args.dataset == 'fashion_mnist':
            self.criterion = nn.CrossEntropyLoss().to(args.device)
        else:
            self.criterion = SoftTargetCrossEntropy().to(args.device)
        self.loss_scaler = NativeScaler()
        self.optimizer = torch.optim.AdamW(self.semcom.parameters(), lr=args.lr, weight_decay=5e-2)
        self.scheduler, num_epochs = create_scheduler_v2(
            self.optimizer,
            num_epochs=args.max_epoch,
            min_lr=1e-5,
            cooldown_epochs=10,
            warmup_epochs=10,
            warmup_lr=1e-5,
        )
        self.amp_autocast = torch.amp.autocast(args.device)
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, f'{args.n_latent}_{args.n_channel}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.saver = CheckpointSaver(
            model=self.semcom,
            optimizer=self.optimizer,
            args=args,
            model_ema=None,
            amp_scaler=self.loss_scaler,
            checkpoint_dir=checkpoint_dir,
            recovery_dir=checkpoint_dir,
            decreasing=False,
            max_history=5
        )
        self.step = 0

    def train(self):
        args = self.args
        for epoch in range(args.max_epoch):
            info = self.get_info(epoch)
            info.update(self.train_epoch(epoch))
            info.update(self.test_epoch(epoch))
            info.update(self.save(epoch, info))
            self.monitor.step(info)
            self.monitor.export_csv()
            self.scheduler.step(epoch + 1, info['test_acc'])

    def train_epoch(self, epoch):
        args = self.args
        self.semcom.train()

        second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        running_loss = 0.0
        num_updates = epoch * len(self.train_loader)
        losses_m = AverageMeter()

        for batch_id, (X, y) in enumerate(self.train_loader):
            X = X.to(args.device)
            y = y.to(args.device)
            with self.amp_autocast:
                y_hat = self.semcom(X)
                loss  = self.criterion(y_hat, y)
            losses_m.update(loss.item(), X.size(0))
            self.optimizer.zero_grad()
            if self.loss_scaler is not None:
                self.loss_scaler(
                    loss,
                    self.optimizer,
                    clip_grad=0.1,
                    clip_mode='norm',
                    parameters=model_parameters(self.semcom, exclude_head='agc' in 'norm'),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                self.optimizer.step()

            torch.cuda.synchronize()
            num_updates += 1

            running_loss += loss.item() * X.size(0)

            if self.scheduler is not None:
                self.scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            self.step += 1
        info = {'train_loss': running_loss / len(self.train_loader.dataset)}
        return info

    def get_info(self, epoch):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        info = {'epoch': epoch, 'lr': lr}
        return info

    @torch.no_grad()
    def test_epoch(self, epoch):
        args = self.args
        self.semcom.eval()

        running_metric = 0.0
        for batch_id, (X, y) in enumerate(self.test_loader):
            X = X.to(args.device)
            y = y.to(args.device)
            y_pred = self.semcom(X).argmax(-1)
            batch_acc = y.eq(y_pred).sum() / len(y)

            running_metric += batch_acc.item() * X.size(0)
        info = {'test_acc': running_metric / len(self.test_loader.dataset)}
        return info

    def unnormalize(self, x):
        if self.args.dataset == 'fashion_mnist':
            mean = torch.tensor(simulator.dataloader.FASHION_MNIST_MEAN)[None, :, None, None].to(x.device)
            std = torch.tensor(simulator.dataloader.FASHION_MNIST_STD)[None, :, None, None].to(x.device)
        else:
            mean = torch.tensor(simulator.dataloader.CIFAR_MEAN)[None, :, None, None].to(x.device)
            std = torch.tensor(simulator.dataloader.CIFAR_STD)[None, :, None, None].to(x.device)
        return x * std + mean

    def save(self, epoch, info):
        best_acc, best_epoch = self.saver.save_checkpoint(epoch, metric=info['test_acc'])
        info = {'best_acc': best_acc, 'best_epoch': best_epoch}
        return info

    def load(self):
        args = self.args
        path = os.path.join(args.checkpoint_dir, args.model.replace('semcom', 'classifier'), f'model_best.pth.tar')
        if os.path.exists(path):
            state_dict = torch.load(path, weights_only=False)
            if 'state_dict' in state_dict:
                self.semcom.pragmatic.load_state_dict(state_dict['state_dict'])
            else:
                self.semcom.pragmatic.load_state_dict(state_dict)
            self.semcom.pragmatic.to(args.device)
            self.semcom.pragmatic.eval()
            for p in self.semcom.pragmatic.parameters():
                p.requires_grad = False
            print(f'[+] {args.trainer}_trainer loaded pretrain classifier')
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, f'{args.n_latent}_{args.n_channel}')
        path = os.path.join(checkpoint_dir, f'model_best.pth.tar')
        if os.path.exists(path):
            state_dict = torch.load(path, weights_only=False)
            if 'state_dict' in state_dict:
                self.semcom.load_state_dict(state_dict['state_dict'])
            else:
                self.semcom.load_state_dict(state_dict)
            self.semcom.to(args.device)
            self.semcom.eval()
            print(f'[+] {args.trainer}_trainer loaded pretrain semcom')
        else:
            path = os.path.join(args.checkpoint_dir, args.model.replace('semcom', 'vae'),
                                f'model_{args.n_latent}_{args.n_channel}.pth')
            if os.path.exists(path):
                self.semcom.vae.load_state_dict(torch.load(path))
                print(f'[+] {args.trainer}_trainer loaded pretrain vae')
