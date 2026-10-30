import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from timm.utils import NativeScaler, CheckpointSaver
from .semcom import Trainer as SemComTrainer

import simulator

class Trainer(SemComTrainer):

    def __init__(self, args):
        self.args = args
        self.data, _, self.test_loader = simulator.dataloader.create(args)
        args.model = f'{args.experiment}_semcom'.replace('ablation_', '')
        self.semcom = simulator.model.create(args)
        self.monitor = simulator.Monitor(args)
        self.create_other()

    @torch.no_grad()
    def _init_weights(self):
        enc    = self.semcom.vae.encoder
        cls    = self.semcom.pragmatic
        groups = [enc.enc1, cls.cls1, cls.cls2, cls.cls3]
        for grp in groups:
            dirac = torch.nn.init.dirac_(torch.empty_like(grp.conv1.weight))
            s, m  = torch.std_mean(grp.conv1.weight)
            grp.conv1.weight.data.add_(dirac)
            sp, mp = torch.std_mean(grp.conv1.weight)
            grp.conv1.weight.data.sub_(mp).div_(sp).mul_(s).add_(m)
            torch.nn.init.dirac_(grp.conv2.weight)
            grp.pool.weight.fill_(1.0 / grp.pool.kernel_size[0] / grp.pool.kernel_size[1])

    def create_other(self):
        args = self.args
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=0.2).to(args.device)
        self.optimizer  = torch.optim.Adam(self.semcom.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_epoch
        )
        self.loss_scaler  = NativeScaler()
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
            recovery_dir=checkpoint_dir, decreasing=False, max_history=5
        )
        self.step = 0

    @staticmethod
    @torch.no_grad()
    def _rand_crop(imgs, pad=2):
        imgs = F.pad(imgs, (pad,) * 4, 'reflect')
        B, C, H, W = imgs.shape
        out_h, out_w = H - 2 * pad, W - 2 * pad
        top  = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)
        left = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)
        rows = top[:, None]  + torch.arange(out_h, device=imgs.device)
        cols = left[:, None] + torch.arange(out_w, device=imgs.device)
        b    = torch.arange(B, device=imgs.device)
        return imgs[b[:, None, None, None],
                    torch.arange(C, device=imgs.device)[None, :, None, None],
                    rows[:, None, :, None],
                    cols[:, None, None, :]]

    @staticmethod
    @torch.no_grad()
    def _rand_flip(imgs, p=0.5):
        flip = torch.rand(imgs.shape[0], device=imgs.device) < p
        return torch.where(flip.view(-1, 1, 1, 1), torch.flip(imgs, dims=(-1,)), imgs)

    @torch.no_grad()
    def batches(self, split, bsz, cm_sz=0):
        device      = self.args.device
        imgs        = self.data[split]['images']
        labels      = self.data[split]['labels']
        idx         = torch.randperm(len(imgs))
        use_cutmix  = split == 'train' and cm_sz > 0
        num_classes = getattr(self.args, 'num_classes', 10)

        for i in range(len(imgs) // bsz):
            sl         = idx[i * bsz:(i + 1) * bsz]
            batch_imgs = imgs[sl].float().to(device, non_blocking=True)
            batch_lbls = labels[sl].to(device, non_blocking=True)

            batch_imgs = batch_imgs.to(memory_format=torch.channels_last)

            if split == 'train':
                batch_imgs = self._rand_crop(batch_imgs, pad=2)
                batch_imgs = self._rand_flip(batch_imgs)

            if use_cutmix:
                B, _, H, W = batch_imgs.shape
                ev   = int(cm_sz % 2 == 0)
                cy   = torch.randint(cm_sz // 2 - ev, H - cm_sz // 2 - ev, (B,), device=device)
                cx   = torch.randint(cm_sz // 2 - ev, W - cm_sz // 2 - ev, (B,), device=device)
                y    = torch.arange(H, device=device).view(1, 1, H, 1) - cy.view(-1, 1, 1, 1)
                x    = torch.arange(W, device=device).view(1, 1, 1, W) - cx.view(-1, 1, 1, 1)
                mask = (y >= -cm_sz // 2 + ev) & (y <= cm_sz // 2) & \
                       (x >= -cm_sz // 2 + ev) & (x <= cm_sz // 2)
                perm       = torch.randperm(B, device=device)
                frac       = cm_sz ** 2 / (H * W)
                tgts_soft  = F.one_hot(batch_lbls, num_classes).float()
                batch_imgs = torch.where(mask, batch_imgs[perm], batch_imgs)
                batch_tgts = frac * tgts_soft[perm] + (1 - frac) * tgts_soft
            else:
                batch_tgts = batch_lbls

            yield batch_imgs, batch_tgts

    def train(self):
        args = self.args
        for epoch in range(args.max_epoch):
            info = self.get_info(epoch)
            info.update(self.train_epoch(epoch))
            info.update(self.test_epoch(epoch))
            info.update(self.save(epoch, info))
            self.monitor.step(info)
            self.monitor.export_csv()
            self.scheduler.step()

    @torch.no_grad()
    def test_epoch(self, epoch):
        args = self.args
        self.semcom.eval()
        correct = 0
        for X, y in self.test_loader:
            X = X.to(args.device)
            y = y.to(args.device)
            logits      = self.semcom(X)
            logits_flip = self.semcom(torch.flip(X, dims=(-1,)))
            y_pred = (0.5 * logits + 0.5 * logits_flip).argmax(-1)
            correct += y.eq(y_pred).sum().item()
        return {'test_acc': correct / len(self.test_loader.dataset)}

    def train_epoch(self, epoch):
        args = self.args
        self.semcom.train()

        recon_w   = getattr(args, 'recon_w', 0.1)
        cutmix_sz = getattr(args, 'cutmix_sz', 3)
        cutmix_ep = getattr(args, 'cutmix_ep', 6)
        cm = cutmix_sz if epoch >= args.max_epoch - cutmix_ep else 0

        second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        tr_loss = tr_acc = n = 0.

        for imgs, tgts in self.batches('train', args.batch_size, cm):
            with self.amp_autocast:
                recon  = self.semcom.vae(imgs)
                logits = self.semcom.pragmatic(recon)
                loss   = self.criterion(logits, tgts) + recon_w * F.mse_loss(recon, imgs)

            self.optimizer.zero_grad(set_to_none=True)
            self.loss_scaler(loss, self.optimizer, create_graph=second_order)

            tr_loss += loss.item()
            acc_tgts = tgts.argmax(-1) if tgts.dim() > 1 else tgts
            tr_acc  += (logits.detach().argmax(-1) == acc_tgts).float().mean().item()
            n += 1

        return {'train_loss': tr_loss / n, 'train_acc': tr_acc / n}
