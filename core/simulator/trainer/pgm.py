import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import simulator
import torch
import sys
import os

from tqdm import tqdm

class Trainer:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'fashion_mnist':
            self.train_loader, self.test_loader = simulator.dataloader.create(args)
        else:
            _, self.train_loader, self.test_loader = simulator.dataloader.create(args)
        args.model = f'{args.experiment}_pgm'.replace('ablation_', '')
        self.semcom, self.generator, self.discriminator = \
                simulator.model.create(args)
        self.monitor = simulator.Monitor(args)
        self.create_other()

    @torch.no_grad()
    def get_latent_feature_stats(self):
        args = self.args
        Z = []
        for batch_id, (X, y) in enumerate(self.train_loader):
            X = X.to(args.device)
            z = self.semcom.vae.encoder(X)
            Z.append(z.cpu())
        Z     = torch.cat(Z).to(args.device)
        power = (Z ** 2).sum(dim=1)
        z2_ub = (power.mean() + 3 * power.std()).item()
        z_mu  = Z.mean(dim=0)
        z_std = Z.std(dim=0)
        return z_mu, z_std, z2_ub

    def create_other(self):
        args = self.args
        self.optimizer_g = optim.Adam(self.generator.parameters(),
                                      lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(),
                                      lr=0.0001, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch):
        args = self.args
        pbar = tqdm(self.train_loader, desc='[Training]', file=sys.stdout, disable=True)
        device = args.device
        G_losses, D_losses = [], []
        G_power_losses = []
        pns = []
        self.discriminator.train()
        self.generator.train()
        z_mu, z_std, z2_ub = self.get_latent_feature_stats()
        max_power = 10 ** (args.adv_snr_threshold / 10) * z2_ub

        for batch_id, (X, y) in enumerate(pbar):
            X = X.to(args.device)
            y = y.to(args.device)
            bs = X.shape[0]
            embedding = self.semcom.vae.encoder(X)
            latent_shape = embedding.shape
            embedding = embedding.view(bs, -1)
            n_latent = embedding.shape[1]
            z  = torch.rand(bs, n_latent).to(device) * 2 - 1
            z1 = torch.normal(z_mu[None].expand(bs, -1),
                              z_std[None].expand(bs, -1)).to(device)
            self.optimizer_d.zero_grad()
            real_labels = torch.ones(bs, 1).to(device)
            real_output = self.discriminator(z1)
            real_loss = self.criterion(real_output, real_labels)
            fake_labels = torch.zeros(bs, 1).to(device)
            fake_data = self.generator(z)
            fake_output = self.discriminator(fake_data.detach())
            fake_loss = self.criterion(fake_output, fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_d.step()
            if batch_id % 2 == 0:
                self.optimizer_g.zero_grad()
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)
                g_loss_adv = self.criterion(fake_output, real_labels)
                generated_mean = torch.mean(fake_data, dim=0)
                generated_var  = torch.var(fake_data, dim=0)
                target_mean    = z_mu.detach()
                target_var     = (z_std ** 2).detach()
                regularizer_R  = F.mse_loss(generated_mean, target_mean) + F.mse_loss(generated_var, target_var)
                perturbed_embedding = (embedding + fake_data).view(latent_shape)
                reconstructed_X = self.semcom.vae.decoder(perturbed_embedding)


                g_loss = g_loss_adv + 0.01 * regularizer_R

                g_power_loss = torch.relu((fake_data ** 2).sum(dim=1) - max_power).mean()
                pns.append((fake_data ** 2).sum(dim=1).mean().item())
                g_loss += g_power_loss

                g_loss.backward()
                self.optimizer_g.step()
            else:
                g_loss = torch.tensor(0.0)
                g_power_loss = torch.tensor(0.0)
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            G_power_losses.append(g_power_loss.item())

        with torch.no_grad():
            d_prob_fake = torch.sigmoid(fake_output).mean().item()
            d_prob_real = torch.sigmoid(real_output).mean().item()
        info = {
            'g_loss': np.mean(G_losses),
            'g_power_loss': np.mean(G_power_losses),
            'd_loss': np.mean(D_losses),
            'd_prob_fake': d_prob_fake,
            'd_prob_real': d_prob_real,
            'adv_noise': np.mean(pns),
            'max_noise_power': max_power,
        }
        return info

    def train(self):
        args = self.args
        best_acc = 100.0
        best_adv_noise = 0.0
        for epoch in range(args.max_epoch):
            info = self.get_info(epoch)
            info.update(self.train_epoch(epoch))
            info.update(self.test_epoch(epoch))
            self.monitor.step(info)
            self.monitor.export_csv()
            if info['pgm_acc'] < best_acc and info['adv_noise'] > best_adv_noise:
                best_acc = info['pgm_acc']
                best_adv_noise = float(info['adv_noise'])
                self.save()
                print(f'[!] Saved: {epoch=} {best_acc=:0.4f}, {best_adv_noise=:0.4f}')

    @torch.no_grad()
    def test_epoch(self, epoch):
        args = self.args
        device = args.device
        self.generator.eval()

        running_metric = 0.0
        running_metric_fake = 0.0
        pbar = tqdm(self.test_loader, desc='[Testing]', file=sys.stdout, disable=True)
        for __cached__, (X, y) in enumerate(pbar):
            X = X.to(args.device)
            y = y.to(args.device)

            embedding = self.semcom.vae.encoder(X)
            latent_shape = embedding.shape
            embedding = embedding.view(X.shape[0], -1)
            n_latent = embedding.shape[1]
            z  = torch.rand(X.shape[0], n_latent).to(device)

            fake_data = self.generator(z)
            perturbed_embedding = (embedding + fake_data).view(latent_shape)
            reconstructed_X = self.semcom.vae.decoder(perturbed_embedding)
            y_pred = self.semcom.pragmatic(reconstructed_X).argmax(-1)
            batch_acc = y.eq(y_pred).sum() / len(y)
            running_metric_fake += batch_acc.item() * X.size(0)

            constructed_X = self.semcom.vae.decoder(embedding.view(latent_shape))
            y_pred = self.semcom.pragmatic(constructed_X).argmax(-1)
            batch_acc = y.eq(y_pred).sum() / len(y)
            running_metric += batch_acc.item() * X.size(0)


        running_metric /= len(self.test_loader.dataset)
        running_metric_fake /= len(self.test_loader.dataset)
        info = {
                'semcom_acc': running_metric,
                'pgm_acc': running_metric_fake,
        }
        return info

    def get_info(self, epoch):
        info = {'epoch': epoch}
        return info
    def unnormalize(self, x):
        if self.args.dataset == 'fashion_mnist':
            mean = torch.tensor(simulator.dataloader.FASHION_MNIST_MEAN)[None, :, None, None].to(x.device)
            std = torch.tensor(simulator.dataloader.FASHION_MNIST_STD)[None, :, None, None].to(x.device)
        else:
            mean = torch.tensor(simulator.dataloader.CIFAR_MEAN)[None, :, None, None].to(x.device)
            std = torch.tensor(simulator.dataloader.CIFAR_STD)[None, :, None, None].to(x.device)
        return x * std + mean

    def load(self):
        args = self.args
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.model.replace('pgm', 'semcom'), f'{args.n_latent}_{args.n_channel}')
        path = os.path.join(checkpoint_dir, f'model_best.pth.tar')
        state_dict = torch.load(path, weights_only=False)
        if 'state_dict' in state_dict:
            self.semcom.load_state_dict(state_dict['state_dict'])
        else:
            self.semcom.load_state_dict(state_dict)
        self.semcom.to(args.device)
        self.semcom.eval()
        for p in self.semcom.parameters():
            p.requires_grad = False
        print(f'[+] {args.trainer}_trainer loaded pretrain semcom')

        path = os.path.join(args.checkpoint_dir, args.model, f'generator_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}.pth')
        if os.path.exists(path):
            state_dict = torch.load(path)
            self.generator.load_state_dict(state_dict)
            print(f'[+] {args.trainer}_trainer loaded adversarial generator')
        path = os.path.join(args.checkpoint_dir, args.model, f'discriminator_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}.pth')
        if os.path.exists(path):
            state_dict = torch.load(path)
            self.discriminator.load_state_dict(state_dict)
            print(f'[+] {args.trainer}_trainer loaded adversarial discriminator')

    def save(self):
        args = self.args
        path = os.path.join(args.checkpoint_dir, args.model, f'generator_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}.pth')
        torch.save(self.generator.state_dict(), path)
        path = os.path.join(args.checkpoint_dir, args.model, f'discriminator_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}.pth')
        torch.save(self.discriminator.state_dict(), path)
