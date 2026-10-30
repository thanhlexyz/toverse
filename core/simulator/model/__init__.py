import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .semcom import SemComModel
import torch.nn as nn
import torch
import os


def create(args):
    path = os.path.join(args.checkpoint_dir, args.model)
    os.makedirs(path, exist_ok=True)
    if args.model in ['cifar10_semcom', 'ablation_cifar10_semcom']:
        from .cifar10 import Classifier
        from .vae.vae_small import VAE
        ae = VAE(args).to(args.device)
        classifier = Classifier(args).to(memory_format=torch.channels_last)
        semcom = SemComModel(ae, classifier).to(args.device)
        return semcom
    elif args.model in ['cifar10_pgm', 'ablation_cifar10_pgm']:
        from .gan import Generator, Discriminator
        from .cifar10 import Classifier
        from .vae.vae_small import VAE
        ae = VAE(args).to(args.device)
        classifier = Classifier(args).to(memory_format=torch.channels_last)
        semcom = SemComModel(ae, classifier).to(args.device)
        generator = Generator(args.n_latent, args.n_latent * 2).to(args.device)
        discriminator = Discriminator(args.n_latent, args.n_latent * 2).to(args.device)
        return semcom, generator, discriminator
    elif args.model in ['fashion_mnist_semcom', 'ablation_fashion_mnist_semcom']:
        from .vae.vae_small import VAE
        from .fc import FC
        classifier = FC().to(args.device)
        vae = VAE(args).to(args.device)
        semcom = SemComModel(vae, classifier)
        return semcom
    elif args.model in ['fashion_mnist_pgm', 'ablation_fashion_mnist_pgm']:
        from .gan import Generator, Discriminator
        from .vae.vae_small import VAE
        from .fc import FC
        vae = VAE(args).to(args.device)
        classifier = FC().to(args.device)
        semcom = SemComModel(vae, classifier)
        generator = Generator(args.n_latent, args.n_latent * 2). to(args.device)
        discriminator = Discriminator(args.n_latent, args.n_latent * 2). to(args.device)
        return semcom, generator, discriminator
    else:
        raise NotImplementedError(args.model)
