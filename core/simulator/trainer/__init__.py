from . import semcom, pgm, semcom_cifar10

def create(args):
    return eval(args.trainer).Trainer(args)
