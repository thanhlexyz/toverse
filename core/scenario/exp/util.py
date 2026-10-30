import yaml
import os

def load_exp_yaml(args):
    path = os.path.join(args.experiment_dir, f'{args.experiment}.yaml')
    with open(path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config
