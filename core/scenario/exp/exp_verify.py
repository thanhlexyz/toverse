import os

from . import util

def exp_verify(args):
    ATTACKERS = []
    VERIFIERS = ['neuralsat']
    for experiment in ['fashion_mnist']:
        args.experiment = experiment
        config = util.load_exp_yaml(args)
        DATASET = config['DATASET']
        for attacker in ATTACKERS:
            cmd = f'python3 -W ignore main.py --dataset={DATASET} --scenario=attack --attacker={attacker} --experiment={experiment}'
            print(cmd)
            os.system(cmd)
        for verifier in VERIFIERS:
            cmd = f'python3 -W ignore main.py --dataset={DATASET} --scenario=verify --verifier={verifier} --experiment={experiment}'
            print(cmd)
            os.system(cmd)
