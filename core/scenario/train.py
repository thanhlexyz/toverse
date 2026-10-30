import simulator

def train(args):
    trainer = simulator.trainer.create(args)
    trainer.load()
    trainer.train()
