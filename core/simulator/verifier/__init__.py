from . import neuralsat, abcrown, e2e, split, e2e_mip, pgm

def verify(onnx, vnnlib, timeout, args):
    return eval(args.verifier).verify(onnx, vnnlib, timeout, args)


def attack(onnx, vnnlib, input_shape, timeout, args):
    return eval(args.verifier).Attacker(onnx, vnnlib, input_shape, args).attack(timeout=timeout*0.1)
