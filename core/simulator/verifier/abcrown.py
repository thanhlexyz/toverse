import os

def verify(onnx, vnnlib, timeout, args):
    os.chdir('abcrown/complete_verifier')
    onnx1 = os.path.abspath(os.path.join(args.onnx_dir, '..', onnx))
    vnnlib1 = os.path.abspath(os.path.join(args.vnnlib_dir, '..', vnnlib))
    assert os.path.exists(onnx1)
    assert os.path.exists(vnnlib1)
    log1 = vnnlib1.replace('.vnnlib', '.log').replace(args.vnnlib_dir, os.path.join(args.log_dir, 'crown'))
    os.makedirs(os.path.dirname(log1), exist_ok=True)
    result_path = log1.replace('.log', '.res')
    os.system(f'rm -rf "{result_path}"')
    assert 'e2e' in onnx1
    assert 'e2e' in vnnlib1
    cmd  = f'python3 abcrown.py --config exp_configs/vnncomp21/cifar2020_2_255.yaml'
    cmd += f' --onnx_path {onnx1}'
    cmd += f' --vnnlib_path {vnnlib1}'
    cmd += f' --results_file {result_path}'
    cmd += f' --timeout {timeout}'
    cmd += f' > {log1}'
    if args.verbose:
        print(cmd)
    os.system(cmd)
    sat = unsat = timeout = 0
    if os.path.exists(result_path):
        status = open(result_path).read().strip().split(',')[0]
        if status == 'sat':
            sat = 1
        elif status == 'unsat':
            unsat = 1
        elif status == 'timeout':
            timeout = 1
        else:
            NotImplementedError
    else:
        timeout = 1
    os.chdir(args.core_dir)
    return sat, unsat, timeout, None
