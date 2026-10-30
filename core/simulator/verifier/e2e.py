import os

def verify(onnx, vnnlib, timeout, args):
    os.chdir('neuralsat/src')
    onnx1 = os.path.abspath(os.path.join(args.onnx_dir, '..', onnx))
    vnnlib1 = os.path.abspath(os.path.join(args.vnnlib_dir, '..', vnnlib))
    assert os.path.exists(onnx1)
    assert os.path.exists(vnnlib1)
    log1 = vnnlib1.replace('.vnnlib', '.log').replace(args.vnnlib_dir, os.path.join(args.log_dir, 'e2e'))
    os.makedirs(os.path.dirname(log1), exist_ok=True)
    result_path = log1.replace('.log', '.res')

    sat = unsat = timeout = 0
    if os.path.exists(result_path):
        status, runtime = open(result_path).read().strip().split('\n')[0].split(',')
        if status == 'sat':
            sat = 1
        elif status == 'unsat':
            unsat = 1
        elif status == 'timeout':
            timeout = 1
        else:
            NotImplementedError
        os.chdir(args.core_dir)
        return sat, unsat, timeout, float(runtime)

    assert 'e2e' in onnx1
    assert 'e2e' in vnnlib1
    cmd  = f'python3 -W ignore main.py --disable_restart --disable_stabilize --verbosity=2 --export_cex --export_runtime'
    cmd += f' --net={onnx1} --spec={vnnlib1}'
    cmd += f' --result_file={result_path} --timeout={timeout}'
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
