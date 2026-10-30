import argparse
import pickle
import torch
import time
import os

from helper.network.read_onnx import parse_onnx
from helper.network.read_pth import parse_pth

from helper.spec.objective import parse_vnnlib

from helper.misc.logger import logger, LOGGER_LEVEL
from helper.misc.export import get_adv_string

from attacker.cpgd.cpgd import CPGDAttacker

from verifier.verifier import ReturnStatus
from verifier.verifier import Verifier

from setting import Settings


if __name__ == '__main__':
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                        help="Input shape of network, e.g., --input_shape 1 3 32 32")
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                        help="Output shape of network, e.g., --output_shape 1 10")
    parser.add_argument('--batch', type=int, default=1000,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2,
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--export_cex', action='store_true',
                        help="enable exporting counter-example to result file.")
    parser.add_argument('--disable_attack', action='store_false',
                        help="disable attack.")
    parser.add_argument('--disable_restart', action='store_false',
                        help="disable RESTART heuristic.")
    parser.add_argument('--disable_stabilize', action='store_false',
                        help="disable STABILIZE heuristic.")
    parser.add_argument('--force_split', type=str, choices=['input', 'hidden'],
                        help="select SPLITTING strategy.")
    parser.add_argument('--reasoning_file', type=str, required=False,
                        help="file to save reasoning steps.")
    parser.add_argument('--setting_file', type=str, required=False,
                        help="file to load specific settings.")
    parser.add_argument('--test', action='store_true',
                        help="test on small example with special settings.")
    parser.add_argument('--export_runtime', action='store_true', required=False,
                        help="output runtime.")
    parser.add_argument('--reference_bounds', type=str, required=False,
                        help="path to external reference bounds file.")
    parser.add_argument('--reference_layer_name', type=str, required=False,
                        help="layer name of reference bounds.")
    parser.add_argument('--noise_generator', type=str, required=False,
                        help="path to noise generator file.")
    parser.add_argument('--noise_constraint', type=str, required=False,
                        help="path to noise constraint file.")
    parser.add_argument('--cpgd_timeout', type=float, default=2.0,
                        help="timeout in seconds for CPGD attack.")
    

    args = parser.parse_args()
    Settings.setup(args)
    print(Settings)

    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'

    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])

    # network
    if args.net.endswith('.onnx'):
        model, input_shape, output_shape = parse_onnx(args.net, args.input_shape, args.output_shape)
    elif args.net.endswith('.pth'):
        model, input_shape, output_shape = parse_pth(args.net, args.input_shape, args.output_shape)
    else:
        raise NotImplementedError('Unsupported network type')

    model.to(args.device)

    if args.verbosity:
        print(model)
    logger.info(f'[!] Input shape: {input_shape}')
    logger.info(f'[!] Output shape: {output_shape}')

    # specification
    objectives = parse_vnnlib(args.spec, input_shape)
    
    status = None
    if args.noise_generator is not None and args.noise_constraint is not None:
        noise_generator, _, _ = parse_onnx(args.noise_generator)
        noise_constraint = torch.load(args.noise_constraint, weights_only=False)
        print(f'[+] noise_constraint: {noise_constraint}')
        cpgd_attacker = CPGDAttacker(
            net=model, 
            objective=objectives, 
            input_shape=input_shape, 
            device=args.device, 
            noise_generator=noise_generator, 
            noise_constraint=noise_constraint
        )
        is_attacked, attack_images = cpgd_attacker.run(timeout=args.cpgd_timeout)
        if is_attacked:
            status = ReturnStatus.SAT

    # verifier
    verifier = Verifier(
        net=model,
        input_shape=input_shape,
        batch=args.batch,
        device=args.device,
    )


    if args.reference_bounds is not None:
        reference_bounds = torch.load(args.reference_bounds)
        print(f'[+] reference_bounds: {reference_bounds}')
        # True reference bounds
        verifier.external_reference_bounds = {
            args.reference_layer_name: [
                torch.tensor(reference_bounds['n_lb']).to(args.device)[None],
                torch.tensor(reference_bounds['n_ub']).to(args.device)[None],
            ]
        }

        # # TODO: remove fake reference bounds
        # verifier.external_reference_bounds = {
        #     args.reference_layer_name: [
        #         torch.zeros(1, 32).to(args.device) - 0.6,
        #         torch.zeros(1, 32).to(args.device) + 0.6,
        #     ]
        # }


    # verify
    timeout = args.timeout - (time.time() - START_TIME)
    if not status:
        status = verifier.verify(objectives, timeout=timeout, force_split=args.force_split)
    runtime = time.time() - START_TIME

    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    if verifier.adv is not None:
        logger.info(f'adv (first 5): {verifier.adv.flatten()[:5].detach().cpu()}')
        logger.debug(f'output: {verifier.net(verifier.adv).flatten().detach().cpu()}')

    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            if args.export_runtime:
                print(f'{status},{runtime:.04f}', file=fp)
            else:
                print(status, file=fp)
            if (verifier.adv is not None) and args.export_cex:
                print(get_adv_string(inputs=verifier.adv, net_path=args.net), file=fp)

    if args.reasoning_file and Settings.use_save_reasoning_step:
        if hasattr(verifier, 'domains_list') and not isinstance(verifier.domains_list, list):
            verifier.domains_list.reasoning_domains.export(args.reasoning_file)
        else:
            print(f'[!] Does not have any reasoning step')

    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')

    print(f'{status},{runtime:.04f}')
