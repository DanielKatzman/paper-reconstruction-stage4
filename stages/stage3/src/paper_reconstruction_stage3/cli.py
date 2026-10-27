"""File-in/file-out alpha entry point, independent of research experiments."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage 3 alpha: frozen observable evidence for a separate Stage 4')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='Authenticate the two packaged frozen models and policy; no inference')
    verify.add_argument('--models', type=Path, default=Path('models'))
    infer = commands.add_parser('infer', help='Read one page through the frozen Stage 1/2 file contract; write a Stage 4 bundle')
    for name in ('stage1', 'stage2', 'output'):
        infer.add_argument('--'+name, type=Path, required=True)
    infer.add_argument('--models', type=Path, default=Path('models'))
    infer.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    infer.add_argument('--puzzle-id', default='production')
    inspect = commands.add_parser('inspect', help='Validate/inspect a Stage 3-to-Stage 4 bundle without torch')
    inspect.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            from .alpha_package import load_package
            root, manifest = load_package(args.models)
            result = {'status': 'ready', 'software_version': __version__, 'models': [r['name'] for r in manifest['rankers']],
                      'models_directory': str(root), 'policy_sha256': manifest['manifest_sha256'],
                      'implementation_sha256': manifest['implementation_sha256'], 'training_performed': False,
                      'correction_scorer_enabled': False, 'demonstrated_reliable_piece_capacity': None}
        elif args.command == 'infer':
            from .alpha_inference import infer as run
            result = run(args.models, args.stage1, args.stage2, args.output, device=args.device, puzzle_id=args.puzzle_id,
                         progress=lambda message: print(message, file=sys.stderr, flush=True))
        else:
            from paper_reconstruction_contracts import load_bundle
            result = load_bundle(args.bundle, full=True).summary()
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except KeyboardInterrupt:
        print('Interrupted; committed caches are preserved. Repeat the same command to resume.', file=sys.stderr); return 130
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': type(error).__name__+': '+str(error)}), file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
