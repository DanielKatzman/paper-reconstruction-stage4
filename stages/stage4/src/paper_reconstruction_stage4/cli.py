import argparse
import json
from pathlib import Path
import sys
import time
from paper_reconstruction_contracts import ContractError
from jsonschema import ValidationError
from .config import SolverConfig
from .inputs import load_problem
from .outputs import atomic_json, build_result, output_session, publish, read_result, request_identity
from .solver import solve


def peak_memory():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        data = Counters()
        data.cb = ctypes.sizeof(data)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        query.restype = wintypes.BOOL
        if query(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
            return int(data.PeakWorkingSetSize)
    else:
        import resource
        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(maximum if sys.platform == "darwin" else maximum * 1024)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent Stage 4 paper assembly")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "solve", "assess"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--config", type=Path)
        command.add_argument('--constraint-profile', type=Path)
        command.add_argument('--capture-profile', type=Path)
        if name == 'assess':
            command.add_argument('--result', type=Path, required=True)
            command.add_argument('--output', type=Path, required=True)
        if name == "solve":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--initial-result", type=Path, help="Verified same-bundle serving primary, used only by joint solver")
            command.add_argument("--resume", action="store_true", help="Verify a completed run, or replay matching unpublished work")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            value = read_result(args.result)
            print(json.dumps({"status": value["status"], "solver": value["solver"], "run_id": value["run_id"],
                              "pieces": value["input"]["counts"]["pieces"], "components": value["layouts"][0]["components"],
                              "selected_joins": len(value["layouts"][0]["selected_joins"]), "confidence_calibrated": False}, indent=2))
            return 0
        config = SolverConfig.from_dict(json.loads(args.config.read_text())) if args.config else SolverConfig().validate()
        start = time.perf_counter()
        if args.command == "solve":
            bundle, output = args.bundle.resolve(), args.output.resolve()
            if bundle.is_file():
                bundle = bundle.parent
            if output.is_relative_to(bundle) or bundle.is_relative_to(output):
                raise ValueError("Stage 4 output must be separate from the immutable input tree")
        problem = load_problem(args.bundle, config)
        from .constraints import bind_profile
        bind_profile(problem, config, args.constraint_profile, args.capture_profile)
        if args.command == 'assess':
            from .assessment import write_package
            receipt = write_package(problem, config, args.result, args.output)
            print(json.dumps({'status': 'assessment_sealed', 'output': str(args.output.resolve()),
                              'manifest_sha256': receipt['manifest_sha256']}))
            return 0
        if args.command == "validate":
            print(json.dumps({"status": "valid_evidence", "puzzle_id": problem.manifest["puzzle_id"],
                              "counts": problem.manifest["counts"], "load": problem.input_summary,
                              "global_assembly_performed": False}, indent=2))
            return 0
        if config.solver == "joint_beam_v3":
            problem.deadline = start+config.joint_wall_seconds
        if args.initial_result:
            if config.solver != "joint_beam_v3":
                raise ValueError("--initial-result requires the joint solver")
            from .seeds import attach_initial_result
            attach_initial_result(problem, args.initial_result)
        request = request_identity(problem, config)
        with output_session(args.output, request, args.resume) as work:
            if work is None:
                if problem.constraint_profile:
                    from .assessment import write_package
                    write_package(problem, config, args.output)
                print(json.dumps({"status": "already_published", "run_id": request["run_id"]}))
                return 0
            result = solve(problem, config)
            value = build_result(problem, config, result, request, peak_memory())
            value["resources"]["elapsed_before_publication_seconds"] = time.perf_counter() - start
            publish(args.output, work, value, problem, result.states, resource_probe=peak_memory)
            if problem.constraint_profile:
                from .assessment import write_package
                write_package(problem, config, args.output)
            print(json.dumps({"status": value["status"], "output": str(args.output.resolve()),
                              "pieces": len(problem.pieces), "components": result.states[0].components(),
                              "selected_joins": len(result.states[0].selected), "termination": result.termination,
                              "accuracy_evaluated": False}, indent=2))
            return 130 if result.termination == "interrupted" else 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted_before_publication"}), file=sys.stderr)
        return 130
    except (ContractError, ValueError, OSError, KeyError, ValidationError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        return 2
