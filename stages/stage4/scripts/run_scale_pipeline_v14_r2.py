"""Original measured pipeline with the revision-2 post-seal report formatter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import scale_pipeline_v14 as pipeline


def main():
    # New requests bind this adapter and the revised formatter as dependencies.
    # Existing prediction requests retain their original serving commitments.
    pipeline.CONTROLLERS = [*pipeline.CONTROLLERS, 'scripts/run_scale_pipeline_v14_r2.py',
                            'scripts/diagnose_scale_pipeline_v14_r2.py']
    original = pipeline.execute
    def execute(command, *args, **kwargs):
        command = [pipeline.WORKSPACE / 'scripts/diagnose_scale_pipeline_v14_r2.py'
                   if str(item) == str(pipeline.WORKSPACE / 'scripts/diagnose_scale_pipeline_v14.py') else item for item in command]
        return original(command, *args, **kwargs)
    pipeline.execute = execute
    return pipeline.main()


if __name__ == '__main__':
    raise SystemExit(main())
