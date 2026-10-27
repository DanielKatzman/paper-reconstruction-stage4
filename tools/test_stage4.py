"""Run portable Stage 4 tests. Historical machine-bound orchestration tests are reported separately."""
import os
import subprocess
import sys
from common import ROOT,environment,python,read


if __name__=='__main__':
    if '--worker' not in sys.argv:
        result=subprocess.run([python('stage4'),'-B',str(ROOT/'tools/test_stage4.py'),'--worker'],
            cwd=ROOT/'stages/stage4',env=environment('stage4'))
        raise SystemExit(result.returncode)
    import unittest
    exclusions=read(ROOT/'provenance/historical_test_exclusions.json')['tests']
    suite=unittest.defaultTestLoader.discover('tests')
    def annotate(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite): annotate(item)
            elif item.id() in exclusions:
                method=getattr(type(item),item._testMethodName)
                setattr(type(item),item._testMethodName,unittest.skip(exclusions[item.id()])(method))
    annotate(suite)
    print(f'Historical archive-dependent tests explicitly marked: {len(exclusions)}',flush=True)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
