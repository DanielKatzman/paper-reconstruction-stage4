import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class ProcessDiagnosticsTests(unittest.TestCase):
    def test_fault_logging_configures_and_exits_without_a_watchdog(self):
        source=Path(__file__).resolve().parents[1]/'src'
        env={**os.environ,'PYTHONPATH':str(source)}
        code='from paper_reconstruction_stage4.process_diagnostics import configure; import json; print(json.dumps(configure()))'
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=env,timeout=15,check=False)
        self.assertEqual(result.returncode,0,result.stderr)
        status=json.loads(result.stdout)
        self.assertTrue(status['native_tracebacks'])
        self.assertFalse(status['timer_watchdog_installed'])
        if sys.platform=='win32':self.assertTrue(status['windows_error_ui_disabled'])


if __name__=='__main__':unittest.main()
