import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('monitor', Path(__file__).resolve().parents[1] / 'scripts/benchmark_android_monitor.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class AndroidMonitorTests(unittest.TestCase):
    def test_rejects_unsafe_identity(self):
        for pid, binary, model in [(0, '/bin/server', '/model'), (True, '/bin/server', '/model'),
                                   (1, '/bin/server', '/model'), (42, '/bin/server', '/x$(id)'),
                                   (42, '/bin/server', '/model\n'), (42, 'relative', '/model')]:
            with self.subTest(pid=pid, model=model):
                with self.assertRaises(ValueError):
                    monitor.render_monitor(pid, binary, model)
        self.assertNotIn('\0', monitor.render_monitor(42, '/bin/server', '/model'))

    def test_stop_after_successful_response_still_invalidates_cohort(self):
        monitor.validate_monitor_log('SAMPLE 123\n  temperature: 390\n')
        for text in ['', 'SAMPLE 123\nSAFETY_STOP TEMPERATURE_ABOVE_40C\n']:
            with self.subTest(text=text):
                with self.assertRaises(RuntimeError):
                    monitor.validate_monitor_log(text)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires procfs and POSIX signals')
    def test_sampled_thermal_stop_preserves_process_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mock = root / 'dumpsys'
            script = root / 'monitor.sh'
            for temperature, matches in [('390', True), ('401', True), ('invalid', True), ('401', False), ('prefix', False)]:
                with self.subTest(temperature=temperature, matches=matches):
                    mock.write_text('#!/bin/sh\necho "  temperature: ' + ('401' if temperature == 'prefix' else temperature) + '"\n')
                    mock.chmod(0o755)
                    should_stop = temperature != '390' and matches
                    child = subprocess.Popen([sys.executable, '-c',
                                              'import time; time.sleep(8)' if should_stop else 'import time; time.sleep(1)',
                                              '/owned/runtime', '/owned/model'])
                    observer = None
                    try:
                        script.write_text(monitor.render_monitor(child.pid, '/owned/runtime',
                                                                 '/owned/model' if matches else ('/owned/mod' if temperature == 'prefix' else '/different/model')))
                        with (root / 'output.log').open('w') as output:
                            observer = subprocess.Popen(['sh', str(script)],
                                                        env={**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH']},
                                                        stdout=output, stderr=output)
                            status = child.wait(timeout=5)
                            observer.wait(timeout=3)
                        self.assertEqual(status < 0, should_stop)
                        self.assertEqual('SAFETY_STOP' in (root / 'output.log').read_text(), temperature != '390')
                    finally:
                        for process in (observer, child):
                            if process is not None and process.poll() is None:
                                process.terminate()
                                process.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
