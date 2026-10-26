"""Check the task launcher after moving the implementation into packages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import WORKFLOWS


class WorkflowCommands(unittest.TestCase):
    def command(self, arguments, cwd):
        return subprocess.run(
            [sys.executable, str(ROOT / 'run.py'), *arguments],
            cwd=cwd, env={**os.environ, 'PYTHONUTF8': '1'},
            capture_output=True, text=True, encoding='utf-8', timeout=30,
        )

    def test_all_task_commands_open_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for task, (_, actions) in WORKFLOWS.items():
                for action in actions:
                    if action == 'listen' and os.name != 'nt':
                        continue  # The replay tool uses Windows audio APIs.
                    with self.subTest(task=task, action=action):
                        result = self.command([task, action, '--help'], directory)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(f'{task} {action}', result.stdout)
                        self.assertIn('usage:', result.stdout)

    def test_validation_exit_status_reaches_the_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            for data, expected in [({'text': '예시', 'words': []}, 0), ({'text': '예시'}, 1)]:
                source.write_text(json.dumps(data), encoding='utf-8')
                result = self.command(['review', 'validate', str(source)], directory)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(json.loads(result.stdout)['n_files'], 1)


if __name__ == '__main__':
    unittest.main()
