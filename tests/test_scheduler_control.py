import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from scheduler_control import env_flag_enabled, release_file_lock, try_acquire_file_lock


class EnvFlagEnabledTests(unittest.TestCase):
    def test_defaults_when_missing(self):
        self.assertTrue(env_flag_enabled("PILAR_TEST_FLAG", default=True))
        self.assertFalse(env_flag_enabled("PILAR_TEST_FLAG", default=False))

    def test_truthy_and_falsey_values(self):
        original = os.environ.get("PILAR_TEST_FLAG")
        try:
            os.environ["PILAR_TEST_FLAG"] = "yes"
            self.assertTrue(env_flag_enabled("PILAR_TEST_FLAG"))
            os.environ["PILAR_TEST_FLAG"] = "off"
            self.assertFalse(env_flag_enabled("PILAR_TEST_FLAG", default=True))
        finally:
            if original is None:
                os.environ.pop("PILAR_TEST_FLAG", None)
            else:
                os.environ["PILAR_TEST_FLAG"] = original


class FileLockTests(unittest.TestCase):
    def _run_child(self, lock_path: Path) -> str:
        script = textwrap.dedent(
            """
            import sys
            from scheduler_control import release_file_lock, try_acquire_file_lock

            handle = try_acquire_file_lock(sys.argv[1])
            print("1" if handle else "0")
            release_file_lock(handle)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(lock_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def test_file_lock_is_single_leader(self):
        lock_path = Path(__file__).resolve().parents[1] / "session_files" / "scheduler_test.lock"
        try:
            handle = try_acquire_file_lock(lock_path)
            self.assertIsNotNone(handle)
            try:
                self.assertEqual(self._run_child(lock_path), "0")
            finally:
                release_file_lock(handle)

            self.assertEqual(self._run_child(lock_path), "1")
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
