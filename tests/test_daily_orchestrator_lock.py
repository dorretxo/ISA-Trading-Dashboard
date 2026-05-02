import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import daily_orchestrator


class OrchestratorLockTests(unittest.TestCase):
    def test_acquire_and_release_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "orchestrator.lock"

            lock = daily_orchestrator._acquire_orchestrator_lock(
                dry_run=False,
                force_discovery=False,
                portfolio_only=False,
                path=lock_path,
                stale_seconds=60,
            )

            self.assertTrue(lock_path.exists())
            metadata = daily_orchestrator._read_lock_metadata(lock_path)
            self.assertEqual(metadata["pid"], os.getpid())
            self.assertEqual(metadata["token"], lock.token)

            lock.release()
            self.assertFalse(lock_path.exists())

    def test_second_acquire_raises_while_lock_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "orchestrator.lock"
            lock = daily_orchestrator._acquire_orchestrator_lock(
                dry_run=False,
                force_discovery=False,
                portfolio_only=False,
                path=lock_path,
                stale_seconds=60,
            )

            with self.assertRaises(daily_orchestrator.OrchestratorAlreadyRunning) as ctx:
                daily_orchestrator._acquire_orchestrator_lock(
                    dry_run=False,
                    force_discovery=False,
                    portfolio_only=False,
                    path=lock_path,
                    stale_seconds=60,
                )

            self.assertEqual(ctx.exception.metadata.get("pid"), os.getpid())
            lock.release()

    def test_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "orchestrator.lock"
            stale = {
                "pid": 999999,
                "token": "stale-token",
                "started_at": "2000-01-01T00:00:00",
                "hostname": "old-host",
            }
            lock_path.write_text(json.dumps(stale), encoding="utf-8")
            old_time = time.time() - 600
            os.utime(lock_path, (old_time, old_time))

            lock = daily_orchestrator._acquire_orchestrator_lock(
                dry_run=False,
                force_discovery=False,
                portfolio_only=False,
                path=lock_path,
                stale_seconds=1,
            )

            self.assertEqual(lock.reclaimed, stale)
            metadata = daily_orchestrator._read_lock_metadata(lock_path)
            self.assertEqual(metadata["pid"], os.getpid())
            self.assertNotEqual(metadata["token"], "stale-token")

            lock.release()


if __name__ == "__main__":
    unittest.main()
