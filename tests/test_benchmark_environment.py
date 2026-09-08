import importlib.util
import pathlib
import unittest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_environment", pathlib.Path(__file__).resolve().parents[1]
    / "scripts" / "benchmark_environment.py")
env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(env)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_android_snapshot_retains_timestamped_raw_evidence(self):
        raw = "30000\n20\n0\nPower Manager State:\n  mWakefulness=Awake\n"
        commands = []
        def shell(command):
            commands.append(command)
            return raw
        sample = env.android_display_snapshot(shell)
        self.assertEqual(sample["raw"], raw)
        self.assertEqual(sample["screen_brightness"], 20)
        self.assertEqual(sample["wakefulness"], "Awake")
        self.assertLessEqual(sample["started_at_unix"], sample["ended_at_unix"])
        self.assertNotIn("settings put", commands[0])
        for incomplete in ["null\n20\n0\nmWakefulness=Awake", "30000\n20\n0\n"]:
            with self.assertRaises(ValueError):
                env.android_display_snapshot(lambda _: incomplete)

    def test_interval_boundary_is_ambiguous_and_still_excluded(self):
        for start, end, relation, valid, proven in [
            (8, 9, "outside", True, False),
            (9.2, 10.2, "boundary_ambiguous", False, False),
            (10, 11, "inside", False, True),
            (19.5, 20.5, "boundary_ambiguous", False, False),
            (20, 21, "outside", True, False),
            (None, 10.2, "unknown_interval", False, False),
            (None, 9, "outside", True, False),
            (None, 10, "outside", True, False),
            (None, 21, "unknown_interval", False, False),
        ]:
            with self.subTest(start=start, end=end):
                result = env.foreign_gpu_activity(
                    start=start, end=end, window_start=10, window_end=20,
                    samples=[{"pid": 4, "utilization": 5.6, "status": 0}], owned_pids={42})
                self.assertEqual((result["relation"], result["valid"],
                                  result["proven_foreign_activity_in_window"]),
                                 (relation, valid, proven))

    def test_owned_counters_and_invalid_samples(self):
        args = dict(start=10, end=11, window_start=10, window_end=20, owned_pids={42})
        self.assertTrue(env.foreign_gpu_activity(
            **args, samples=[{"pid": 42, "utilization": 99, "status": 0}])["valid"])
        for samples in [[], [{"pid": 4, "utilization": 0, "status": 1}],
                        [{"pid": 4, "utilization": float("nan"), "status": 0}]]:
            self.assertFalse(env.foreign_gpu_activity(**args, samples=samples)["valid"])

    def test_hash_warming_waits_and_keeps_all_readings(self):
        timer = FakeClock()
        readings = iter([35, 34, 33])
        records, checks = [], []
        result = env.wait_for_prelaunch(
            lambda: {"temperature_c": next(readings)}, lambda: checks.append(timer.now()),
            records.append, clock=timer.now, sleep=timer.sleep, poll_seconds=2)
        self.assertEqual(result["temperature_c"], 33)
        self.assertEqual([x["temperature_c"] for x in records], [35, 34, 33])
        self.assertEqual(checks, [0, 2, 4, 4])

    def test_hot_device_times_out_without_inference(self):
        timer, records = FakeClock(), []
        with self.assertRaises(TimeoutError):
            env.wait_for_prelaunch(lambda: {"temperature_c": 35}, lambda: None,
                                  records.append, timeout_seconds=3, poll_seconds=2,
                                  clock=timer.now, sleep=timer.sleep)
        self.assertEqual(timer.now(), 3)
        self.assertEqual(len(records), 3)

    def test_ownership_change_prevents_launch_after_cooling(self):
        checks = []
        def verify():
            checks.append(True)
            if len(checks) == 2:
                raise RuntimeError("lease changed")
        with self.assertRaisesRegex(RuntimeError, "lease changed"):
            env.wait_for_prelaunch(lambda: {"temperature_c": 32}, verify, lambda _: None)

    def test_manual_brightness_drift_is_not_silently_reset(self):
        expected = dict(screen_off_timeout=30000, screen_brightness_mode=0,
                        screen_brightness=20, wakefulness="Awake")
        actual = dict(expected, screen_brightness=1699)
        records = []
        with self.assertRaisesRegex(RuntimeError, "screen_brightness"):
            env.verify_display(lambda: actual, expected, records.append)
        self.assertEqual(records, [actual])
        self.assertEqual(actual["screen_brightness"], 1699)

    def test_auto_brightness_and_async_restoration(self):
        expected = dict(screen_off_timeout=30000, screen_brightness_mode=1,
                        screen_brightness=5, wakefulness="Asleep")
        readings = iter([dict(expected, wakefulness="Dozing", screen_brightness=113),
                         dict(expected, screen_brightness=114)])
        timer, records = FakeClock(), []
        result = env.verify_display(lambda: next(readings), expected, records.append,
                                    clock=timer.now, sleep=timer.sleep)
        self.assertEqual(result["screen_brightness"], 114)
        self.assertEqual(len(records), 2)

    def test_missing_display_fields_and_transition_timeout(self):
        self.assertTrue(env.display_mismatches({}, {}))
        timer = FakeClock()
        expected = dict(screen_off_timeout=30000, screen_brightness_mode=1,
                        wakefulness="Asleep")
        with self.assertRaises(TimeoutError):
            env.verify_display(lambda: dict(expected, wakefulness="Dozing"), expected,
                               lambda _: None, timeout_seconds=1,
                               clock=timer.now, sleep=timer.sleep)


if __name__ == "__main__":
    unittest.main()
