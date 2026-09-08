import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("htp", Path(__file__).resolve().parents[1] / "scripts/benchmark_htp_placement.py")
htp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(htp)
START = 'slot: id 0 | task 1 | new prompt, n_ctx_slot = 2048, task.n_tokens = 128\n'
END = 'slot: id 0 | task 1 | stop processing: n_tokens = 135\n'


def node(device, op='GET_ROWS', name='embd'):
    return f'node # 0 ( {op}): {name} ( 256K) [ {device} ] use=2,c=1: input ( 270M) [ CPU ]\n'


class PlacementTests(unittest.TestCase):
    def test_reservation_only_cannot_pass(self):
        result = htp.review_placement(node('HTP0') * 2)
        self.assertFalse(result['eligible_for_pure_htp_review'])
        self.assertEqual(result['requests'], [])

    def test_cpu_embedding_overrides_reservation(self):
        result = htp.review_placement(node('HTP0') * 2 + START + node('CPU') + node('HTP0') + END)
        self.assertEqual(result['placement'], 'cpu_assisted')
        self.assertEqual(result['requests'][0]['cpu_ops'], {'GET_ROWS': 1})
        self.assertFalse(result['eligible_for_pure_htp_review'])

    def test_cpu_k_quant_matmul_is_reported(self):
        result = htp.review_placement(START + node('CPU', 'MUL_MAT', 'Qcur-0') * 2 + END)
        self.assertEqual(result['requests'][0]['cpu_ops'], {'MUL_MAT': 2})

    def test_destination_not_source_determines_device(self):
        result = htp.review_placement(START + node('HTP0') * 2 + END)
        self.assertTrue(result['eligible_for_pure_htp_review'])
        self.assertFalse(result['performance_publishable'])
        self.assertFalse(result['correctness_verified'])

    def test_incomplete_unknown_and_overlapping_fail_closed(self):
        for log in [START + node('HTP0') * 2, START + node('NULL') * 2 + END,
                    START + node('HTP0') + END, START + 'node # malformed\n' + END,
                    START + START + node('HTP0') * 2 + END,
                    START + node('HTP0') * 2 + END.replace('task 1', 'task 2')]:
            with self.subTest(log=log):
                self.assertFalse(htp.review_placement(log)['eligible_for_pure_htp_review'])

    def test_each_request_requires_evidence(self):
        result = htp.review_placement(START + node('HTP0') * 2 + END + START + END)
        self.assertFalse(result['eligible_for_pure_htp_review'])


if __name__ == '__main__':
    unittest.main()
