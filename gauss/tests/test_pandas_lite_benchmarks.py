import unittest

from gauss.evaluation.pandas.autopandas_benchmarks import AutoPandasBenchmarks


class TestAutoPandasBenchmarks(unittest.TestCase):
    def test_pandas_benchmarks(self):
        benchmarks = AutoPandasBenchmarks()
