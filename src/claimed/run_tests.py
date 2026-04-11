import unittest
import subprocess


class TestClaimed(unittest.TestCase):

    def test_ibm_sql_query_cpd_test(self):
        correct = subprocess.run(['ipython', './transform/ibm-sql-query-cpd-test.ipynb'], check=True, text=True)
        self.assertFalse('Foo'.isupper())

unittest.main()
