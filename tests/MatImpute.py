import unittest

import pandas as pd

from tests.utils import generate_random_missing_data, check_df_filled
from matimpute import MatImputer


class MyTestCase(unittest.TestCase):
    def test_impute_success(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5]})
        df_with_null = generate_random_missing_data(df, 0.1)
        mat_impute = MatImputer()
        df_filled = mat_impute.transform(df_with_null)
        self.assertTrue(check_df_filled(df_filled))


if __name__ == '__main__':
    unittest.main()
