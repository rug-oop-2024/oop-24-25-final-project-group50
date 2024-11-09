# from autoop.core.ml.dataset import Dataset
# from sklearn.datasets import load_iris
# from autoop.functional.feature import detect_feature_types
# import pandas as pd

# iris = load_iris()
# df = pd.DataFrame(
#             iris.data,
#             columns=iris.feature_names,
#         )

# dataset = Dataset.from_dataframe(
#             name="iris",
#             asset_path="iris.csv",
#             data=df,
#         )

# print(detect_feature_types(dataset))

import unittest
#from autoop.tests.test_database import TestDatabase
#from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
#from autoop.tests.test_pipeline import TestPipeline

if __name__ == '__main__':
    unittest.main()