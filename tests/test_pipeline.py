import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mice_protein_pipeline.data_processing import create_group_datasets, load_dataset


class PipelineProcessingTests(unittest.TestCase):
    def test_load_dataset_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            df = pd.DataFrame(
                {
                    "MouseID": [1, 2, 3, 4],
                    "Genotype": ["A", "A", "B", "B"],
                    "Treatment": ["t1", "t1", "t2", "t2"],
                    "Behavior": ["b1", "b1", "b2", "b2"],
                    "class": ["c-CS-m", "c-CS-s", "t-CS-m", "t-CS-s"],
                    "Protein1": [1.0, 2.0, 3.0, 4.0],
                    "Protein2": [4.0, 3.0, 2.0, 1.0],
                }
            )
            df.to_csv(csv_path, index=False)

            processed = load_dataset(csv_path)
            self.assertNotIn("class", processed.columns)
            self.assertIn("mice_class", processed.columns)
            self.assertEqual(processed["mice_class"].nunique(), 4)

            groups = create_group_datasets(processed)
            self.assertIn("normal_learning", groups)
            self.assertIn("trisomy_success_vs_failure", groups)
            self.assertIn("normal_vs_trisomy_failure", groups)
            self.assertGreater(len(groups["normal_learning"]), 0)
            self.assertGreater(len(groups["trisomy_success_vs_failure"]), 0)


if __name__ == "__main__":
    unittest.main()
