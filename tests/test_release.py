import importlib.util, json, sys, tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import export_model,package_submission

class ReleaseTests(unittest.TestCase):
    def test_export_replaces_all_parameters_without_numeric_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'export'
            export_model.export(ROOT/'weights/gru_full.pt',ROOT/'weights/ae20.pt',target)
            original=onnx.load(ROOT/'submission/model.onnx');rebuilt=onnx.load(target/'model.onnx')
            self.assertEqual(original.SerializeToString(),rebuilt.SerializeToString())
            for name in ('mean','std'):
                with np.load(target/'normalization.npz') as a,np.load(ROOT/'submission/normalization.npz') as b:
                    np.testing.assert_array_equal(a[name],b[name])

    def test_package_contains_exact_callback_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'submission.zip'
            package_submission.package(ROOT/'submission',path)
            with zipfile.ZipFile(path) as z:
                for name in package_submission.FILES:
                    self.assertEqual(z.read(name),(ROOT/'submission'/name).read_bytes())
            with self.assertRaises(FileExistsError): package_submission.package(ROOT/'submission',path)

    def test_split_is_identical_to_training_loader(self):
        import train_gru_mse_fold as g
        actual=g.load_split(ROOT/'sequence_folds.json',0,0,0,20260928)
        split=json.loads((ROOT/'configs/split.json').read_text())
        self.assertEqual(actual[0],split['train_groups']);self.assertEqual(actual[1],split['holdout_groups'])
        self.assertFalse(set(actual[0])&set(actual[1]))

if __name__=='__main__': unittest.main()
