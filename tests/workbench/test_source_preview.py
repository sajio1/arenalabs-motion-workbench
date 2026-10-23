from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from arenalab_workbench import source_preview as preview

class PreviewTests(unittest.TestCase):
    def test_world_vertices_and_clock_are_exact_and_no_model_is_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();store=root/'store';store.mkdir()
            vertices=np.arange(36,dtype=np.float32).reshape(3,4,3)/100
            faces=np.array([[0,1,2],[1,2,3]],np.int32);times=10+np.arange(3)*.033333
            np.savez(source/'source_mesh.npz',vertices_world_m=vertices,faces=faces,timestamps_s=times,source_frame_indices=[900,903,906],exact_fps=1/.033333)
            (source/'source_provenance.json').write_text(json.dumps({'human_model':'smplx','source_evidence':'fitted','source_uniform_similarity_applied':False}))
            with patch.object(preview,'paths',SimpleNamespace(store=store)):
                result=preview.export_source_preview(source,store/'scene')
                self.assertEqual(result,preview.export_source_preview(source,store/'scene'))
                path=store/result['vertices_url'].removeprefix('/assets/')
                np.testing.assert_array_equal(np.fromfile(path,dtype='<f4').reshape(3,4,3),vertices)
                clock=json.loads((store/result['clock_url'].removeprefix('/assets/')).read_text())
                np.testing.assert_array_equal(clock['timestamps_s'],times)
                self.assertFalse(result['source_uniform_similarity_applied'])
                with self.assertRaisesRegex(ValueError,'asset store'):
                    preview.export_source_preview(source,root/'outside')
            self.assertEqual({p.suffix for p in (store/'scene').rglob('*') if p.is_file()},{'.bin','.json'})

if __name__=='__main__':unittest.main()
