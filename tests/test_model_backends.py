import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import model_profiles as profiles
from tools.backends import common


class ModelContracts(unittest.TestCase):
    def test_temporal_rules_and_invalid_settings(self):
        self.assertEqual(profiles.frames_for_seconds(profiles.WAN, 2, 12), 25)
        self.assertEqual(profiles.frames_for_seconds(profiles.MODELSCOPE, 2, 8), 16)
        self.assertEqual(profiles.frames_for_seconds(profiles.LTX, 2, 24), 49)
        for label in (profiles.MODELSCOPE, profiles.LTX):
            contract = profiles.contract(label)
            for values in ({'rank': 65}, {'rank': 1.5}, {'learning_rate': float('nan')}, {'target_frames': '5'}):
                with self.assertRaises(ValueError):
                    contract.settings_for('training', values)
            with self.assertRaises(ValueError):
                contract.settings_for('generation', {'prompt': 'test', 'format': '832x480'})

    def test_cache_reuse_and_caption_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'clip.mp4'
            clip.write_bytes(b'video fixture')
            caption = clip.with_suffix('.txt')
            caption.write_text('subject_token moving')
            cache = root / 'cache/modelscope_video'
            request = dict(backend='modelscope_video', dataset_dir=str(root),
                           dataset_files=common.fingerprint(root), cache_dir=str(root / 'cache'), training_cache=str(cache))
            settings = dict(resolution='256x256', target_frames='8', seed=42)
            builds = []

            def build():
                cache.mkdir()
                result = cache / 'clip_000000.safetensors'
                result.write_bytes(b'tensor fixture')
                builds.append(result)
                return [result]

            common.cached_training_data(request, settings, build)
            common.cached_training_data(request, settings, build)
            self.assertEqual(len(builds), 1)
            caption.write_text('subject_token walking in a garden')
            request['dataset_files'] = common.fingerprint(root)
            common.cached_training_data(request, settings, build)
            self.assertEqual(len(builds), 2)
            settings['target_frames'] = '16'
            common.cached_training_data(request, settings, build)
            self.assertEqual(len(builds), 3)
            request['training_cache'] = str(root)
            with self.assertRaises(ValueError):
                common.cached_training_data(request, settings, build)
            self.assertTrue(clip.exists())

    def test_decoded_dataset_validation(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / 'clip.mp4'
            with av.open(str(clip), 'w') as container:
                stream = container.add_stream('mpeg4', rate=8)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for _ in range(8):
                    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            clip.with_suffix('.txt').write_text('subject_token moving')
            self.assertEqual(len(common.validate_dataset(root, 'subject_token', 8)), 2)
            with self.assertRaisesRegex(ValueError, 'at least 17'):
                common.validate_dataset(root, 'subject_token', 17)
            with self.assertRaisesRegex(ValueError, 'exact trigger'):
                common.validate_dataset(root, 'other_token', 8)


class ModelInterface(unittest.TestCase):
    def test_switching_and_command_routing(self):
        import app
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(app, 'OUTPUT_DIR', root / 'output'), \
                 patch.object(app.VideoLoraTrainerApp, '_load_settings', return_value={}), \
                 patch.object(app.VideoLoraTrainerApp, '_load_storyboard', return_value=[]), \
                 patch.object(app.VideoLoraTrainerApp, '_save_settings', return_value=True):
                ui = app.VideoLoraTrainerApp()
                ui.withdraw()
                try:
                    for label in profiles.LABELS:
                        ui.model_var.set(label)
                        ui._on_model_selected()
                        ui.update_idletasks()
                        self.assertEqual(ui.checkpoint_var.get(), '')
                        command = ui._training_command()
                        if label == profiles.WAN:
                            self.assertIn('networks.lora_wan', command)
                            self.assertEqual(ui._frames_for_seconds(2, 12), 25)
                        else:
                            self.assertIn(ui._backend_id(), command)
                            self.assertIn('train', command)
                            self.assertNotIn('networks.lora_wan', command)
                            values = json.loads(command[command.index('--settings') + 1])
                            self.assertEqual(values['rank'], 4)
                            self.assertEqual(str(ui.training_blocks_widget.cget('state')), 'disabled')
                            self.assertEqual((ui.width_var.get(), ui.height_var.get()),
                                             (512, 320) if label == profiles.LTX else (256, 256))
                            ui._set_fast_preview()
                            self.assertEqual((ui.width_var.get(), ui.height_var.get()), (256, 256))
                finally:
                    if ui.settings_save_after_id:
                        ui.after_cancel(ui.settings_save_after_id)
                    ui.destroy()


if __name__ == '__main__':
    unittest.main()
