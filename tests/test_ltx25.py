"""Support boundaries, synchronized media and configuration without model downloads."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from tools.backends import ltx25, ltx25_contract as contract, ltx25_media as media
from tools.backends.ltx25_config import training_config
from tools import split_video

GIB = 1024**3


class LTX25Contracts(unittest.TestCase):
    def test_installer_rejects_1060_before_creating_environment(self):
        from tools.ltx25_runtime import setup_runtime
        probe = SimpleNamespace(returncode=0, stdout='NVIDIA GeForce GTX 1060 6GB, 6.1, 6144\n')
        with patch('sys.argv', ['setup_runtime.py']), patch.dict('os.environ', {}, clear=True), \
             patch.object(setup_runtime.subprocess, 'run', return_value=probe), \
             patch.object(setup_runtime, 'runtime_root', side_effect=AssertionError('must reject before installation')):
            with self.assertRaisesRegex(RuntimeError, 'native BF16'):
                setup_runtime.main()

    def test_gpu_support_is_architecture_and_memory_not_only_vram(self):
        pascal = contract.hardware_report('GTX 1060', (6, 1), 6 * GIB, 32 * GIB)
        self.assertEqual(len(pascal['errors']), 2)
        self.assertIn('native BF16', pascal['errors'][0])
        self.assertTrue(contract.hardware_report('RTX 2060', (7, 5), 12 * GIB, 32 * GIB)['errors'])
        self.assertFalse(contract.hardware_report('RTX 3060', (8, 6), 12 * GIB, 32 * GIB)['errors'])
        small = contract.hardware_report('8 GB Ampere', (8, 6), 8 * GIB, 16 * GIB)
        self.assertFalse(small['errors'])
        self.assertEqual(len(small['warnings']), 2)
        self.assertIn('unverified', small['warnings'][0])

    def test_initial_step_cap_audio_and_continuation_config(self):
        settings = contract.settings_for('training', {})
        request = {'settings': settings, 'clip_count': 32, 'epochs': 10, 'output_dir': 'output/test',
                   'model_name': 'test', 'store': 'models/ltx25', 'prepared_dataset': 'cache/prepared'}
        process = training_config(request)['config']['process'][0]
        self.assertEqual(process['train']['steps'], 1)
        self.assertTrue(process['datasets'][0]['do_audio'])
        self.assertEqual(process['model']['layer_offloading_transformer_percent'], 1.)
        settings.update(max_steps=0, gradient_accumulation=2, train_audio=False)
        request['resume_from'] = 'adapter.safetensors'
        process = training_config(request)['config']['process'][0]
        self.assertEqual(process['train']['steps'], 160)
        self.assertFalse(process['datasets'][0]['do_audio'])
        self.assertEqual(process['network']['pretrained_lora_path'], 'adapter.safetensors')
        self.assertFalse(process['save']['push_to_hub'])

    def test_store_is_portable_and_external_store_is_local_configuration(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {}, clear=True):
            root = Path(folder)
            self.assertEqual(contract.store_root(root), root / 'models/ltx25')
            runtime = contract.runtime_root(root)
            runtime.mkdir(parents=True)
            (runtime / 'paths.json').write_text(json.dumps({'store': str(root / 'ssd')}))
            self.assertEqual(contract.store_root(root), root / 'ssd')


class LTX25Media(unittest.TestCase):
    def test_sound_is_retained_and_silent_clips_rejected_only_for_audio_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.mkv'
            media.command([media.ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y',
                           '-f', 'lavfi', '-i', 'color=s=64x64:r=24:d=4',
                           '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=4',
                           '-c:v', 'libx264', '-c:a', 'pcm_f32le', str(source)])
            split = root / 'clip.mkv'
            split_video.encode_audio_clip(source, split, .5)
            self.assertEqual(split_video.evenly_spaced_starts(3., 1, 3.), [0.])
            info = media.probe(split)
            self.assertEqual(info['audio']['codec_name'], 'pcm_f32le')
            self.assertAlmostEqual(info['duration'], 3., places=1)
            self.assertEqual(info['fps'], 24.)
            # Source outside the active dataset; test real preparation, prompt captions and archive exclusion.
            dataset = root / 'dataset'
            dataset.mkdir()
            split.replace(dataset / 'clip.mkv')
            (dataset / 'clip.txt').write_text('subject_token moving. Audio: A steady tone.')
            settings = contract.settings_for('training', {})
            rows = ltx25.validate_dataset(dataset, settings)
            self.assertEqual(len(rows), 1)
            with patch.object(ltx25, 'ROOT', root):
                cache, prepared = ltx25.prepare_dataset(dataset, settings, rows)
                self.assertEqual(media.probe(prepared / 'clip_00000.mkv')['audio']['codec_name'], 'pcm_f32le')
                with patch.object(ltx25, 'segment', side_effect=AssertionError('cache should be reused')):
                    self.assertEqual(ltx25.prepare_dataset(dataset, settings, rows)[0], cache)
                (dataset / 'clip.txt').write_text('subject_token walking. Audio: A steady tone.')
                changed, _ = ltx25.prepare_dataset(dataset, settings, ltx25.validate_dataset(dataset, settings))
                self.assertNotEqual(cache, changed)
            archive = dataset / '_source_archive'
            archive.mkdir()
            source.replace(archive / 'source.mkv')  # No caption: validation must not read this archived original.
            self.assertEqual(len(ltx25.validate_dataset(dataset, settings)), 1)
            silent = dataset / 'silent.mkv'
            ltx25.segment(dataset / 'clip.mkv', silent, 0, 1., 64, 64, audio=False)
            silent.with_suffix('.txt').write_text('subject_token moving')
            with self.assertRaisesRegex(ValueError, 'requires a soundtrack'):
                ltx25.validate_dataset(dataset, settings)
            settings['train_audio'] = False
            self.assertEqual(len(ltx25.validate_dataset(dataset, settings)), 2)


if __name__ == '__main__':
    unittest.main()
