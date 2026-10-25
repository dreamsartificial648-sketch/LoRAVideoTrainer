"""Model identities and defaults; safe to import from the Tkinter process."""
from importlib import import_module

WAN = 'Wan 2.1 T2V 1.3B'
MODELSCOPE = 'ModelScope T2V 1.7B'
LTX = 'LTX-Video 2B 0.9.6 dev'
LABELS = (WAN, MODELSCOPE, LTX)
IDS = {MODELSCOPE: 'modelscope_video', LTX: 'ltx_video'}
WAN_FORMATS = ('Landscape — 832 x 480', 'Portrait — 480 x 832')


def contract(label):
    return import_module(f'tools.backends.{IDS[label]}_contract')


def frames_for_seconds(label, seconds, fps):
    if label == WAN:
        return max(5, int(round((seconds * fps - 1) / 4)) * 4 + 1)
    return contract(label).frames_for_seconds(seconds, fps)
