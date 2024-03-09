import torch
from torch.utils.data import Dataset

import os
import io
import lmdb
import numpy as np
import random
import pypianoroll

from .consts import STEPS_PER_BAR, N_STEP

NOTE_COUNT_TRESHOLD = 15


class PypianorollLMDB(Dataset):
    def __init__(self, db_path):

        self.db_path = db_path

        self.env = None
        self.txn = None

    def __len__(self):
        return self.env.stat()["entries"]

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        pianoroll = self.read_lmdb(idx)
        pianoroll_sample = self.get_pianoroll(pianoroll)

        return torch.Tensor(pianoroll_sample)

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False,
            readahead=False, meminit=False
        )

        self.txn = self.env.begin()

    def read_lmdb(self, idx):
        str_id = '{:08}'.format(idx)
        lmdb_data = self.txn.get(str_id.encode())
        lmdb_data = self.bytes_to_pypianoroll(lmdb_data)

        return lmdb_data

    def bytes_to_pypianoroll(self, bytes_obj):
        memfile = io.BytesIO()
        memfile.write(bytes_obj)
        memfile.seek(0)

        return pypianoroll.load(memfile)

    def get_pianoroll(self, pianoroll):
        track = random.choice(pianoroll.tracks)
        track_pianoroll = track.pianoroll

        max_len = track_pianoroll.shape[0] - N_STEP
        max_len = (max_len // STEPS_PER_BAR) * STEPS_PER_BAR
        bar_indices = list(range(0, max_len, STEPS_PER_BAR)) + [max_len]
        bar_index = random.choice(bar_indices)

        selected_bars = track_pianoroll[bar_index:bar_index + N_STEP]
        x = selected_bars / 127

        if x.sum() == 0:
            return self.get_pianoroll(pianoroll)

        return x[None, :, :]
