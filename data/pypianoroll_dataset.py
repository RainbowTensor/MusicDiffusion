import torch
from torch.utils.data import Dataset

import os
import io
import lmdb
import numpy as np
import random
import pypianoroll
from scipy.ndimage import gaussian_filter


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

        # if lmdb_data is None:
        #     rand_idx = random.randint(0, self.__len__())
        #     return self.read_lmdb(rand_idx)

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

        # if x.sum() == 0:
        #     return self.get_pianoroll(pianoroll)

        if x.shape[0] < N_STEP:
            x = np.pad(x, ((0, N_STEP - x.shape[0])))

        if x.shape[1] < 128:
            x = np.pad(x, ((0, 0), (0, 128 - x.shape[1])))

        if x.shape[1] > 128:
            x = np.zeros([N_STEP, 128])

        if x.shape[0] > N_STEP:
            x = x[:N_STEP]

        x = self.blur_input()

        return x[None, :, :]

    def blur_input(self, x):
        x_blured = gaussian_filter(x, 5) * 15

        return x_blured.clip(min=0, max=1)
