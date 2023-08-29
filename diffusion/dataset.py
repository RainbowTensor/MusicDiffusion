import torch
from torch.utils.data import Dataset

import os
import io
import lmdb
import numpy as np
import random
from scipy.ndimage import gaussian_filter


INSTR_TO_INDEX = {
    -1: 0,
    0: 1,
    25: 2,
    32: 3,
    40: 4,
    80: 5
}

NOTE_COUNT_TRESHOLD = 15

class LakhPrmat2cLMDB(Dataset):
    def __init__(self, db_path):

        self.db_path = db_path
        self.index_to_instr = dict([(v, k) for k, v in INSTR_TO_INDEX.items()])

        self.env = None
        self.txn = None

    def __len__(self):
        return self.env.stat()["entries"]

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        prmat2c_instr = self.read_lmdb(idx)
        prmat2c = self.get_prmat2c_and_label(prmat2c_instr)

        return torch.Tensor(prmat2c)

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
        lmdb_data = self.bytes_to_array(lmdb_data)

        return lmdb_data

    def bytes_to_array(self, bytes_obj):
        memfile = io.BytesIO()
        memfile.write(bytes_obj)
        memfile.seek(0)

        return np.load(memfile)['pianoroll']

    def get_prmat2c_and_label(self, prmat2c_instr):
        instr_note_count = prmat2c_instr[:, 0, :, :].sum(-1).sum(-1)
        valid_instr_idxs = np.where(instr_note_count > NOTE_COUNT_TRESHOLD)[0]

        if valid_instr_idxs.sum() < 2:
            return self.__getitem__(random.randint(0, len(self)))

        prmat2c_instr = np.clip(prmat2c_instr / 128, a_min=[0], a_max=[1])
        
        return prmat2c_instr
