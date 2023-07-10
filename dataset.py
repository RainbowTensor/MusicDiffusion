import torch
from torch.utils.data import Dataset

import os
import io
import lmdb
import numpy as np
import random


INSTR_TO_INDEX = {
    -1: 0,
    0: 1,
    25: 2,
    32: 3,
    40: 4,
    80: 5
}

NOTE_COUNT_TRESHOLD = 15


class POP909Prmat2cDataset(Dataset):
    def __init__(self, file_list):

        data = []
        for data_file in file_list:
            data_loaded = np.load(data_file)
            data.append(data_loaded.astype(np.uint8))

        self.data = np.concatenate(data, axis=0)

        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx].astype(np.int32)
        # sample[sample == 0] = -1

        sample = torch.Tensor(sample)

        return sample

    
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
        prmat2c, instr_label = self.get_prmat2c_and_label(prmat2c_instr)

        return torch.Tensor(prmat2c), torch.from_numpy(instr_label).long()

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

        if len(valid_instr_idxs) == 0:
            return prmat2c_instr[0], np.array(0)

        random_idx = random.choice(valid_instr_idxs)

        return prmat2c_instr[random_idx], np.array(random_idx)