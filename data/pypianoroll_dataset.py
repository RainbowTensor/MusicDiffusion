import torch
from torch.utils.data import Dataset

import os
import io
import lmdb
import numpy as np
import random
from math import ceil
from copy import deepcopy
import pypianoroll
from scipy.ndimage import gaussian_filter


from .consts import STEPS_PER_BAR, N_STEP

NOTE_COUNT_TRESHOLD = 15


class PypianorollLMDB(Dataset):
    def __init__(self, db_path):

        self.db_path = db_path
        self.perturbation_ratio = 0.4

        self.env = None
        self.txn = None

    def __len__(self):
        return self.env.stat()["entries"]

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        pianoroll = self.read_lmdb(idx)
        pianoroll_input, pianoroll_target = self.get_pianoroll(pianoroll)

        return torch.from_numpy(pianoroll_input), torch.from_numpy(pianoroll_target)

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
        perturbed_pianoroll = selected_bars

        if selected_bars.sum() > 0 and random.random() > 0.5:
            pianoroll_pm = self.pianoroll_array_to_pretty_midi(selected_bars)

            pianoroll_pm = self.shif_notes(pianoroll_pm)
            pianoroll_pm = self.add_adjacent_note(pianoroll_pm)
            pianoroll_pm = self.shif_rhythm(pianoroll_pm)
            pianoroll_pm = self.delete_notes(pianoroll_pm)

            perturbed_pianoroll = pypianoroll.from_pretty_midi(
                pianoroll_pm, resolution=16, algorithm='custom', first_beat_time=0)
            perturbed_pianoroll = perturbed_pianoroll.tracks[0].pianoroll

        return self.normalize_input(perturbed_pianoroll), self.normalize_input(selected_bars)

    def normalize_input(self, x):
        x = x / 127
        if x.shape[0] < N_STEP:
            x = np.pad(x, ((0, N_STEP - x.shape[0])))

        if x.shape[1] < 128:
            x = np.pad(x, ((0, 0), (0, 128 - x.shape[1])))

        if x.shape[1] > 128:
            x = np.zeros([N_STEP, 128])

        if x.shape[0] > N_STEP:
            x = x[:N_STEP]

        return x[None, :, :]

    def blur_input(self, x):
        x_blured = gaussian_filter(x, 2) * 4

        return x_blured.clip(min=0, max=1)

    def pianoroll_array_to_pretty_midi(self, pianoroll_arr):
        pianoroll_back = pypianoroll.Multitrack(
            resolution=16,
            tracks=[pypianoroll.StandardTrack(
                program=0, is_drum=False, pianoroll=pianoroll_arr
            )],
        )

        return pianoroll_back.to_pretty_midi()

    def shif_notes(self, pianoroll_pm):
        notes = pianoroll_pm.instruments[0].notes
        n_perturbed_notes = ceil(len(notes) * self.perturbation_ratio)

        shifted_notes = random.choices(notes, k=n_perturbed_notes)

        for note in shifted_notes:
            note.pitch += random.randint(-2, 2)

        return pianoroll_pm

    def add_adjacent_note(self, pianoroll_pm):
        notes = pianoroll_pm.instruments[0].notes
        n_perturbed_notes = ceil(len(notes) * self.perturbation_ratio)

        shifted_notes = random.choices(notes, k=n_perturbed_notes)
        added_notes = []

        for note in shifted_notes:
            adjacent_note = deepcopy(note)
            offset_direction = 1 if random.random() > 0.5 else -1
            adjacent_note.pitch += 1 * offset_direction
            adjacent_note.start = (note.start + note.duration)
            adjacent_note.end = (note.end + note.duration)

            added_notes.append(adjacent_note)

        notes.extend(added_notes)
        pianoroll_pm.instruments[0].notes = notes

        return pianoroll_pm

    def shif_rhythm(self, pianoroll_pm):
        notes = pianoroll_pm.instruments[0].notes
        n_perturbed_notes = ceil(len(notes) * self.perturbation_ratio)

        shifted_notes = random.choices(notes, k=n_perturbed_notes)

        for note in shifted_notes:
            offset_direction = 1 if random.random() > 0.5 else -1
            note.start += 0.1 * offset_direction
            note.end += 0.1 * offset_direction

        return pianoroll_pm

    def delete_notes(self, pianoroll_pm):
        notes = pianoroll_pm.instruments[0].notes
        n_perturbed_notes = ceil(len(notes) * self, self.perturbation_ratio)

        for i in range(n_perturbed_notes):
            rand_idx = random.randint(0, len(notes))
            try:
                notes.pop(rand_idx)
            except:
                continue

        pianoroll_pm.instruments[0].notes = notes

        return pianoroll_pm
