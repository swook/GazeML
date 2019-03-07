"""HDF5 data source for gaze estimation."""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from core import BaseDataSource
import util.gazemap


class HDF5Source(BaseDataSource):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 keys_to_use: List[str],
                 hdf_path: str,
                 testing=False,
                 eye_image_shape=(36, 60),
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            self._short_name += ':test'

        # Cache some parameters
        self._eye_image_shape = eye_image_shape

        # Create global index over all specified keys
        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = hdf5[key]['eye'].shape[0]
            for i in range(n):
                self._index_to_key[index_counter] = (key, i)
                index_counter += 1
        self._num_entries = index_counter

        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from HDF5."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                key, index = self._index_to_key[current_index]
                data = self._hdf5[key]
                entry = {}
                for name in ('eye', 'gaze', 'head'):
                    if name in data:
                        entry[name] = data[name][index, :]
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Resize eye image and normalize intensities."""
        oh, ow = self._eye_image_shape
        eye = entry['eye']
        eye = cv.resize(eye, (ow, oh))
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, axis=0 if self.data_format == 'NCHW' else -1)
        entry['eye'] = eye

        entry['gazemaps'] = util.gazemap.from_gaze2d(
            entry['gaze'], output_size=(oh, ow), scale=0.5,
        ).astype(np.float32)
        if self.data_format == 'NHWC':
            np.transpose(entry['gazemaps'], (1, 2, 0))

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry
