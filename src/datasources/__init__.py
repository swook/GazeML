"""Data-source definitions (one class per file)."""
from .frames import FramesSource
from .hdf5 import HDF5Source
from .unityeyes import UnityEyes
from .video import Video
from .webcam import Webcam
__all__ = ('FramesSource', 'HDF5Source', 'UnityEyes', 'Video', 'Webcam')
