"""Webcam data source for gaze estimation."""
import cv2 as cv

from .frames import FramesSource


class Webcam(FramesSource):
    """Webcam frame grabbing and preprocessing."""

    def __init__(self, camera_id=0, fps=60, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'Webcam'

        self._capture = cv.VideoCapture(camera_id)
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self._capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self._capture.set(cv.CAP_PROP_FPS, fps)

        # Call parent class constructor
        super().__init__(**kwargs)

    def frame_generator(self):
        """Read frame from webcam."""
        while True:
            ret, bgr = self._capture.read()
            if ret:
                yield bgr
