"""Data source of stream of frames."""
import bz2
import dlib
import queue
import shutil
import threading
import time
from typing import Tuple
import os
from urllib.request import urlopen

import cv2 as cv
import numpy as np
import tensorflow as tf

from core import BaseDataSource


class FramesSource(BaseDataSource):
    """Preprocessing of stream of frames."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 eye_image_shape: Tuple[int, int],
                 staging: bool=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._eye_image_shape = eye_image_shape
        self._proc_mutex = threading.Lock()
        self._read_mutex = threading.Lock()

        self._frame_read_queue = queue.Queue(maxsize=1)
        self._frame_read_thread = threading.Thread(target=self.frame_read_job, name='frame_read')
        self._frame_read_thread.daemon = True
        self._frame_read_thread.start()

        self._current_index = 0
        self._last_frame_index = 0
        self._indices = []
        self._frames = {}
        self._open = True

        # Call parent class constructor
        super().__init__(tensorflow_session, batch_size=batch_size, num_threads=1,
                         fread_queue_capacity=batch_size, preprocess_queue_capacity=batch_size,
                         shuffle=False, staging=staging, **kwargs)

    _short_name = 'Frames'

    @property
    def short_name(self):
        """Short name specifying source."""
        return self._short_name

    def frame_read_job(self):
        """Read frame from webcam."""
        generate_frame = self.frame_generator()
        while True:
            before_frame_read = time.time()
            bgr = next(generate_frame)
            if bgr is not None:
                after_frame_read = time.time()
                with self._read_mutex:
                    self._frame_read_queue.queue.clear()
                    self._frame_read_queue.put_nowait((before_frame_read, bgr, after_frame_read))
        self._open = False

    def frame_generator(self):
        """Read frame from webcam."""
        raise NotImplementedError('Frames::frame_generator not implemented.')

    def entry_generator(self, yield_just_one=False):
        """Generate eye image entries by detecting faces and facial landmarks."""
        try:
            while range(1) if yield_just_one else True:
                # Grab frame
                with self._proc_mutex:
                    before_frame_read, bgr, after_frame_read = self._frame_read_queue.get()
                    bgr = cv.flip(bgr, flipCode=1)  # Mirror
                    current_index = self._last_frame_index + 1
                    self._last_frame_index = current_index

                    grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                    frame = {
                        'frame_index': current_index,
                        'time': {
                            'before_frame_read': before_frame_read,
                            'after_frame_read': after_frame_read,
                        },
                        'bgr': bgr,
                        'grey': grey,
                    }
                    self._frames[current_index] = frame
                    self._indices.append(current_index)

                    # Keep just a few frames around
                    frames_to_keep = 120
                    if len(self._indices) > frames_to_keep:
                        for index in self._indices[:-frames_to_keep]:
                            del self._frames[index]
                        self._indices = self._indices[-frames_to_keep:]

                # Eye image segmentation pipeline
                self.detect_faces(frame)
                self.detect_landmarks(frame)
                self.calculate_smoothed_landmarks(frame)
                self.segment_eyes(frame)
                self.update_face_boxes(frame)
                frame['time']['after_preprocessing'] = time.time()

                for i, eye_dict in enumerate(frame['eyes']):
                    yield {
                        'frame_index': np.int64(current_index),
                        'eye': eye_dict['image'],
                        'eye_index': np.uint8(i),
                    }

        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Preprocess segmented eye images for use as neural network input."""
        eye = entry['eye']
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if self.data_format == 'NHWC' else 0)
        entry['eye'] = eye
        return entry

    def detect_faces(self, frame):
        """Detect all faces in a frame."""
        frame_index = frame['frame_index']
        previous_index = self._indices[self._indices.index(frame_index) - 1]
        previous_frame = self._frames[previous_index]
        if ('last_face_detect_index' not in previous_frame or
                frame['frame_index'] - previous_frame['last_face_detect_index'] > 59):
            detector = get_face_detector()
            if detector.__class__.__name__ == 'CascadeClassifier':
                detections = detector.detectMultiScale(frame['grey'])
            else:
                detections = detector(cv.resize(frame['grey'], (0, 0), fx=0.5, fy=0.5), 0)
            faces = []
            for d in detections:
                try:
                    l, t, r, b = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
                    l *= 2
                    t *= 2
                    r *= 2
                    b *= 2
                    w, h = r - l, b - t
                except AttributeError:  # Using OpenCV LBP detector on CPU
                    l, t, w, h = d
                faces.append((l, t, w, h))
            faces.sort(key=lambda bbox: bbox[0])
            frame['faces'] = faces
            frame['last_face_detect_index'] = frame['frame_index']

            # Clear previous known landmarks. This is to disable smoothing when new face detect
            # occurs. This allows for recovery of drifted detections.
            previous_frame['landmarks'] = []
        else:
            frame['faces'] = previous_frame['faces']
            frame['last_face_detect_index'] = previous_frame['last_face_detect_index']

    def detect_landmarks(self, frame):
        """Detect 5-point facial landmarks for faces in frame."""
        predictor = get_landmarks_predictor()
        landmarks = []
        for face in frame['faces']:
            l, t, w, h = face
            rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l+w), bottom=int(t+h))
            landmarks_dlib = predictor(frame['grey'], rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts
            landmarks.append(np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)]))
        frame['landmarks'] = landmarks

    _smoothing_window_size = 10
    _smoothing_coefficient_decay = 0.5
    _smoothing_coefficients = None

    def calculate_smoothed_landmarks(self, frame):
        """If there are previous landmark detections, try to smooth current prediction."""
        # Cache coefficients based on defined sliding window size
        if self._smoothing_coefficients is None:
            coefficients = np.power(self._smoothing_coefficient_decay,
                                    list(reversed(list(range(self._smoothing_window_size)))))
            coefficients /= np.sum(coefficients)
            self._smoothing_coefficients = coefficients.reshape(-1, 1)

        # Get a window of frames
        current_index = self._indices.index(frame['frame_index'])
        a = current_index - self._smoothing_window_size + 1
        if a < 0:
            """If slice extends before last known frame."""
            return
        window_indices = self._indices[a:current_index + 1]
        window_frames = [self._frames[idx] for idx in window_indices]
        window_num_landmark_entries = np.array([len(f['landmarks']) for f in window_frames])
        if np.any(window_num_landmark_entries == 0):
            """Any frame has zero faces detected."""
            return
        if not np.all(window_num_landmark_entries == window_num_landmark_entries[0]):
            """Not the same number of faces detected in entire window."""
            return

        # Apply coefficients to landmarks in window
        window_landmarks = np.asarray([f['landmarks'] for f in window_frames])
        frame['smoothed_landmarks'] = np.sum(
            np.multiply(window_landmarks.reshape(self._smoothing_window_size, -1),
                        self._smoothing_coefficients),
            axis=0,
        ).reshape(window_num_landmark_entries[-1], -1, 2)

    def segment_eyes(self, frame):
        """From found landmarks in previous steps, segment eye image."""
        eyes = []

        # Final output dimensions
        oh, ow = self._eye_image_shape

        # Select which landmarks (raw/smoothed) to use
        frame_landmarks = (frame['smoothed_landmarks'] if 'smoothed_landmarks' in frame
                           else frame['landmarks'])

        for face, landmarks in zip(frame['faces'], frame_landmarks):
            # Segment eyes
            # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
            for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
                x1, y1 = landmarks[corner1, :]
                x2, y2 = landmarks[corner2, :]
                eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
                if eye_width == 0.0:
                    continue
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                # Centre image on middle of eye
                translate_mat = np.asmatrix(np.eye(3))
                translate_mat[:2, 2] = [[-cx], [-cy]]
                inv_translate_mat = np.asmatrix(np.eye(3))
                inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

                # Rotate to be upright
                roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
                rotate_mat = np.asmatrix(np.eye(3))
                cos = np.cos(-roll)
                sin = np.sin(-roll)
                rotate_mat[0, 0] = cos
                rotate_mat[0, 1] = -sin
                rotate_mat[1, 0] = sin
                rotate_mat[1, 1] = cos
                inv_rotate_mat = rotate_mat.T

                # Scale
                scale = ow / eye_width
                scale_mat = np.asmatrix(np.eye(3))
                scale_mat[0, 0] = scale_mat[1, 1] = scale
                inv_scale = 1.0 / scale
                inv_scale_mat = np.asmatrix(np.eye(3))
                inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

                # Centre image
                centre_mat = np.asmatrix(np.eye(3))
                centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
                inv_centre_mat = np.asmatrix(np.eye(3))
                inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

                # Get rotated and scaled, and segmented image
                transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
                inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                     inv_centre_mat)
                eye_image = cv.warpAffine(frame['grey'], transform_mat[:2, :], (ow, oh))
                if is_left:
                    eye_image = np.fliplr(eye_image)
                eyes.append({
                    'image': eye_image,
                    'inv_landmarks_transform_mat': inv_transform_mat,
                    'side': 'left' if is_left else 'right',
                })
        frame['eyes'] = eyes

    def update_face_boxes(self, frame):
        """Update face bounding box based on detected landmarks."""
        frame_landmarks = (frame['smoothed_landmarks'] if 'smoothed_landmarks' in frame
                           else frame['landmarks'])
        for i, (face, landmarks) in enumerate(zip(frame['faces'], frame_landmarks)):
            x_min, y_min = np.amin(landmarks, axis=0)
            x_max, y_max = np.amax(landmarks, axis=0)
            x_mid, y_mid = 0.5 * (x_max + x_min), 0.5 * (y_max + y_min)
            w, h = x_max - x_min, y_max - y_min
            new_w = 2.2 * max(h, w)
            half_w = 0.5 * new_w
            frame['faces'][i] = (int(x_mid - half_w), int(y_mid - half_w), int(new_w), int(new_w))

            # x1, y1 = landmarks[0, :]
            # x2, y2 = landmarks[3, :]
            # face_width = 2.5 * np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # if face_width == 0.0:
            #     continue
            #
            # cx, cy = landmarks[4, :]
            # roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
            #
            # hdx = 0.5 * face_width * (2. - np.abs(np.cos(roll)))
            # hdy = 0.5 * face_width * (1. + np.abs(np.sin(roll)))
            # print(np.degrees(roll), face_width, hdx, hdy)
            # frame['faces'][i] = (int(cx - hdx), int(cy - hdy), int(2*hdx), int(2*hdy))

_face_detector = None
_landmarks_predictor = None


def _get_dlib_data_file(dat_name):
    dat_dir = os.path.relpath('%s/../3rdparty' % os.path.basename(__file__))
    dat_path = '%s/%s' % (dat_dir, dat_name)
    if not os.path.isdir(dat_dir):
        os.mkdir(dat_dir)

    # Download trained shape detector
    if not os.path.isfile(dat_path):
        with urlopen('http://dlib.net/files/%s.bz2' % dat_name) as response:
            with bz2.BZ2File(response) as bzf, open(dat_path, 'wb') as f:
                shutil.copyfileobj(bzf, f)

    return dat_path


def _get_opencv_xml(xml_name):
    xml_dir = os.path.relpath('%s/../3rdparty' % os.path.basename(__file__))
    xml_path = '%s/%s' % (xml_dir, xml_name)
    if not os.path.isdir(xml_dir):
        os.mkdir(xml_dir)

    # Download trained shape detector
    if not os.path.isfile(xml_path):
        url_stem = 'https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades'
        with urlopen('%s/%s' % (url_stem, xml_name)) as response:
            with open(xml_path, 'wb') as f:
                shutil.copyfileobj(response, f)

    return xml_path


def get_face_detector():
    """Get a singleton dlib face detector."""
    global _face_detector
    if not _face_detector:
        try:
            dat_path = _get_dlib_data_file('mmod_human_face_detector.dat')
            _face_detector = dlib.cnn_face_detection_model_v1(dat_path)
        except:
            xml_path = _get_opencv_xml('lbpcascade_frontalface_improved.xml')
            _face_detector = cv.CascadeClassifier(xml_path)
    return _face_detector


def get_landmarks_predictor():
    """Get a singleton dlib face landmark predictor."""
    global _landmarks_predictor
    if not _landmarks_predictor:
        dat_path = _get_dlib_data_file('shape_predictor_5_face_landmarks.dat')
        # dat_path = _get_dlib_data_file('shape_predictor_68_face_landmarks.dat')
        _landmarks_predictor = dlib.shape_predictor(dat_path)
    return _landmarks_predictor
