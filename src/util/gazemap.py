"""Utility methods for generating gazemaps."""
import cv2 as cv
import numpy as np

height_to_eyeball_radius_ratio = 1.1
eyeball_radius_to_iris_diameter_ratio = 1.0

def from_gaze2d(gaze, output_size, scale=1.0):
    """Generate a normalized pictorial representation of 3D gaze direction."""
    gazemaps = []
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)
    oh_2 = int(np.round(0.5 * oh))
    ow_2 = int(np.round(0.5 * ow))
    r = int(height_to_eyeball_radius_ratio * oh_2)
    theta, phi = gaze
    theta = -theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Draw iris
    eyeball_radius = int(height_to_eyeball_radius_ratio * oh_2)
    iris_radius_angle = np.arcsin(0.5 * eyeball_radius_to_iris_diameter_ratio)
    iris_radius = eyeball_radius_to_iris_diameter_ratio * eyeball_radius
    iris_distance = float(eyeball_radius) * np.cos(iris_radius_angle)
    iris_offset = np.asarray([
        -iris_distance * sin_phi * cos_theta,
        iris_distance * sin_theta,
    ])
    iris_centre = np.asarray([ow_2, oh_2]) + iris_offset
    angle = np.degrees(np.arctan2(iris_offset[1], iris_offset[0]))
    ellipse_max = eyeball_radius_to_iris_diameter_ratio * iris_radius
    ellipse_min = np.abs(ellipse_max * cos_phi * cos_theta)
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv.ellipse(gazemap, box=(iris_centre, (ellipse_min, ellipse_max), angle),
                         color=1.0, thickness=-1, lineType=cv.LINE_AA)
    gazemaps.append(gazemap)

    # Draw eyeball
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv.circle(gazemap, (ow_2, oh_2), r, color=1, thickness=-1)
    gazemaps.append(gazemap)

    return np.asarray(gazemaps)
