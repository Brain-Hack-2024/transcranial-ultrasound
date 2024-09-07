import cv2 as cv
import numpy as np

def process_and_create_flow(arr1, arr2, blur_radius=5, flow_threshold=0.75):
    """
    Compute the optical flow between two numpy arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        First array.
    arr2 : np.ndarray
        Second array, same size as arr1.
    blur_radius : int
        Blur radius for the Gaussian filter during preprocessing (to remove background noise).
    flow_threshold : float
        Threshold for which pixels are considered part of optical flow.

    Returns
    -------
    np.ndarray
        Optical flow array. Shape: (arr1.shape[0], arr1.shape[1], 2).
        Contains the x and y components of the optical flow for each pixel.
    """
    def process_arr(arr):
        arr_copy = np.copy(arr)
        arr_copy = cv.GaussianBlur(np.abs(arr_copy),(blur_radius,blur_radius),0)
        arr_copy = arr_copy / np.max(arr_copy)
        arr_copy = arr_copy * (arr_copy > flow_threshold)
        arr_copy = (arr_copy*255).astype(np.uint8)
        return arr_copy

    return cv.calcOpticalFlowFarneback(process_arr(arr1), process_arr(arr2), None, 0.5, 3, 15, 3, 5, 1.2, 0)