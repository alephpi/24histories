from typing import Literal
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocessing(image: np.ndarray, invert=True) -> np.ndarray:
  """preprocessing function

  Args:
    image (np.ndarray): input image

  Returns:
    np.ndarray: return a denoised and binarized image with black background and white content
  """
  # your preprocessing code here
  assert len(image.shape) == 2, "Input image must be a grayscale image"
  thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  if invert:
    thresh = 255 - thresh
  return thresh

def cut_line(image: np.ndarray, ignore=10) -> np.ndarray:
    """cut block into lines based on the histogram projection

      Args:
          image (np.ndarray): image of block
          ignore (int, optional): ignore those hists that have black pixels less than the threshold. Defaults to 10.

      Returns:
          np.ndarray: the cutting indices
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    bold = cv2.dilate(image, kernel, iterations = 1) # 加粗使其更显著
    # plt.imshow(bold, cmap='gray')
    proj = np.sum(bold, axis=1) // 255
    plt.plot(proj)
    # we always assume the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    valley = np.where(proj < ignore)[0]
    left_break_idx = np.where(np.diff(valley) != 1)[0]
    right_break_idx = left_break_idx + 1
    left_breaks = valley[left_break_idx]
    right_breaks = valley[right_break_idx]
    central_breaks = (left_breaks[1:] + right_breaks[:-1]) // 2
    comb = np.zeros(len(central_breaks)+2, dtype=int)
    comb[0], comb[1:-1], comb[-1] = 0, central_breaks, len(proj)-1
    image_sep = bold
    image_sep[comb] = 255
    plt.figure(figsize=(20,10))
    plt.imshow(image_sep, cmap='gray')
    return comb

def remove_underscore(image: np.ndarray) -> np.ndarray:
    """remove underscore in image

    Args:
        image (np.ndarray): image

    Returns:
        np.ndarray: cleaned image
    """
    kernel = np.ones((1,40), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    proj = morphed.sum(1)
    bound = proj.nonzero()[0][0]
    return image[:bound]

def cut_char(image: np.ndarray, ignore=5) -> np.ndarray:
    """cut line into chars based on the histogram projection

      Args:
          image (np.ndarray): image of line
          ignore (int, optional): ignore those hists that have black pixels less than the threshold. Defaults to 10.

      Returns:
          np.ndarray: the cutting indices
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)) 
    bold = cv2.dilate(image, kernel, iterations = 1) # 加粗使其更显著
    # plt.imshow(bold, cmap='gray')
    proj = np.sum(bold, axis=0) // 255
    plt.plot(proj)
    # we always assume the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    valley = np.where(proj < ignore)[0]
    left_break_idx = np.where(np.diff(valley) != 1)[0]
    # right_break_idx = left_break_idx + 1
    left_breaks = valley[left_break_idx]
    # right_breaks = valley[right_break_idx]
    # central_breaks = (left_breaks[1:] + right_breaks[:-1]) // 2
    
    # merge oversplit pieces (due to the existence of underline we have to have a bigger ignore value to cut them, which also breaks character)
    # comb = left_breaks
    delta = np.diff(left_breaks)
    comb = []
    span = 0
    for d in delta:
        span += d
        if span > 20:
            comb.append(span)
            span = 0
    comb = np.cumsum(comb) + left_breaks[0]

    image_sep = bold
    image_sep[:, comb] = 255
    plt.figure(figsize=(20,10))
    plt.imshow(image_sep, cmap='gray')
    return comb
