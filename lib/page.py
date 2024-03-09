import cv2
import numpy as np
from typing import List, Literal

from lib.utils import bound_box

def divide_two_cols(image: np.ndarray, ignore=10):
    """divide two cols layout to left and right

    Args:
        image (np.ndarray): image of two-cols text area
        ignore (int, optional): ignore those hists that have black pixels less than the threshold. Defaults to 10.

    Returns:
        boundary lines of left and right cols
    """
    proj = np.sum(image, axis=0) // 255
    # plt.plot(proj)
    valley = np.where(proj < ignore)[0]
    
    # find the longest gap
    pointer = 0
    old_length = 0
    new_length = 0
    for (idx, value) in enumerate(np.diff(valley)):
        if value == 1:
            new_length += 1
        else:
            if new_length > old_length:
                pointer = idx
                old_length = new_length
                new_length = 0
    begin = pointer - old_length
    end = pointer

    l = valley[begin]
    r = valley[end]

    return l, r


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
    proj = np.sum(bold, axis=1) // 255
    # comb = find_local_minima(proj, threshold=ignore)
    # we always assert the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    proj[0], proj[-1] = 0,0
    valley = np.where(proj < ignore)[0]
    # print(valley)
    left_break_idx = np.where(np.diff(valley) != 1)[0]
    right_break_idx = left_break_idx + 1
    left_breaks = valley[left_break_idx]
    right_breaks = valley[right_break_idx]
    central_breaks = (left_breaks[1:] + right_breaks[:-1]) // 2
    # when cutting line, we use central breaks to cut more robustly
    comb = np.zeros(len(central_breaks)+2, dtype=int)
    comb[0], comb[1:-1], comb[-1] = 0, central_breaks, len(proj)-1

    # debug
    # plt.plot(proj)
    # plt.vlines(comb, 0, 255, colors='r')
    # plt.hlines(ignore, 0, len(proj), colors='g')
    # image_sep = image
    # image_sep[comb] = 255
    # plt.figure(figsize=(20,10))
    # plt.imshow(image_sep, cmap='gray')
    return comb

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
    # plt.plot(proj)
    # we always assert the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    proj[0], proj[-1] = 0,0
    valley = np.where(proj < ignore)[0]
    left_break_idx = np.where(np.diff(valley) != 1)[0]
    left_breaks = valley[left_break_idx]
    if left_breaks.size == 0: # if line is totally empty, return empty array
        return left_breaks
    # when cutting char we only use left breaks to cut evenly (otherwise the punctuation will be in a tiny slice, which will be misrecognized as a oversplit)

    # merge oversplit pieces (left-right structure is less connected horizontally may be overcut)
    delta = np.diff(left_breaks)
    comb = []
    span = 0
    for d in delta:
        span += d
        if span > 20: # 20 is the empirical width threshold for a character 
            comb.append(span)
            span = 0
    comb = np.cumsum(comb) + left_breaks[0]

    # debug
    # image_sep = image
    # image_sep[:, comb] = 255
    # plt.figure(figsize=(20,10))
    # plt.imshow(image_sep, cmap='gray')
    return comb


def line_type(right_line, left_line=None):
    # 由于文言文总是比白话文短，故左半行其实是多余的
    # 左有右无，小节标题
    comb = cut_char(right_line)
    if comb.size == 0:
        return 'title'
    # comb = cut_char(left_line)
    char_size = np.diff(comb).mean()
    indent = comb[0]
    # 左右皆缩进两空格，段首
    if indent >= 1.8 * char_size:
        return 'head' # 缩进两字符为段首
    # 否则为一般行
    else:
        return 'body'

def combing(image: np.ndarray, comb: np.ndarray, mode:Literal['h','v']) -> List[np.ndarray]:
    parts = []
    if mode == 'h':
        for i in range(len(comb)-1):
            parts.append(image[comb[i]:comb[i+1], :])
    elif mode == 'v':
        for i in range(len(comb)-1):
            parts.append(image[:, comb[i]:comb[i+1]])
    return parts

class Page:
    """Container for a page and its contents"""
    def __init__(self, title, text) -> None:
        self.title = title
        self.text = text

    def process(self):
        """Get useful statistics"""
        self.l, self.r = divide_two_cols(self.text)
        self.comb_line = cut_line(self.text, 10)
        self.num_lines = len(self.comb_line)
        self.line_type = []
        para_num = 0
        for yl, yr in zip(self.comb_line, self.comb_line[1:]):
            # left_line = self.text[yl:yr+1, 0:self.l]
            right_line = self.text[yl:yr+1, self.r+1:]
            match line_type(right_line):
                case 'title': self.line_type.append(0)
                case 'head':
                    para_num += 1
                    self.line_type.append(para_num)
                case 'body':
                    self.line_type.append(para_num)

    def prepare_data(self):
        """Prepare data for inference"""
        for yl, yr in zip(self.comb_line, self.comb_line[1:]):
            left_line = self.text[yl:yr+1, 0:self.l]
            right_line = self.text[yl:yr+1, self.r+1:]
            left_comb = cut_char(left_line)
            left_chars = combing(left_line, left_comb, 'v')
            right_comb = cut_char(right_line)
            right_chars = combing(right_line, right_comb, 'v')
            left_chars = [bound_box(char) for char in left_chars]
            right_chars = [bound_box(char) for char in right_chars]
            left_chars = np.asarray(left_chars)
            right_chars = np.asarray(right_chars)