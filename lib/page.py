import cv2
import numpy as np
from typing import List, Literal

from lib.utils import bound_box, find_local_minima, view
import matplotlib.pyplot as plt

def divide_two_cols(image: np.ndarray, ignore=50):
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
    arr = np.diff(valley)
    value = 1
    n = len(arr)
    max_len = 0
    start = 0
    end = 0
    curr_start = 0
    prev_value = None
    
    for i in range(n):
        # If the current element matches the target value
        if arr[i] == value:
            # If this is the first occurrence or the previous element is different
            if prev_value is None or arr[i-1] != value:
                curr_start = i
            
            # Update the maximum length and indices
            curr_len = i - curr_start + 1
            if curr_len > max_len:
                max_len = curr_len
                start = curr_start
                end = i
            
            prev_value = value
        else:
            prev_value = arr[i]

    l = valley[start]
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

    # char is square, so height should be the width of a char
    height, length = image.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)) 
    bold = cv2.dilate(image, kernel, iterations = 1) # 加粗使其更显著
    view(bold)
    # plt.imshow(bold, cmap='gray')
    proj = np.sum(bold, axis=0) // 255
    plt.figure()
    plt.xlim(150,200)
    plt.ylim(0,10)
    plt.plot(proj)
    print(list(zip(range(150,170),proj[150:170])))
    # we always assert the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    proj[0], proj[-1] = 0,0
    precut = find_local_minima(proj, ignore)
    # valley = np.where(proj < ignore)[0]
    # left_break_idx = np.where(np.diff(valley) != 1)[0]
    # left_breaks = valley[left_break_idx]
    plt.vlines(precut, 0, 10, colors='r')
    # plt.figure()
    # plt.plot(proj)
    if precut.size == 0: # if line is totally empty, return empty array
        return precut
    # when cutting char we only use left breaks to cut evenly (otherwise the punctuation will be in a tiny slice, which will be misrecognized as a oversplit)

    resegment_limit = 3
    segment_times = 0
    inc = height + 1 # initialize inc as line height, as height is close to the char width
    while True:
        print(segment_times)
        # scan the break from left to right
        cut = 0
        # print(inc)
        comb = []
        # print(length)
        # print(left_breaks[-5:])
        while cut < precut[-1]:
            idx = np.argmin(np.abs(precut - cut))
            # idx = np.searchsorted(left_breaks, cut)
            # try:
            #     dist_to_left_candidate, dist_to_right_candidate = np.abs(left_breaks[idx-1:idx+1] - cut)
            # except:
            #     print(left_breaks)
            #     print(left_)
            # if dist_to_left_candidate < dist_to_right_candidate:
            #     closest_break = left_breaks[idx-1]
            # else:
            #     closest_break = left_breaks[idx]
            # print(left_breaks[idx-1:idx+1])
            # print(cut)
            closest_break = precut[idx]
            comb.append(closest_break)
            cut = closest_break + inc
            
            # print(cut,closest_break)
        comb.append(length-1)
        char_size = np.mean(np.diff(comb)).round()
        segment_times += 1
        if segment_times > resegment_limit or inc == char_size:
            break
        inc = char_size
    
    comb = np.asarray(comb)

    # merge oversplit pieces (left-right structure is less connected horizontally may be overcut)
    # delta = np.diff(left_breaks)
    # comb = np.zeros(len(left_breaks)+2, dtype=int)
    # temp = []
    # span = 0
    # for d in delta:
    #     span += d
    #     if span > 0*char_size: 
    #         temp.append(span)
    #         span = 0
    # temp = np.cumsum(temp) + left_breaks[0]
    # comb = np.zeros(len(temp)+2, dtype=int)
    # comb[0], comb[1:-1], comb[-1] = 0, temp, len(proj)-1

    # print(comb)
    # debug
    image_sep = image.copy()
    # image_sep[:, left_breaks] = 255
    image_sep[:, comb] = 255
    plt.figure(figsize=(20,10))
    plt.imshow(image_sep, cmap='gray')
    return comb


def line_type(right_line, left_line=None):
    # 由于文言文总是比白话文短，故左半行其实是多余的
    # 左有右无，小节标题
    # comb = cut_char(right_line)
    proj = np.sum(right_line, axis=0) // 255
    # plt.figure()
    # plt.plot(proj)
    if np.sum(proj) == 0:
        return 'title'
    # comb = cut_char(left_line)
    char_size = right_line.shape[0]
    print(char_size)
    indent = np.argmax(proj > 5)
    # print(indent)
    # 左右皆缩进两空格，段首
    if indent >= 1 * char_size:
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
                case 'title': self.line_type.append(-1) # -1 represents title
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
            # view(left_line)
            view(right_line)
            # left_line is always shorter since it's in Fangsong
            left_line = bound_box(left_line)
            right_line = bound_box(right_line)
            left_comb = cut_char(left_line, 10)
            left_chars = combing(left_line, left_comb, 'v')
            right_comb = cut_char(right_line, 10)
            right_chars = combing(right_line, right_comb, 'v')
            left_chars = [bound_box(char) for char in left_chars]
            right_chars = [bound_box(char) for char in right_chars]
            # left_chars = np.asarray(left_chars)
            # right_chars = np.asarray(right_chars)
            break
        return left_chars, right_chars, left_line, right_line