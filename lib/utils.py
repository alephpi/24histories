from typing import Literal, List
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocessing(image: np.ndarray) -> np.ndarray:
    """preprocessing function

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: return a denoised, binarized, adjusted and underscore free image with black background and white content
    """
    # your preprocessing code here
    assert len(image.shape) == 2, "Input image must be a grayscale image"
    processed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed = cv2.bitwise_not(processed)
    processed = adjust_skew(processed, True)
    processed = remove_underscore(processed, False)

    return processed


def adjust_skew(image: np.ndarray, debug=False) -> np.ndarray:
    """correct skew image"""
    center, angle = find_min_area_rect(image, debug)
    Mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    h,w = image.shape
    # h, w = cropped.shape
    # print(h, w)
    # fill the border with black in consistent with black background
    rotated = cv2.warpAffine(image, Mat, (w,h), flags=cv2.INTER_CUBIC, borderValue=(0,0,0))
    if debug:
        view(rotated)
    return rotated 

def detect_header_line(image: np.ndarray) -> np.ndarray:
    '''detect the header lines
    '''

    height, width = image.shape
    kernel = np.ones((1, 100), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    angle_range = [85,95]

    # 边缘检测
    # edges = cv2.Canny(closed[:int(0.2*height),:], 50, 200, None, 3)

    # 方法1.霍夫变换检测直线

    # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100, min_theta=angle_range[0]*np.pi/180, max_theta=angle_range[1]*np.pi/180)
    # # 绘制检测到的线
    # if lines is not None:
    #     for line in lines:
    #         rho, theta = line[0]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    # # 方法2.概率霍夫变换检测直线
    # linesP = cv2.HoughLinesP(edges, 1, np.pi / 1000, threshold=50,
    #                         #  minLineLength=int(0.3*width), maxLineGap=10
    #                          )

    # if linesP is not None:
    #     left_ends = []
    #     right_ends = []
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         # print(l)
    #         left_ends.append((l[0], l[1]))
    #         right_ends.append((l[2], l[3]))
    #     left_ends = sorted(left_ends)
    #     right_ends = sorted(right_ends)
    #     left_end = left_ends[0]
    #     right_end = right_ends[-1]
    #     # cv2.line(image, left_end, right_end, (0,0,255), 1, cv2.LINE_AA)
    

        # # 计算原始直线的斜率
        # m = (np.arctan((left_end[1]-right_end[1])/(left_end[0]-right_end[0])))

        # # 计算旋转角度
        # theta = np.arctan(m)

        # # 构建仿射变换矩阵
        # print(left_end)
        # center = (int(left_end[0]), int(left_end[1]))
        # rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(theta), scale=1)

        # # 应用仿射变换到整张图像上
        # rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # return rotated_image

    return image 


# adapt from https://zhuanlan.zhihu.com/p/81341622
def find_min_area_rect(image: np.ndarray, debug=False) -> np.ndarray:
    '''find minimal bounding rectangle'''
    height, width = image.shape

    # denoising
    kernel_size = int(51*width/1600) # kernel size adapt to image size, 51 for width 1600
    if kernel_size % 2 == 0: kernel_size += 1 # assert odd as Gaussian kernel
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    whereid = np.where(thresh > 0)
    # 交换横纵坐标的顺序，否则下面得到的每个像素点为(y,x)
    whereid = whereid[::-1]
    # 将像素点格式转换为(n_coords, 2)，每个点表示为(x,y)
    coords = np.column_stack(whereid)
    (x,y), (w,h), angle = cv2.minAreaRect(coords)
    if angle < -45:
        angle = 90 + angle
    # center = (width//2, height//2)
    center = (int(x), int(y))

    if debug:
        vis = image.copy()
        box = cv2.boxPoints(((x,y), (w,h), angle))
        box = np.intp(box)
        x, y, w, h = cv2.boundingRect(box)
        cv2.drawContours(vis,[box],0,(255,255,255),2)
        view(vis)

    return center, angle

def remove_underscore(image: np.ndarray, debug=False) -> np.ndarray:
    """remove underscore in image

    Args:
        image (np.ndarray): image

    Returns:
        np.ndarray: cleaned image
    """
    height, width = image.shape
    # detect underscore with a very thin and long kernel
    kernel_size = int(40*width/1600) # kernel size adapt to image size, 40 for width 1600
    kernel = np.ones((1,kernel_size), np.uint8)
    underscore = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # remove underscore (but may contain horizontal strokes with low intensity)
    image -= underscore
    # zeroing those with high intensity (true underscore)
    image[underscore > 127] = 0
    if debug:
        view(underscore)
    return image

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
    # plt.plot(proj)
    # we always assume the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    valley = np.where(proj < ignore)[0]
    left_break_idx = np.where(np.diff(valley) != 1)[0]
    right_break_idx = left_break_idx + 1
    left_breaks = valley[left_break_idx]
    right_breaks = valley[right_break_idx]
    central_breaks = (left_breaks[1:] + right_breaks[:-1]) // 2
    # when cutting line, we use central breaks to cut more robustly
    comb = np.zeros(len(central_breaks)+2, dtype=int)
    comb[0], comb[1:-1], comb[-1] = 0, central_breaks, len(proj)-1

    # debug
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
    plt.plot(proj)
    # we always assume the margin is white, i.e proj[0]==0, proj[-1]==0 to simplify the case
    valley = np.where(proj < ignore)[0]
    left_break_idx = np.where(np.diff(valley) != 1)[0]
    left_breaks = valley[left_break_idx]
    # when cutting char we only use left breaks to cut evenly (otherwise the punctuation will be in a tiny slice, which will be misrecognized as a oversplit)

    # merge oversplit pieces (left-right structure is less connected horizontally may be overcut)
    delta = np.diff(left_breaks)
    comb = []
    span = 0
    for d in delta:
        span += d
        if span > 20:
            comb.append(span)
            span = 0
    comb = np.cumsum(comb) + left_breaks[0]

    # debug
    # image_sep = image
    # image_sep[:, comb] = 255
    # plt.figure(figsize=(20,10))
    # plt.imshow(image_sep, cmap='gray')
    return comb

def combing(image: np.ndarray, comb: np.ndarray, mode:Literal['h','v']) -> List[np.ndarray]:
    parts = []
    if mode == 'h':
        for i in range(len(comb)-1):
            parts.append(image[comb[i]:comb[i+1], :])
    elif mode == 'v':
        for i in range(len(comb)-1):
            parts.append(image[:, comb[i]:comb[i+1]])
    return parts

def bound_box(image: np.ndarray) -> np.ndarray:
    """
    Find the contour bound box of the image.

    Args:
        image (np.ndarray): the input image.

    Returns:
        np.ndarray: the bounded image
    """
    nonzero_indices = np.nonzero(image)

    # Get bounding box coordinates
    min_x, min_y = np.min(nonzero_indices[1]), np.min(nonzero_indices[0])
    max_x, max_y = np.max(nonzero_indices[1]), np.max(nonzero_indices[0])
    return image[min_y:max_y, min_x:max_x]

def resize(image: np.ndarray):
    return cv2.resize(image, (64,64))

def view(image: np.ndarray):
    plt.figure(figsize=(20,15))
    plt.imshow(image, cmap='gray')