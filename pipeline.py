import os
import numpy as np
import cv2
import torch
import ultralytics
from ultralytics import YOLO
from lib.resnet import ResNet
from lib.utils import *
from lib.page import Page

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(detect_model_path='./ckpt/layout-analysis/yolov8n_best.pt', recog_model_path='./ckpt/recognition/best.pt'):
    detect_model = YOLO('./ckpt/layout-analysis/yolov8n_best.pt')
    recog_model = ResNet(num_classes=12328,
                         c_in=1,
                         c_hidden=[16,32,64],
                         num_blocks=[3,3,3],
                         act_fn_name='relu',
                         block_name='PreActResNetBlock')
    recog_model.load_state_dict(torch.load('./ckpt/recognition/best.pt'))
    recog_model.to(device)
    recog_model.eval()
    return detect_model, recog_model

def load_data(data_path='./data/pages'):
    volumes_collection = []
    books = os.listdir(data_path)
    books.sort()
    for book in books:
        volumes = os.listdir(os.path.join(data_path, book))
        volumes.sort()
        volumes = [os.path.join(data_path, book, volume) for volume in volumes]
        volumes_collection.extend(volumes)
    return volumes_collection

def layout_analysis(model:YOLO, images: list[np.ndarray]):
    """detect and extract title, table, keep text for further processing and drop out header and etc.

    Args:
        images (np.ndarray):  a list of RGB images
    """
    images = [preprocessing(img) for img in images]
    inputs = [cv2.bitwise_not(img) for img in images]
    inputs = [np.repeat(img[:,:,None], 3, axis=-1) for img in inputs]
    la_results = model.predict(inputs, imgsz=(640,480), iou=0, agnostic_nms=True)
    pages = []
    for image, result in zip(images, la_results):
        boxes = result.boxes.cpu().numpy()
        # view(result.plot())
        cls = boxes.cls.astype('int')
        coords = boxes.xyxy.astype('int')
        # remove non text part and extract titles
        title_coords = coords[cls == 7]
        text_coords = coords[cls == 1]
        titles = []
        for coords in title_coords:
            titles.append(image[coords[1]:coords[3], coords[0]:coords[2]])
        # surrounding box between texts with tolerance of 20 pixels
        top_texts = text_coords[:,1].min() - 10
        bottom_texts = text_coords[:,3].max() + 20
        left_texts = text_coords[:,0].min() - 20
        right_texts = text_coords[:,2].max() + 20
        # remove any thing outside the box
        text_only = image[top_texts:bottom_texts, left_texts:right_texts]
        remove_underscore(text_only)
        # view(text_only)
        pages.append(Page(titles, text_only))
    return pages, la_results



def proc():
    detect_model, recog_model = load_model()

    volumes = load_data()
    for volume in volumes:
        pages = os.listdir(volume)
        pages = sorted(pages, key=lambda path: int(path.split('/')[-1]))
        pages.sort()
        pages = [os.path.join(volume, page) for page in pages]
        results = detect_model(pages)
        