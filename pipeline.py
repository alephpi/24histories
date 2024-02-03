import os
import numpy as np
import cv2
import torch
import ultralytics
from ultralytics import YOLO
from lib.resnet import ResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(detect_model_path='./ckpt/layout-analysis/yolov8n_best.pt', recog_model_path='./ckpt/recognition/best.pt'):
    detect_model = YOLO('./ckpt/layout-analysis/yolov8n_best.pt').to(device)
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

def layout_analysis(model, *image_paths):
    """detect and extract title, table, keep text for further processing and drop out header and etc.
    """
    results = model(image_paths)

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

def proc():
    detect_model, recog_model = load_model()

    volumes = load_data()
    for volume in volumes:
        pages = os.listdir(volume)
        pages = sorted(pages, key=lambda path: int(path.split('/')[-1]))
        pages.sort()
        pages = [os.path.join(volume, page) for page in pages]
        results = detect_model(pages)
        