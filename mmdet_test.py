from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pycocotools._mask as _mask
import json

config_file = '/content/mmdetection/work_dirs/nuclei/nuclei.py'
checkpoint_file = '/content/mmdetection/work_dirs/nuclei/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

coco = COCO('/content/dataset/test/test.json')

def detect(img_id, bbox_conf, mask):
    out = {}
    out['image_id'] = int(img_id)

    xmin, ymin, xmax, ymax = bbox_conf[0:4]
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    out['bbox'] = [x, y, w, h]
    out['bbox'] = [float(f) for f in out['bbox']]

    out['score'] = float(bbox_conf[4])
    out['category_id'] = 1

    mask = np.asfortranarray(mask)
    h, w = mask.shape 
    out['segmentation'] = _mask.encode(mask.reshape((h, w, 1), order='F'))[0]
    out['segmentation']['counts'] = out['segmentation']['counts'].decode('utf-8')
    
    return out


imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])

output = []
for imgId in imgIds:
    img_info = coco.loadImgs(imgId)[0]
    name = img_info['file_name']
    id = img_info['id']
    print(name, id)
    img = '/content/dataset/test/' + name
    result = inference_detector(model, img)
    
    pred_num = len(result[0][0])
    for i in range(pred_num):
        output.append(detect(id, result[0][0][i], result[1][0][i]))

with open('answer.json', 'w') as f:
    f.write(json.dumps(output))
