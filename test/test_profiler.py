import pytest

import cv2
import torch
from torchvision.models import vgg16

from pyprof.profiler import Profiler

def test_profiler_sanity():

    with Profiler(device=0, output_file_path='/home/yotam/Desktop/output.html') as prof:
        image = cv2.imread('/home/yotam/Pictures/Lenna.png')
        image_opposite = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized= cv2.resize(image_opposite, dsize=(1024,1024))
        edge_map = cv2.Canny(image_opposite, 10, 200)
        sift_obj = cv2.SIFT_create()
        kps = sift_obj.detect(image_opposite)

        dummy_model = vgg16().to('cuda')
        tensor = torch.rand((96, 3, 224, 224), device='cuda')
        results = dummy_model(tensor)
        # numpy_array = results.detach().cpu().numpy()

    df = prof.get_recorded_data()
    assert len(df) < 200
