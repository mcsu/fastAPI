import base64
import io
import re
import numpy as np

from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from captcha_cnn_model import CNN

from captcha_setting import ALL_CHAR_SET, ALL_CHAR_SET_LEN


def base64_to_tensor(data):  # 将base64编码转换为张量并升至4维
    try:
        base64_data = re.sub('^data:image/.+;base64,', '', data)
        byte_data = base64.b64decode(base64_data)
        image_data = io.BytesIO(byte_data)
        img = Image.open(image_data)
        img = img.convert('L')
        trans = transforms.ToTensor()(img)
        trans = torch.unsqueeze(trans, 1)  # 升维
        return trans
    except ValueError:
        return "ERR"


def predict(tensor_data):  # 预测张量图像
    try:
        cnn = CNN()
        cnn.eval()
        cnn.load_state_dict(torch.load('model.pkl'))
        v_image = Variable(tensor_data)
        predict_label = cnn(v_image)

        c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
        c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
        c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
        c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        # print(c)
        return c
    except ValueError:
        return "ERR"


def run(base64_data):
    try:
        tensor_data = base64_to_tensor(base64_data)
        result = predict(tensor_data)
        return result
    except TypeError:
        return 'TypeError'


