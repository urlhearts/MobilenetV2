import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
# Mobv2
from model import MobileNet
# Mobv3
from model_V3 import mobilenet_v3_large

import json

data_transform = transforms.Compose(
    [
        transforms.Resize(256),
        # 然后中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

img = Image.open('./11.jpg')
plt.imshow(img)
img = data_transform(img)
# 扩展在第0维度加1维度
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open('class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = mobilenet_v3_large(num_classes=5)

model_weight = './MobilenetV3_large.pth'
# unexpected_keys 都是不进行辅助分类的层
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight), strict=False)

model.eval()  # 关闭dropout方法
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(predict, class_indict[str(predict_cla)], predict[predict_cla].item())
# 这里显示的 9.8939e-01 是表示  9.8939 * 10 * -1次方   !!!
plt.show()
