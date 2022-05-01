import torch.nn as nn
import torch
from torchvision import transforms, datasets, utils  # 引入变换 数据集  和工具
import torch.optim as optim

import os
import json
import time

# v2 v3 你选
from model import MobileNet
from model_V3 import mobilenet_v3_large

device = torch.device('cpu')

data_transform = {
    # 当key为train 的时候我们就返回训练集
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随即裁剪
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  # 水平方向随即反转
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ]),
    'val': transforms.Compose([
        # 先缩放到256
        transforms.Resize(256),
        # 然后中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# data_root = os.path.abspath(os.path.join(os.getcwd(),'../..'))    获取根目录  os.getcwd 是获取当前路径

data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
image_path = data_root + '/data_flower'
train_dataset = datasets.ImageFolder(root=image_path + '/train',
                                     transform=data_transform['train']
                                     )

train_num = len(train_dataset)  # 我们训练集有多少张图片

flower_list = train_dataset.class_to_idx  # 获取train文件夹下花分类的索引
class_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(class_dict, indent=4)  # 打包成json的格式
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 这里如果是cpu  线程得是0
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
validate_dataset = datasets.ImageFolder(root=image_path + '/val',
                                        transform=data_transform['val']
                                        )
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(
    validate_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

images = iter(validate_loader)
image, label = images.next()

# def imshow(img):
#     img = img/2 +0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#
# imshow(utils.make_grid(image))

net = mobilenet_v3_large(num_classes=5)

model_weight_path = './mobilenet_v3_large_pre.pth'
# 判断是地址是否存在 如果存在往下执行 ，如果不存在 则报错   'file {} dose not exit.'
assert os.path.exists(model_weight_path), 'file {} dose not exit.'.format(model_weight_path)

pre_weights = torch.load(model_weight_path, map_location=device)

# delete features weights
pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
# pre_dict = {k: v for k, v in pre_weights.items() if 'classifer' not in k}

missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

# 冻结除了最后一层的全部参数
for param in net.features.parameters():
    param.requires_grad = False
net.to(device)

loss_function = nn.CrossEntropyLoss()
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

save_path = './MobilenetV3_large.pth'
best_acc = 0.0

for epoch in range(30):
    # train
    net.train()  # 启动dropout的方法
    running_loss = 0.0
    t1 = time.perf_counter()  # 查看训练一个epoch的时间  ！！！小技巧
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        output = net(images.to(device))
        loss = loss_function(output, labels.to(device))
        loss.backward()
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # item取出的精度更高
        rate = (step + 1) / len(train_loader)
        a = '*' * int(rate * 50)
        b = '.' * int((1 - rate) * 50)
        print('\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}'.format(int(rate * 100), a, b, loss), end='')
    print()
    print(time.perf_counter() - t1)  # 计算训练一个epoch的时间

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_image, test_labels = data_test
            outputs = net(test_image.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)  # 第一个参数是 所有的参数 ，第二个参数是路径
        print('[epoch %d]  train_loss : %.3f  test_accurate : %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))

print('Finished')
