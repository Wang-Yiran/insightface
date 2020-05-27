from deploy import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--model', default='/Users/wangyiran/git/insightface/insightface/datasets/model-r50-am-lfw/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='/Users/wangyiran/git/insightface/insightface/datasets/gamodel-r50/model,0', help='path to load model.')
# parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

# 加载model
model = face_model.FaceModel(args)
# 读取图片
# img = cv2.imread('Tom_Hanks_54745.png')
# img = cv2.imread('/Users/wangyiran/Documents/face_data_set/范冰冰/145.jpg')
img = cv2.imread('/Users/wangyiran/Desktop/WechatIMG2100.jpeg')
# img = cv2.imread('/Users/wangyiran/git/insightface/insightface/deploy/pic2.jpg')
# 模型加载图片
img = model.get_input(img)
# 获得特征
f1 = model.get_feature(img)
# 输出特征
print('特征值: ', f1[0:10])
gender, age = model.get_ga(img)
print('性别', gender)
print('年龄', age)
# sys.exit(0)
# img = cv2.imread('Tom_Hanks_54745.png')
# img = cv2.imread('/Users/wangyiran/git/crawlpic/pic/pic34.jpg')
img = cv2.imread('/Users/wangyiran/Desktop/WechatIMG2092.jpeg')
img = model.get_input(img)
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print('距离: ', dist)
sim = np.dot(f1, f2.T)
print('点乘角度距离: ', sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
