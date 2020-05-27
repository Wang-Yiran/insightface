# coding:utf-8
from deploy import face_model
import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import shutil

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/Users/wangyiran/git/insightface/insightface/datasets/model-r50-am-lfw/model,0000', help='path to load model.')
parser.add_argument('--ga-model', default='/Users/wangyiran/git/insightface/insightface/datasets/gamodel-r50/model,0000', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--image_path', default='/Users/wangyiran/Documents/face_data_set_who', help='test image path')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold1', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
# parser.add_argument('--threshold2', default=0, type=int, help='threshold for those whose dist is above threshold1')
parser.add_argument('--threshold2', default=7, type=int, help='threshold for those whose dist is above threshold1')
args = parser.parse_args()

face_path = '/Users/wangyiran/Documents/face_data_set_gallery/face_gallery.npy'
label_path = '/Users/wangyiran/Documents/face_data_set_gallery/face_labels.npy'

faces = np.load(face_path)
labels = np.load(label_path)

testresult = open('/Users/wangyiran/Documents/face_recognition_result.txt','w')
def test_Faces(args):
    model = face_model.FaceModel(args)
    imgs = os.listdir(args.image_path)
    flag = False
    # 设置阈值，这两个阈值用来判定该人是否是库里的人
    a = zip([args.threshold1], [args.threshold2])
    for k, v in a:
        for img in imgs:
            if img.startswith('.'):
                continue
            pic = cv2.imread(os.path.join(args.image_path, img))
            pic = model.get_input(pic)

            if pic is None:
                continue
            else:
                f1 = model.get_feature(pic)
                for i in range(faces.shape[0]):
                    cnt = 0
                    for j in range(faces.shape[1]):
                        dist = np.sqrt(np.sum(np.square(f1 - faces[i][j])))
                        # 如果与库中某人距离小于阈值1
                        if dist < k:
                            cnt += 1
                    # 如果与库中某个人的相似度大于阈值2，则证明是该人
                    if cnt >= v:
                        name = labels[i]
                        testresult.writelines(os.path.join(args.image_path, img) + ' is ' + name + '\n')
                        print(os.path.join(args.image_path, img) + ' is ' + name)
                        flag = True
                if flag is False:
                    print("image:" + os.path.join(args.image_path, img) + ' is not in the gallery, refused!' + '\n')
    testresult.close()


if __name__ == '__main__':
    test_Faces(args)
