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
import random
import time
# import itertools

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
# 这个比较重要，人脸识别的model，我的经验是要用绝对路径
parser.add_argument('--model', default='/Users/wangyiran/git/insightface/insightface/datasets/model-r50-am-lfw/model,0000', help='path to load model.')
# 这个比较重要，年龄性别的model，我的经验是要用绝对路径
parser.add_argument('--ga-model', default='/Users/wangyiran/git/insightface/insightface/datasets/gamodel-r50/model,0000', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

## 该方法用于根据模型检测同一个文件夹中的图片是否属于同一张人脸
def test_all():
    im_path = '/Users/wangyiran/Documents/face_data_set'
    file = open('/Users/wangyiran/Documents/bad_case.txt', 'a')
    # 人脸检测
    folders = os.listdir(im_path)
    cnt = 0
    name_list = []
    for folder in folders:
        # 跳过mac隐藏文件
        if folder.startswith('.'):
            continue
        random.seed(time.time() * 100000 % 10000)
        sublist = []
        imgs = os.listdir(os.path.join(im_path, folder))
        random.shuffle(imgs)
        for i in range(0, len(imgs), 2):
            # 跳过mac隐藏文件
            if imgs[i].startswith("."):
                continue
            try:
                model = face_model.FaceModel(args)
                pic_path1 = os.path.join(im_path, folder, imgs[i])
                img1 = cv2.imread(pic_path1)
                img1 = model.get_input(img1)
                f1 = model.get_feature(img1)

                pic_path2 = os.path.join(im_path, folder, imgs[i+1])
                img2 = cv2.imread(pic_path2)
                img2 = model.get_input(img2)
                f2 = model.get_feature(img2)

                dist = np.sum(np.square(f1-f2))
                if dist < args.threshold:
                    print('they are the same one')
                else:
                    # pdb.set_trace()
                    bad_case_folder = '/Users/wangyiran/Documents/face_data_set_result/' + str(cnt)
                    os.mkdir(bad_case_folder)
                    shutil.copy(pic_path1, bad_case_folder)
                    shutil.copy(pic_path2, bad_case_folder)
                    cnt += 1
                    file.writelines(pic_path1 + ',' + pic_path2+ ',' + str(dist) + ',' + str(args.threshold) + '\n')
                    print('they are the different one')
                print(dist)

            except:
                continue
    file.close()

def test():
    model = face_model.FaceModel(args)
    for i in range(1):
        cnt = 0
        out_txt = open('/media/dhao/系统/05-weiwei/FR/dataset/test_result/combine_test/1.38/result' +str(i+1) + '.txt', 'a')
        file = open('/media/dhao/系统/05-weiwei/FR/politician_code/same_pairs.txt', 'r')
        lines = file.readlines()
        for j in range(len(lines)):
            name1 = lines[j].strip().split(',')[0]
            name2 = lines[j].strip().split(',')[1]
            try:
                
                img1 = cv2.imread(name1)
                img1 = model.get_input(img1)
                f1 = model.get_feature(img1)

                img2 = cv2.imread(name2)
                img2 = model.get_input(img2)
                f2 = model.get_feature(img2)

                dist = np.sqrt(np.sum(np.square(f1 - f2)))

                if dist < args.threshold:
                    # print('they are the same one')
                    out_txt.writelines(name1 + ',' + name2 + ',' + str(dist) + ',' + str(args.threshold) + '*' *3 + str(1) + '\n')

                else:
                    out_txt.writelines(name1 + ',' + name2 + ',' + str(dist) + ',' + str(args.threshold) + '*' * 3 + str(0) + '\n')
                    bad_case_folder = '/media/dhao/系统/05-weiwei/FR/dataset/test_result/combine_test/1.38/all' + str(i) + '/' + str(cnt) + '/'
                    if not os.path.exists(bad_case_folder):
                        os.mkdirs(bad_case_folder)
                    if os.path.isdir(bad_case_folder):
                        shutil.copy(name1, bad_case_folder)
                        shutil.copy(name2, bad_case_folder)
                        cnt += 1
                    else:
                        sys.exit(0)

            except:
                continue

        file.close()
        out_txt.close()


def compute_sim():
    model = face_model.FaceModel(args)
    out_txt = open('/media/dhao/系统/05-weiwei/FR/politician_code/result_dist.txt', 'a')
    file = open('/media/dhao/系统/05-weiwei/FR/politician_code/same_pairs.txt', 'r')
    lines = file.readlines()
    for j in range(len(lines)):
        name1 = lines[j].strip().split(',')[0]
        name2 = lines[j].strip().split(',')[1]
        try:
            img1 = cv2.imread(name1)
            img1 = model.get_input(img1)
            f1 = model.get_feature(img1)

            img2 = cv2.imread(name2)
            img2 = model.get_input(img2)
            f2 = model.get_feature(img2)

            dist = np.sqrt(np.sum(np.square(f1 - f2)))
            out_txt.writelines(name1 + ',' + name2 + ',' + ',' + str(dist) + '\n')
        except:
            continue

    file.close()
    out_txt.close()


def face_refine():
    im_path = '/Users/wangyiran/Documents/face_data_set'
    result_path = '/Users/wangyiran/Documents/face_data_set_result'
    # 人脸检测
    folders = os.listdir(im_path)
    cnt = 0
    name_list = []
    for folder in folders:
        # 跳过mac隐藏文件
        if folder.startswith('.'):
            continue
        random.seed(time.time() * 100000 % 10000)
        # 在结果集中建立文件夹
        result_dir = os.path.join(result_path, folder)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        sublist = []
        imgs = os.listdir(os.path.join(im_path, folder))
        random.shuffle(imgs)
        for i in range(0, len(imgs), 1):
            # 跳过mac隐藏文件
            if imgs[i].startswith("."):
                continue
            try:
                model = face_model.FaceModel(args)
                pic_path1 = os.path.join(im_path, folder, imgs[i])
                img1 = cv2.imread(pic_path1)
                img1 = model.get_input(img1)
                # f1 = model.get_feature(img1)
                if type(img1) == np.ndarray:
                    if not os.path.exists(os.path.join(result_path, folder)):
                        os.mkdir(os.path.join(result_path, folder))
                    cv2.imwrite(os.path.join(result_path, folder, imgs[i]), np.transpose(img1, (1, 2, 0))[:, :, ::-1])

            except:
                print('错误：文件名:' + folder + '/' + str(imgs[i]))
                continue


if __name__ == '__main__':
    # 测试所有图片是否是同一个人
    # test_all()
    # test()
    # compute_sim()
    face_refine()

