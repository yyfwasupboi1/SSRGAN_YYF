#!/usr/bin/env python
# title           :Utils.py
# description     :Have helper functions to process images and plot images
# author          :Deepak Birla
# date            :2018/10/30
# usage           :imported in other files
# python_version  :3.5.4

from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import scipy
import cv2
import os
import sys

import matplotlib.pyplot as plt

#plt.switch_backend('agg')
plt.switch_backend('TkAgg')

#####################################
def imshow(title, array):
    import matplotlib.pyplot as plt
    plt.set_cmap('gray')#gray
    #plt.title(title)
    plt.imshow(array, interpolation='none')
    plt.show()
#####################################

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)
####################################################################################################

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr

# def lr_images(images):
#     images_lr = array(images)
#     return images_lr
####################################################################################################
#Takes list of images and provide LR images in form of numpy array
def lr_images(images_real, downscale):
    images = []
    for img in range(len(images_real)):
        images.append(imresize(images_real[img], [images_real[img].shape[0] // downscale,
                                                  images_real[img].shape[1] // downscale],
                               interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr

#####################################################################################################
def normalize(input_data):
    print("1866666")
    #return (input_data.astype(np.float32) - 127.5) / 127.5
    return (input_data.astype(np.uint8) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def load_data_from_dirs(dirs, ext):  # 遍历文件夹里的文件
    files = []#首先命名一个空的列表名字是files
    file_names = []#再命名一个列表名字是file_names
    count = 0#从0开始计数
    for d in dirs:#遍历文件夹中的文件
        for f in os.listdir(d):#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            if f.endswith(ext):
                # image = cv2.imread(os.path.join(d,f))#这里使用CV2进行图像的读取，保证图像的通道数是3，skimage读出来是单通的
                # cv2.imread第二个参数为0指定读取出来就是灰度图
                image = cv2.imread(os.path.join(d, f), 0)
                # # 灰度图只有H,W 二维数据，需要扩充为3维度【H,W】->【H,W,C】此时C 为1
                # image = np.expand_dims(image, axis=2)
                # image = data.imread(os.path.join(d,f))
                # image = np.reshape(image,(image.shape[0],image.shape[1],1))
                print("sssssss", image.shape)
                # 先用2维度灰度图，lr_images函数中恢复到三维
                if len(image.shape) == 2:
                    files.append(image)
                    file_names.append(os.path.join(d, f))
                count = count + 1
    return files


def load_data(directory, ext):
    print("1?")
    files = load_data_from_dirs(load_path(directory), ext)
    return files


def load_training_data(directory, ext, number_of_images=1000, train_test_ratio=0.8):  # 数量默认设置为1000

    number_of_train_images = int(number_of_images * train_test_ratio)
    print("88888888888888")
    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    test_array = array(files)
    if len(test_array.shape) < 3:
        print("Images are of not same shape")
        print("Please provide same shape images")
        sys.exit()

    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]

    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)

    x_train_lr = lr_images(x_train, 4)
    x_train_lr = normalize(x_train_lr)

    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(x_test, 4)
    x_test_lr = normalize(x_test_lr)

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_test_data_for_model(directory, ext, number_of_images=100):  # 这里改变图片喂入的数量 默认为100

    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)
    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr, x_test_hr


def load_test_data(directory, ext, number_of_images=100):  # 这里改变图片喂入的数量 默认为100
    print("first_step")
    files = load_data_from_dirs(load_path(directory), ext)#此处调用文件加载函数，加载目标文件中的图像
    print('files,find it,finally', files)
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        print("len:", str(len(files)))
        sys.exit()

    x_test_lr = lr_images(files)  # 直接当做lr
    print("x_test_lr:", x_test_lr.shape)
    x_test_lr = normalize(x_test_lr)
    print("x_test_lr2:", x_test_lr.shape)

    return x_test_lr

######################################################
#重建图像测评指标

from keras import backend as k

def PSNRLossnp(y_true, y_pred):
    return 10 * np.log(255 * 2 / (np.mean(np.square(y_pred - y_true))))


def SSIM(y_true, y_pred):
    u_true = k.mean(y_true)
    u_pred = k.mean(y_pred)
    var_true = k.var(y_true)
    var_pred = k.var(y_pred)
    std_true = k.sqrt(var_true)
    std_pred = k.sqrt(var_pred)
    c1 = k.square(0.01 * 7)
    c2 = k.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


def PSNRLoss(y_true, y_pred):
    return 10 * k.log(255 ** 2 / (k.mean(k.square(y_pred - y_true))))

######################################################



# While training save generated image(in form LR, SR, HR)
# Save only one image as sample
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3),
                          figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    print(examples)
    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    print("11111Reconstruction Gain:", PSNRLossnp(image_batch_hr, gen_img))############20201203

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    # plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.imshow(np.squeeze(image_batch_lr[value], axis=2), interpolation='nearest', cmap=plt.cm.gray)
    #plt.imshow(np.squeeze(image_batch_lr[value], axis=2), interpolation='nearest')
    plt.axis('off')  ######

    plt.subplot(dim[0], dim[1], 2)
    # plt.imshow(generated_image[value], interpolation='nearest')
    plt.imshow(np.squeeze(generated_image[value], axis=2), interpolation='nearest',cmap=plt.cm.gray)
    #plt.imshow(np.squeeze(generated_image[value], axis=2), interpolation='nearest')
    plt.axis('off')  ######

    plt.subplot(dim[0], dim[1], 3)
    # plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.imshow(np.squeeze(image_batch_hr[value], axis=2), interpolation='nearest', cmap=plt.cm.gray)
    #plt.imshow(np.squeeze(image_batch_hr[value], axis=2), interpolation='nearest')
    plt.axis('off')  ######

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.jpg' % epoch)

    plt.show()


# Plots and save generated images(in form LR, SR, HR) from model to test the model 从模型中绘制并保存生成的图像(以LR、SR、HR的形式)，以测试模型
# Save output for all images given for testing 保存输出的所有图像给出的测试
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr, dim=(1, 3),
                                         figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    print("11image_batch_lr shape:", image_batch_lr.shape)
    imshow('33333333', image_batch_lr[0, :, :, 0])  ##################
    gen_img = generator.predict(image_batch_lr)
    print("11gen_img shape:", gen_img.shape)
    imshow('33333333', gen_img[0, :, :, 0])  ##################
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)



    for index in range(examples):
        plt.figure(figsize=figsize)
        print("kkkkkkkkkkk",image_batch_lr.shape)
        plt.subplot(dim[0], dim[1], 1)
        # plt.imshow(image_batch_lr[index], interpolation='nearest')
        plt.imshow(np.squeeze(image_batch_lr[index], axis=2), interpolation='nearest', cmap=plt.cm.gray)

        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        # plt.imshow(generated_image[index], interpolation='nearest')
        plt.imshow(np.squeeze(generated_image[index], axis=2), interpolation='nearest',
                   cmap=plt.cm.gray)
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        # plt.imshow(image_batch_hr[index], interpolation='nearest')
        plt.imshow(np.squeeze(image_batch_hr[index], axis=2), interpolation='nearest', cmap=plt.cm.gray)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.jpg' % index)

        plt.show()


# Takes LR images and save respective HR images 获取LR图像并保存相应的HR图像
def plot_test_generated_images(output_dir, generator, x_test_lr, figsize=(5, 5)):#（5，5）

    examples = x_test_lr.shape[0]
    #image_batch_lr = denormalize(x_test_lr)#原
    #x_test_lr = denormalize(x_test_lr)
    image_batch_lr = x_test_lr  ################################就是这个地方没有改
    # import sys#到这停
    # sys.exit()
    print("22image_batch_lr shape:", image_batch_lr.shape)  #
    imshow('11111111', image_batch_lr[0, :, :, 0])  #############
    gen_img = generator.predict(image_batch_lr)
    print("22gen_img shape:", gen_img.shape)  #

    imshow('33333333', gen_img[0, :, :, 0])  #############
    generated_image = denormalize(gen_img)
    imshow('33333333', generated_image[0, :, :, 0])  #######
    #############################################
    new_gen_img = np.reshape(gen_img, (gen_img.shape[1], gen_img.shape[2]))
    print("looklook", new_gen_img.shape)
    print("looklook", new_gen_img.dtype)
    cv2.imwrite("./predict.jpg", cv2.normalize(new_gen_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    #############################################
    for index in range(examples):

        plt.figure(figsize=figsize)

        #plt.imshow(generated_image[index], interpolation='nearest')
        #plt.imshow(generated_image[index],cmap = plt.get_cmap('gray'),vmin = 0,vmax = 255)#尝试设置一下颜色浮动的范围
        plt.imshow(np.squeeze(generated_image[index], axis=2), interpolation='nearest', cmap=plt.cm.gray)
        plt.axis('off')#off

        plt.tight_layout()
        plt.savefig(output_dir + 'high_res_result_image_%d.jpg' % index)

        #plt.show()#弹出窗口展示生成的图像





