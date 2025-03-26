
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:


    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):

        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)


    @staticmethod
    def randomCrop(image, label):

        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label

    @staticmethod
    def randomContrast(image, label):
        random_factor = np.random.uniform(0.5, 1.5)  # 随机生成对比度增强因子，这里使用均匀分布来获取一个范围内的随机值
        new_image = ImageEnhance.Contrast(image).enhance(random_factor)  # 使用随机因子来增强图像的对比度
        new_label = ImageEnhance.Contrast(label).enhance(random_factor)  # 使用相同的随机因子来增强标签的对比度
        return new_image, new_label  # 返回增强后的图像和标签

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):

        def gaussianNoisy(im, mean=0.2, sigma=0.3):

            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.asarray(image)
        width, height = img.shape[:2]
        if len(img.shape) == 2:  # Check if grayscale
            img = gaussianNoisy(img.flatten(), mean, sigma)
            img = img.reshape([width, height])
        else:
            img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
            img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
            img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception:
        print(str(Exception))



def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=3):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomContrast": DataAugmentation.randomContrast,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image, new_label = funcMap[func_name](image, label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))


opsList = {"randomRotation", "randomContrast", "randomGaussian"}


def threadOPS(img_path, new_img_path, label_path, new_label_path):

    # img path
    print(img_path)
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
        print(img_names)
    else:
        img_names = [img_path]

    # label path
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]

    img_num = 0
    label_num = 0

    # img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1;
    # label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if os.path.isdir(tmp_label_name):
            print('contain file folder')
            exit()
        else:
            label_num = label_num + 1

    if img_num != label_num:
        print('the num of img and label is not equl')
        exit()
    else:
        num = img_num

    for i in range(num):
        img_name = img_names[i]

        label_name = label_names[i]


        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)

        print(tmp_img_name)
        image = DataAugmentation.openImage(tmp_img_name)

        label = DataAugmentation.openImage(tmp_label_name)

        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(target=imageOps,
                                                   args=(ops_name, image, label, new_img_path, new_label_path, img_name,
                                                         label_name))
            threadImage[_index].start()
            _index += 1
            time.sleep(5)

# Please modify the path
if __name__ == '__main__':
    threadOPS("dataset/OCTA500_6M/train/image", #set your path of training images
              "dataset/OCTA500_6M/training/image",
              "dataset/OCTA500_6M/train/label",# set your path of training labels
              "dataset/OCTA500_6M/training/label")
