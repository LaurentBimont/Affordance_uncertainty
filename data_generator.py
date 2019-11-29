import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy import misc
import scipy.io
import skimage.transform as trans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

im = misc.imread('myers/part-affordance-dataset/tools/bowl_01/bowl_01_00000001_rgb.jpg')
lab = scipy.io.loadmat('myers/part-affordance-dataset/tools/bowl_01/bowl_01_00000001_label.mat')['gt_label']
lab_comp = scipy.io.loadmat('myers/part-affordance-dataset/tools/bowl_01/bowl_01_00000001_label_rank.mat')['gt_label']

# Création des données labellisées
def create_folder(training=True):
    ''' Put Image in myers/training/image and Label in myers/training/label '''
    rule = open('myers/part-affordance-dataset/category_split.txt', 'r')
    lines = rule.readlines()
    num = 0
    cur_type = ''
    for line in lines:
        print(line)
        line = line.rstrip('\n').split(' ')
        if training:
            if line[0] == '1':
                directory = 'myers/part-affordance-dataset/tools/{}'.format(line[1])
                for file in os.listdir(directory):
                    filename = os.fsdecode(file)
                    if filename.endswith(".jpg"):
                        im = misc.imread(directory+'/'+filename)
                        lab = scipy.io.loadmat(directory+'/'+filename.replace('_rgb.jpg', '_label.mat'))['gt_label']
                        new_lab = np.zeros((lab.shape[0], lab.shape[1], 3), dtype=np.uint8)
                        new_lab[:, :, 2], new_lab[:, :, 1], new_lab[:, :, 0] = lab, lab, lab
                        misc.toimage(im, cmin=0.0, cmax=...).save('myers/training/image/im_{}.png'.format(num))
                        misc.toimage(new_lab, cmin=0.0, cmax=...).save('myers/training/label/im_{}.png'.format(num))
                        # np.save('myers/training/image/img_{}.npy'.format(im_num), im)
                        # np.save('myers/training/label/lab_{}.npy'.format(lab_num), lab)
                        num += 1
        else:
            if line[0] == '2':
                if cur_type != line[1]:
                    cur_type = line[1]
                    directory = 'myers/part-affordance-dataset/tools/{}'.format(line[1])
                    for file in os.listdir(directory):
                        filename = os.fsdecode(file)
                        if filename.endswith(".jpg"):
                            im = misc.imread(directory+'/'+filename)
                            lab = scipy.io.loadmat(directory+'/'+filename.replace('_rgb.jpg', '_label.mat'))['gt_label']
                            new_lab = np.zeros((lab.shape[0], lab.shape[1], 3), dtype=np.uint8)
                            new_lab[:, :, 2], new_lab[:, :, 1], new_lab[:, :, 0] = lab, lab, lab
                            misc.toimage(im, cmin=0.0, cmax=...).save('myers/testing/image/im_{}.png'.format(num))
                            misc.toimage(new_lab, cmin=0.0, cmax=...).save('myers/testing/label/im_{}.png'.format(num))
                            # np.save('myers/training/image/img_{}.npy'.format(im_num), im)
                            # np.save('myers/training/label/lab_{}.npy'.format(lab_num), lab)
                            num += 1
                        break
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(lab)
    plt.show()
    return

def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        # print(mask.shape)
        # mask = mask[:, :, :, 2] if (len(mask.shape) == 4) else mask[:, :, 0]
        # new_mask = np.zeros(mask.shape + (num_class,))
        # for i in range(num_class):
        #     new_mask[mask == i, i] = 1
        # new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1], new_mask.shape[2],
        #                                  new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
        # new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        # mask = new_mask.astype(np.int)
        # np.save('my_mask', mask)

    elif (np.max(img) > 1):
        img = img / 255
        # mask = mask / 255
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
    return img, mask

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "rgb",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=True, num_class=8, save_to_dir='myers/test', target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

def testGenerator(test_path, num_image=11, target_size=(256, 256), flag_multi_class=True, as_gray=True):
    for i in range(num_image):
        filename_im = 'image/im_{}.jpg'.format(i)
        img = misc.imread(test_path + '/' + filename_im)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img

def testlabelGenerator(test_path, num_image, target_size=(256, 256)):
    for i in range(num_image):
        filename_lab = 'label/lab_{}.jpg'.format(i)
        label = misc.imread(test_path + '/' + filename_lab)
        # label = trans.resize(label, target_size)
        yield label

def weight_creation(generator):
    weights = np.zeros((8))
    for im, lab in generator:
        for idx in range(weights.shape[0]):
            weights[idx] += np.sum(np.where(lab==idx, 1, 0))
    return weights

grasp = [128, 128, 128]  # Gris
cut = [128, 0, 0]  # Bordeau
scoop = [192, 192, 128]  # Vert peau
contain = [128, 64, 128]  # Violet
pound = [60, 40, 222]  # Bleu
support = [128, 128, 0]  # Vert terre
wrapgrasp = [192, 128, 128]  # Rose
unlabelled = [0, 0, 0]  # Noir

color_dict = np.array([unlabelled, grasp, cut, scoop, contain, pound, support, wrapgrasp])

def plot_legend_color():

    grasp_patch = mpatches.Patch(color=np.array(grasp)/255, label='grasp')
    cut_patch = mpatches.Patch(color=np.array(cut)/255, label='cut')
    scoop_patch = mpatches.Patch(color=np.array(scoop)/255, label='scoop')
    contain_patch = mpatches.Patch(color=np.array(contain)/255, label='contain')
    pound_patch = mpatches.Patch(color=np.array(pound)/255, label='pound')
    support_patch = mpatches.Patch(color=np.array(support)/255, label='support')
    wrapgrasp_patch = mpatches.Patch(color=np.array(wrapgrasp)/255, label='wrapgrasp')
    unlabelled_patch = mpatches.Patch(color=np.array(unlabelled)/255, label='unlabelled')

    plt.legend(handles=[grasp_patch, cut_patch,scoop_patch, contain_patch, pound_patch, support_patch, wrapgrasp_patch, unlabelled_patch])
    return([grasp_patch, cut_patch,scoop_patch, contain_patch, pound_patch, support_patch, wrapgrasp_patch, unlabelled_patch])
def labelVisualize(label):
    ## Définition des couleurs

    label = label[:, :, 0] if len(label.shape) == 3 else label
    img_out = np.zeros(label.shape + (3,))
    print(color_dict.shape[0])
    for i in range(color_dict.shape[0]):
        img_out[label == i, :] = color_dict[i]
    return img_out / 255

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

if __name__=="__main__":
    create_folder(training=False)
    myTrainGen = trainGenerator(5, 'myers/training/', 'image', 'label', data_gen_args)
    mes_poids = weight_creation(myTrainGen)
    print(mes_poids)
