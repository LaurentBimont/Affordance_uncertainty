import numpy as np
from keras_segmentation.data_utils.data_loader import image_segmentation_generator, get_pairs_from_paths
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict, predict_bayesian
import os
from sklearn.utils import shuffle
from model import unet
import shutil


# my_gen_pool = image_segmentation_generator('myers/active_learning/pool/image/', 'myers/active_learning/pool/label/',
#                                               batch_size=len(os.listdir('myers/active_learning/pool/image/')), n_classes=8,
#                                               input_height=224, input_width=224, output_height=112, output_width=112)
#
# my_gen_eval = image_segmentation_generator('myers/active_learning/evaluation/image/', 'myers/active_learning/evaluation/label/',
#                                               batch_size=len(os.listdir('myers/active_learning/evaluation/image/')), n_classes=8,
#                                               input_height=224, input_width=224, output_height=112, output_width=112)
# print(len(os.listdir('myers/active_learning/labeled/image/')))
# print(os.listdir('myers/active_learning/labeled/image/'))
# my_gen_label = image_segmentation_generator('myers/active_learning/labeled/image/', 'myers/active_learning/labeled/label/',
#                                               batch_size=len(os.listdir('myers/active_learning/labeled/image/')), n_classes=8,
#                                               input_height=224, input_width=224, output_height=112, output_width=112)


weights = [0.15, 1, 1, 1, 1, 1, 1, 1]

training_path = 'myers/training/'
shutil.copyfile('myers/active_learning/pool.txt', 'myers/training/pool.txt')
shutil.copyfile('myers/active_learning/labeled.txt', 'myers/training/labeled.txt')

def nb_above_50(inp_images_dir, annotations_dir):
    paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
    paths = list(zip(*paths))
    inp_images = list(paths[0])
    annotations = list(paths[1])
    for inp, ann in zip(inp_images, annotations):
        pr, pr_max = predict(model, inp)
        variation_ratio = 1 - pr_max
        

model = vgg_unet(n_classes=8, input_height=224, input_width=224, bayesian=False)
evaluation_result = []
for i in range(2):
    model.train(train_images=training_path+"image/",
                train_annotations=training_path+"label/",
                checkpoints_path="/tmp/vgg_unet_1", epochs=1,
                verify_dataset=False,
                weighted_loss=weights,
                from_file='myers/training/labeled.txt')
    evaluation_result.append(model.evaluate_segmentation(inp_images_dir='myers/active_learning/evaluation/image/', annotations_dir='myers/active_learning/evaluation/label/'))
    sampling_fun
print(evaluation_result)
