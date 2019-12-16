from model import unet
from data_generator import *
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.mytest import test_bayesian, test_common, test_ensemble
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras_segmentation.models.model_utils import save_mymodel, load_mymodel
import os

import matplotlib.pyplot as plt
from scipy import misc
train_again = False
run_all_test = True

### data ###
model_arch = None
n_classes = 8
testing_path = 'myers/testing/'
training_path = 'myers/training/'
save_folder = 'first_test/'
save_file = 'model_bayesian_default{}.h5'
train_again = False
weights = [0.15, 1, 1, 1, 1, 1, 1, 1]

######## Using keras-segmentation #########

# evaluation = model.evaluate_segmentation(inp_images_dir="myers/testing/image/", annotations_dir="myers/testing/label/")
# print('Résultat de l\'évaluation', evaluation)
# print(evaluation["average fw"])
model = vgg_unet(n_classes=8, input_height=224, input_width=224, bayesian=True)
model.train(train_images=training_path+"image/",
            train_annotations=training_path+"label/",
            checkpoints_path="/tmp/vgg_unet_1", epochs=5,
            verify_dataset=False,
            weighted_loss=weights)

if run_all_test:
    # test_ensemble(model_arch=vgg_unet, save_folder='first_test/', train_again=False)
    test_bayesian(model_arch=vgg_unet, train_again=True)
    # test_common(model_arch=vgg_unet)

######## Using the example dataset ##########
# model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608)

# model.train(
#     train_images =  "dataset1/images_prepped_train/",
#     train_annotations = "dataset1/annotations_prepped_train/",
#     checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
# )

# out = model.predict_segmentation(
#     inp="dataset1/images_prepped_test/0016E5_07965.png",
#     out_fname="/tmp/out.png"
# )

# import matplotlib.pyplot as plt
# plt.imshow(out)
# # evaluating the model
# print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

# plt.imshow(out)
# plt.show()

# # evaluating the model
# print(model.evaluate_segmentation(inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

######## Custom U-net ########
# grasp = [128, 128, 128]          # Gris
# cut = [128, 0, 0]                # Bordeau
# scoop = [192, 192, 128]          # Vert peau
# contain = [128, 64, 128]         # Violet
# pound = [60, 40, 222]            # Bleu
# support = [128, 128, 0]          # Vert terre
# wrapgrasp = [192, 128, 128]      # Rose
# unlabelled = [0, 0, 0]           # Noir

# COLOR_DICT = np.array([unlabelled, grasp, cut, scoop, contain, pound, support, wrapgrasp])

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

# myGene = trainGenerator(5, 'myers/training/', 'image', 'label', data_gen_args)
# im, lab = next(myGene)
# print(lab.dtype)
# train = True

# if train:
#     model = unet(nb_class=8)
#     model_checkpoint = ModelCheckpoint('unet_affordance.hdf5', monitor='loss', verbose=1, save_best_only=True)
#     model.fit_generator(myGene, steps_per_epoch=300, epochs=2, callbacks=[model_checkpoint])
# else:
#     model = unet(nb_class=8, pretrained_weights='unet_affordance.hdf5')

# testGene = testGenerator("myers/testing")
# results = model.predict_generator(testGene, 11, verbose=1)
# my_pred = np.argmax(results, axis=3)

# testGene = testGenerator("myers/testing", num_image=11)
# testGeneLabel = testlabelGenerator("myers/testing", num_image=11)

# for i in range(11):
#     im, label = next(testGene), next(testGeneLabel)
#     plt.subplot(1, 3, 1)
#     plt.imshow(im[0])
#     plt.subplot(1, 3, 2)
#     plt.imshow(label)
#     plt.subplot(1, 3, 3)
#     plt.imshow(my_pred[i, :, :])
#     plt.show()
