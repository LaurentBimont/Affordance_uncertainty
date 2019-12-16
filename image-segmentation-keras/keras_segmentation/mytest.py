import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from data_generator import *
import cv2
import os

def test_ensemble(model_arch=None, n_classes=8, save_folder='ensemble_default/', testing_path='myers/testing/', training_path='myers/training/', save_file='model_bayesian_default{}.h5', train_again=False, weights=[0.15, 1, 1, 1, 1, 1, 1, 1]):
    N = 5  # Nb modèles ensemblistes
    for i in range(N):
        if train_again:
            model = model_arch(n_classes=n_classes, input_height=224, input_width=224, bayesian=False)
            model.train(train_images=training_path + "image/",
                        train_annotations=training_path + "label/",
                        checkpoints_path="/tmp/vgg_unet_1", epochs=5,
                        verify_dataset=False,
                        weighted_loss=weights)

            if not os.path.exists('model_saved/' + save_folder):
                os.mkdir('model_saved/' + save_folder)
            model.save_weights('model_saved/' + save_folder + save_file.format(i))

    directory = testing_path + 'image/'
    N_in_dir = len(os.listdir('model_saved/' + save_folder))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        im = misc.imread(testing_path + "image/" + filename)
        ground_truth = misc.imread(testing_path + "label/" + filename)
        model = model_arch(n_classes=n_classes, input_height=224, input_width=224, bayesian=False)
        out, out_pre, unc = model.predict_ensembliste(inp=testing_path + "image/" + filename,
                                                   out_fname="/tmp/out.png", save_folder=save_folder)

        np.save('for_test/' + file.rstrip('.png') + '_ensembliste_out', out)
        np.save('for_test/' + file.rstrip('.png') + '_ensembliste_out_pre', out_pre)
        np.save('for_test/' + file.rstrip('.png') + '_ensembliste_unc', unc)

        variation_ratio = 1 - np.max(out_pre, axis=2)
        entropy = -np.sum(out_pre * np.log(out_pre), axis=2)
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle("Méthode ensembliste (N={})".format(N_in_dir), fontsize=14)

        ax = fig.add_subplot(2, 3, 3)
        ax.set_title('Image')
        ax.imshow(im)
        # ax.axis('off')
        my_handle = plot_legend_color()
        fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
                   ncol=8, mode="expand", borderaxespad=0.)

        ax = fig.add_subplot(2, 3, 2)
        ax.set_title('Ground Truth')
        ax.imshow(labelVisualize(ground_truth))

        ax = fig.add_subplot(2, 3, 6)
        ax.set_title('Entropie')
        entropy = cv2.resize(entropy[:, :], dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(entropy, vmin=np.min(entropy), vmax=np.max(entropy))

        ax = fig.add_subplot(2, 3, 4)
        ax.set_title('Maximum variance')
        unc = cv2.resize(unc[:, :, 0], dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(unc, vmin=0, vmax=np.max(unc))

        ax = fig.add_subplot(2, 3, 1)
        ax.set_title('Segmentation result')
        out = labelVisualize(out)
        out = cv2.resize(out, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(out)

        ax = fig.add_subplot(2, 3, 5)
        ax.set_title('Variation Ratio')
        variation_ratio = cv2.resize(variation_ratio, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(variation_ratio, vmin=0, vmax=1)
        plt.show()

def test_bayesian(model_arch=None, n_classes=8, testing_path='myers/testing/', training_path='myers/training/',
                  save_file= 'model_bayesian_default.h5', train_again=False, weights=[0.15, 1, 1, 1, 1, 1, 1, 1]):
    if train_again:
        model = model_arch(n_classes=n_classes, input_height=224, input_width=224, bayesian=True)
        model.train(train_images=training_path+"image/",
                    train_annotations=training_path+"label/",
                    checkpoints_path="/tmp/vgg_unet_1", epochs=5,
                    verify_dataset=False,
                    weighted_loss=weights)
        model.save_weights('model_saved/'+save_file)
    else:
        model = model_arch(n_classes=8, input_height=224, input_width=224, bayesian=True)
        model.load_weights('model_saved/'+save_file)

    directory = testing_path+'image/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        im = misc.imread(testing_path+"image/"+filename)
        ground_truth = misc.imread(testing_path+"label/"+filename)
        out, out_pre, unc = model.predict_bayesian(inp=testing_path+"image/"+filename,
                                         out_fname="/tmp/out.png")

        np.save('for_test/' + file.rstrip('.png') + '_bayesian_out', out)
        np.save('for_test/' + file.rstrip('.png') + '_bayesian_out_pre', out_pre)
        np.save('for_test/' + file.rstrip('.png') + '_bayesian_unc', unc)

        variation_ratio = 1 - np.max(out_pre, axis=2)
        entropy = -np.sum(out_pre * np.log(out_pre), axis=2)
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle("Monte Carlo Dropout", fontsize=14)

        ax = fig.add_subplot(2, 3, 3)
        ax.set_title('Image')
        ax.imshow(im)
        # ax.axis('off')
        my_handle = plot_legend_color()
        fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
                   ncol=8, mode="expand", borderaxespad=0.)

        ax = fig.add_subplot(2, 3, 2)
        ax.set_title('Ground Truth')
        ax.imshow(labelVisualize(ground_truth))

        ax = fig.add_subplot(2, 3, 6)
        ax.set_title('Entropie')
        entropy = cv2.resize(entropy[:, :], dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(entropy, vmin=np.min(entropy), vmax=np.max(entropy))

        ax = fig.add_subplot(2, 3, 4)
        ax.set_title('Maximum variance')
        unc = cv2.resize(unc[:, :, 0], dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(unc, vmin=0, vmax=np.max(unc))

        ax = fig.add_subplot(2, 3, 1)
        ax.set_title('Segmentation result')
        out = labelVisualize(out)
        out = cv2.resize(out, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(out)

        ax = fig.add_subplot(2, 3, 5)
        ax.set_title('Variation Ratio')
        variation_ratio = cv2.resize(variation_ratio, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax.imshow(variation_ratio, vmin=0, vmax=1)
        plt.show()


def test_common(model_arch=None, testing_path='myers/testing/', training_path='myers/training/', save_file= 'model_bayesian_default.h5', train_again=False, weights=[0.15, 1, 1, 1, 1, 1, 1, 1]):
    if train_again:
        model = model_arch(n_classes=8, input_height=224, input_width=224)
        model.train(train_images=training_path + "image/",
                    train_annotations=training_path + "label/",
                    checkpoints_path="/tmp/vgg_unet_1", epochs=5,
                    verify_dataset=False,
                    weighted_loss=weights)
        model.save_weights('model_saved/' + save_file)
    else:
        model = model_arch(n_classes=8, input_height=224, input_width=224)
        model.load_weights('model_saved/' + save_file)

    directory = testing_path + 'image/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        im = misc.imread(testing_path + "image/" + filename)
        ground_truth = misc.imread(testing_path + "label/" + filename)
        out = model.predict_segmentation(inp=testing_path + "image/" + filename,
                                                   out_fname="/tmp/out.png")

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(labelVisualize(out))
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(labelVisualize(ground_truth))
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(im)
        my_handle = plot_legend_color()
        fig.legend(handles=my_handle, bbox_to_anchor=(2, 0), loc='best',
                   ncol=8, mode="expand", borderaxespad=0.)
        plt.show()
