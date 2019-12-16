import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from data_generator import labelVisualize, plot_legend_color
import cv2
from myfunction import plot_decision_or_not, plot_raw_decision, plot_CRF
from myfunction import precision, get_mean_iou, precision_recall_curve

type_unc = 'bayesian'    # ensembliste

for i in range(11):
    im = misc.imread('affordance_uncertainty/testing/image/im_{}.png'.format(i))
    ground_truth = misc.imread('affordance_uncertainty/testing/label/im_{}.png'.format(i))
    ground_truth = ground_truth[:, :, 2]

    out = np.load('affordance_uncertainty/for_test/im_{}_{}_out.npy'.format(i, type_unc))
    out_pre = np.load('affordance_uncertainty/for_test/im_{}_{}_out_pre.npy'.format(i, type_unc))
    unc = np.load('affordance_uncertainty/for_test/im_{}_{}_unc.npy'.format(i, type_unc))

    # out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)
    # print(precision(ground_truth, out), get_iou_custom(ground_truth, out))

    variation_ratio = 1 - np.max(out_pre, axis=2)
    entropy = -np.sum(out_pre * np.log(out_pre), axis=2)

    # plot_raw_decision(im, ground_truth, out, variation_ratio, entropy, unc)
    # plot_decision_or_not(im, ground_truth, out, variation_ratio)

    plot_CRF(im, ground_truth, out_pre)
