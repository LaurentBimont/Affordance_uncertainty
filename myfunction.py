import numpy as np
import matplotlib.pyplot as plt
from data_generator import labelVisualize, plot_legend_color
import cv2
import time
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

def get_mean_iou(gt, pr, range_inf=1, range_sup=8, n_classes=8):
    class_wise = np.zeros(range_sup-range_inf)
    EPS=1e-6
    for cl in range(range_inf, range_sup):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.minimum(np.maximum((gt == cl), (pr == cl)), (pr != 8)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl-range_inf] = iou
    result = np.sum(class_wise)/np.sum(np.where(class_wise != 0, 1, 0))
    return result


def precision(gt, pr):
    pr, gt = pr[gt != 0], gt[gt != 0]
    pr, gt = pr[pr != 8], gt[pr != 8]
    pr, gt = pr[pr != 0], gt[pr != 0]
    num = np.sum(np.where((gt == pr), 1, 0))
    den = len(gt)
    return 100*num/den


def decision_or_not(unc, out, threshold=0.5):
    out[unc > threshold] = 8
    return out


def precision_recall_curve(out, unc, gt, precision_func):
    thresh = np.linspace(0, 100)/100
    # gt = cv2.resize(gt, (112, 112), interpolation=cv2.INTER_NEAREST)
    unc = unc[gt != 0]
    out = out[gt != 0]
    gt = gt[gt != 0]
    out = np.array([o for _, o in sorted(zip(unc, out))])
    gt = np.array([g for _, g in sorted(zip(unc, gt))])
    unc = sorted(unc)
    res, unc_thresh = [], []
    for t in thresh[:-1]:
        res.append(precision_func(gt[0:int(t * len(gt))], out[0:int(t * len(gt))]))
        unc_thresh.append(unc[int(t * len(gt))])

    return thresh[:-1], res, unc_thresh

UNCERTAINTY_THRESHOLD = 0.5

def plot_raw_decision(im, ground_truth, out, variation_ratio, entropy, unc):
    out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Sortie brute du réseau (U-net avec VGG16 encoder)', fontsize=14)

    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Image')
    ax.imshow(im)
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Ground Truth')
    my_handle = plot_legend_color(ground_truth)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    ax.axis('off')
    ax.imshow(labelVisualize(ground_truth))

    ax = fig.add_subplot(2, 3, 3)
    score, prec = np.mean(get_mean_iou(ground_truth, out)), precision(ground_truth, out)
    ax.set_title('Segmentation result (IOU:{}, precision:{})'.format(round(score, 2), round(prec, 2)))
    my_handle = plot_legend_color(out)
    out = labelVisualize(out)
    out = cv2.resize(out, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    ax.axis('off')
    ax.imshow(out)

    ax = fig.add_subplot(2, 3, 4)
    ax.set_title('Maximum variance')
    # plt.colorbar(ax.pcolor(unc[:, :, 0]))
    unc = cv2.resize(unc, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(unc, vmin=0, vmax=np.max(unc))

    ax = fig.add_subplot(2, 3, 6)
    ax.set_title('Variation Ratio')
    # plt.colorbar(ax.pcolor(np.array([[1, 0]])))
    variation_ratio = cv2.resize(variation_ratio, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(variation_ratio, vmin=0, vmax=0.9)

    ax = fig.add_subplot(2, 3, 5)

    # plt.colorbar(ax.pcolor(np.array([[2, 0]])))
    ax.set_title('Entropy')
    entropy = cv2.resize(entropy, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(entropy, vmin=0, vmax=2)

    plt.show()

def plot_decision_or_not(im, ground_truth, out, variation_ratio):
    raw_out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)
    variation_ratio = cv2.resize(variation_ratio, (640, 480), interpolation=cv2.INTER_NEAREST)
    thresh, res, unc_thresh = precision_recall_curve(raw_out, variation_ratio, ground_truth, precision)
    out_temp = np.copy(raw_out)
    out = decision_or_not(variation_ratio, out_temp, threshold=UNCERTAINTY_THRESHOLD)
    # thresh, res, unc_thresh = precision_recall_curve(out, variation_ratio, ground_truth, precision)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Decision or not (variation ratio > {})'.format(UNCERTAINTY_THRESHOLD), fontsize=14)

    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Image')
    ax.imshow(im)
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Ground Truth')
    my_handle = plot_legend_color(ground_truth)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    ax.axis('off')
    ax.imshow(labelVisualize(ground_truth))

    ax = fig.add_subplot(2, 3, 3)
    score, prec = np.mean(get_mean_iou(ground_truth, raw_out)), precision(ground_truth, raw_out)
    ax.set_title('Segmentation result (IOU:{}, precision:{})'.format(round(score, 2), round(prec/2, 2)))
    my_handle = plot_legend_color(raw_out)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    # viz_out = labelVisualize(raw_out)
    # viz_out = cv2.resize(viz_out, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(labelVisualize(raw_out))

    ax = fig.add_subplot(2, 3, 4)
    ax1 = ax.twinx()
    ax.set_title('Precision-Recall curve')
    ax.plot(100*thresh, res, color='g', linewidth=5, label='Precision')
    ax1.plot(100*thresh, unc_thresh, color='black', linestyle='dashed', label='Uncertainty threshold')

    line1, label1 = ax.get_legend_handles_labels()
    line2, label2 = ax1.get_legend_handles_labels()
    ax1.legend(line1 + line2, label1 + label2, loc='lower right')

    # plots = plot1 + plot2
    # labs = [l.get_label() for l in plots]
    ax.axhline(50, linewidth=1, color='b', linestyle='dashed')
    ax.set_ylim(0, 110)
    ax1.set_ylim(0, 1.10)


    ax.set_xlabel('Percentage of recall sorted pixels (in %)')
    ax.set_ylabel('Precision in %')
    ax1.set_ylabel('Variation-ratio value')
    ax.set_title('Precision-Recall curve')

    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Variation Ratio')
    variation_ratio = cv2.resize(variation_ratio, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(variation_ratio, vmin=0, vmax=1)

    ax = fig.add_subplot(2, 3, 6)
    score, prec = np.mean(get_mean_iou(ground_truth, out)), precision(ground_truth, out)
    ax.set_title('Segmentation decision (IOU:{}, precision {})'.format(round(score, 2), round(prec/2, 2)))
    my_handle = plot_legend_color(out)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    viz_out = cv2.resize(labelVisualize(out), dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(viz_out)

    # plt.savefig('decision.png', dpi=1000)
    plt.show()

def CRF_inference(out_pre, im):
    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    out_pre = out_pre.squeeze()
    out_pre = out_pre.transpose((2, 0, 1))
    im_0 = cv2.resize(im, (112, 112), interpolation=cv2.INTER_NEAREST)
    unary = unary_from_softmax(out_pre)
    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)
    print(unary.shape)

    d = dcrf.DenseCRF(im_0.shape[0] * im_0.shape[1], 8)
    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=im_0.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=im_0, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(1)
    var_Q = (1 - np.max(np.array(Q), axis=0)).reshape(112, 112)
    var_Q = cv2.resize(var_Q.reshape(112, 112), (640, 480), interpolation=cv2.INTER_NEAREST)
    Q = np.argmax(np.array(Q), axis=0)
    Q = cv2.resize(Q.reshape(112, 112), (640, 480), interpolation=cv2.INTER_NEAREST)

    return Q, var_Q


def plot_CRF(im, ground_truth, out_pre):
    print(out_pre.shape)

    Q, var_Q = CRF_inference(out_pre, im)

    out_pre = cv2.resize(out_pre, (640, 480), interpolation=cv2.INTER_NEAREST)
    soft_out = np.argmax(out_pre, axis=2)
    variation_ratio = 1 - np.max(out_pre, axis=2)
    # plt.subplot(1, 2, 1)
    # plt.imshow(soft_out)
    # plt.subplot(1, 2, 2)
    # plt.imshow(variation_ratio)
    # plt.show()

    thresh, res, unc_thresh = precision_recall_curve(soft_out, variation_ratio, ground_truth, precision)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Decision or not (variation ratio > {})'.format(UNCERTAINTY_THRESHOLD), fontsize=14)
    n_row, n_col = 2, 4
    ax = fig.add_subplot(n_row, n_col, 1)
    ax.set_title('Image')
    ax.imshow(im)
    ax.axis('off')

    ax = fig.add_subplot(n_row, n_col, 5)
    ax.set_title('Ground Truth')
    my_handle = plot_legend_color(ground_truth)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    ax.axis('off')
    ax.imshow(labelVisualize(ground_truth))

    ax = fig.add_subplot(n_row, n_col, 2)
    score, prec = np.mean(get_mean_iou(ground_truth, soft_out)), precision(ground_truth, soft_out)
    ax.set_title('Segmentation result (IOU:{}, precision:{})'.format(round(score, 2), round(prec / 2, 2)))
    my_handle = plot_legend_color(soft_out)
    fig.legend(handles=my_handle, bbox_to_anchor=(0, 0), loc='best',
               ncol=8, mode="expand", borderaxespad=0.)
    # viz_out = labelVisualize(raw_out)
    # viz_out = cv2.resize(viz_out, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.axis('off')
    ax.imshow(labelVisualize(soft_out))

    ax = fig.add_subplot(n_row, n_col, 3)
    ax1 = ax.twinx()
    ax.set_title('Precision-Recall curve')
    ax.plot(100 * thresh, res, color='g', linewidth=5, label='Precision')
    ax1.plot(100 * thresh, unc_thresh, color='black', linestyle='dashed', label='Uncertainty threshold')
    line1, label1 = ax.get_legend_handles_labels()
    line2, label2 = ax1.get_legend_handles_labels()
    ax1.legend(line1 + line2, label1 + label2, loc='lower right')
    ax.axhline(50, linewidth=1, color='b', linestyle='dashed')
    ax.set_ylim(0, 110)
    ax1.set_ylim(0, 1.10)
    ax.set_xlabel('Percentage of recall sorted pixels (in %)')
    ax.set_ylabel('Precision in %')
    ax1.set_ylabel('Variation-ratio value')
    ax.set_title('Precision-Recall curve')

    ax = fig.add_subplot(n_row, n_col, 6)
    ax.set_title('Variation Ratio')
    ax.axis('off')
    ax.imshow(labelVisualize(Q))

    ax = fig.add_subplot(n_row, n_col, 7)
    ax1 = ax.twinx()
    ax.set_title('Precision-Recall curve')
    thresh, res, unc_thresh = precision_recall_curve(Q, var_Q, ground_truth, precision)
    ax.plot(100 * thresh, res, color='g', linewidth=5, label='Precision')
    ax1.plot(100 * thresh, unc_thresh, color='black', linestyle='dashed', label='Uncertainty threshold')
    line1, label1 = ax.get_legend_handles_labels()
    line2, label2 = ax1.get_legend_handles_labels()
    ax1.legend(line1 + line2, label1 + label2, loc='lower right')
    ax.axhline(50, linewidth=1, color='b', linestyle='dashed')
    ax.set_ylim(0, 110)
    ax1.set_ylim(0, 1.10)
    ax.set_xlabel('Percentage of recall sorted pixels (in %)')
    ax.set_ylabel('Precision in %')
    ax1.set_ylabel('Variation-ratio value')
    ax.set_title('Precision-Recall curve')

    ax = fig.add_subplot(n_row, n_col, 4)
    ax.set_title('Variation Ratio')
    ax.axis('off')
    ax.imshow(variation_ratio, vmin=0, vmax=1)

    ax = fig.add_subplot(n_row, n_col, 8)
    ax.set_title('Variation Ratio')
    ax.axis('off')
    ax.imshow(var_Q, vmin=0, vmax=1)

    # plt.tight_layout()

    plt.show()


