import numpy as np
import keras.backend as K
EPS = 1e-12

def get_iou(gt, pr, n_classes=8):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

def iou_metric_score(y_true, y_pred, smooth=1):
    y_true = y_true[:, :, 1:-1]
    y_pred = y_pred[:, :, 1:-1]
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Création de la métrique F omega beta
def dist(i, j, shape):
    return (i//shape-j//shape)**2+(i%shape-j%shape)**2

def dist_class(G, i):
    shape = G.shape[0]
    my_stuff = np.where(G != 0)
    my_dist = np.array([[my_stuff[0][i], my_stuff[1][i]] for i in range(len(my_stuff[0]))])
    return np.min(np.linalg.norm(my_dist-np.array([i//shape, i%shape]), axis=1))

def A(G, D):
    '''
    :param G: Ground Truth
    :param D: Output
    :return: Fomegabeta score
    '''
    G_flat, D_flat = G.flatten(), D.flatten()
    shape = G.shape[0]
    A = np.zeros((len(G_flat), len(G_flat)))
    for i in range(len(G_flat)):
        for j in range(len(G_flat)):
            if G_flat[i] != 0 and G_flat[j] != 0:
                sigma = 5
                A[i, j] = np.exp(dist(i, j, shape)/(2*sigma))/(np.sqrt(2*np.pi)*sigma)
            if G_flat[i] == 0 and i == j:
                A[i, j] = 1
    return A

def B(G):
    G_flat = G.flatten()
    B = np.ones(len(G_flat))
    alpha = np.log(5)/5
    for i in range(len(G_flat)):
        if G_flat[i] == 0:
            B[i] = 2-np.exp(alpha*dist_class(G, i))
    return B

def weighted_F(G, D, beta=1):
    G_flat, D_flat = np.where(G!=0, 1, 0).flatten(), np.where(D!=0, 1, 0).flatten()
    E = np.abs(G_flat-D_flat)
    EA = np.dot(E, A(G, D))
    minE = np.minimum(E, EA)
    E_omeg = minE * B(G)

    TP = (1-E)*G_flat
    FP = E*(1-G_flat)
    TN = (1-E)*(1-G_flat)
    FN = E*G_flat
    precision = np.sum(TP)/(np.sum(TP)+np.sum(FP)+0.000000000001)
    recall = np.sum(TP)/(np.sum(TP)+np.sum(FN)+0.000000000001)
    Fb = (1+beta**2)*(precision*recall)/(beta**2*precision+recall)
    return Fb #, precision, recall


if __name__=="__main__":
    G = np.zeros((224,224))
    D = np.zeros((224,224))
    # G = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 1]])
    # D = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print(weighted_F(G, D))

# sklearn.metrics import fbeta_score
#
# function [Q]= WFb(FG,GT)
# # % WFb Compute the Weighted F-beta measure (as proposed in "How to Evaluate
# # % Foreground Maps?" [Margolin et. al - CVPR'14])
# # % Usage:
# # % Q = FbW(FG,GT)
# # % Input:
# # %   FG - Binary/Non binary foreground map with values in the range [0 1]. Type: double.
# # %   GT - Binary ground truth. Type: logical.
# # % Output:
# # %   Q - The Weighted F-beta score
# #
# # %Check input
# if (~isa( FG, 'double' ))
#     error('FG should be of type: double');
# end
# if ((max(FG(:))>1) || min(FG(:))<0)
#     error('FG should be in the range of [0 1]');
# end
# if (~islogical(GT))
#     error('GT should be of type: logical');
# end
#
# dGT = double(GT); %Use double for computations.
#
# E = abs(FG-dGT);
# % [Ef, Et, Er] = deal(abs(FG-GT));
#
# [Dst,IDXT] = bwdist(dGT);
# %Pixel dependency
# K = fspecial('gaussian',7,5);
# Et = E;
# Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
# EA = imfilter(Et,K);
# MIN_E_EA = E;
# MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
# %Pixel importance
# B = ones(size(GT));
# B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
# Ew = MIN_E_EA.*B;
#
# TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
# FPw = sum(sum(Ew(~GT)));
#
# R = 1- mean2(Ew(GT)); %Weighed Recall
# P = TPw./(eps+TPw+FPw); %Weighted Precision
#
# Q = (2)*(R*P)./(eps+R+P); %Beta=1;
# % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
# end
