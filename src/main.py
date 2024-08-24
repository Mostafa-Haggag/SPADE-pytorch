import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import torchvision.models as models

import datasets.mvtec as mvtec


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    # what is this topk paramters ???
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def main():

    args = parse_args()
    path_of_dataset = r'/media/mostafahaggag/Shared_Drive/selfdevelopment/datasets'
    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    # he just loaded a pretrained network from torch visions
    # download the weights
    model = wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
    model.to(device)
    model.eval()

    # set model's intermediate outputs
    # you are putting everything in a list called outputs
    outputs = []
    def hook(module, input, output):
        # The user defined hook to be registered.
        outputs.append(output)
    #  The hook will be called every time after :func:`forward` has computed an output..
    # I need to understand the concept of hooks in geeneral
    '''
    So what are hooks? Hooks are functions that help to update the gradients, inputs or outputs dynamically. 
    That is I can change the behaviour of the Neural Network even when I am training it.

    Hooks are used in two places

        On tensors
        On torch.nn.Modules
    One more important thing is that to apply a hook we have to first “register” where we want to apply it.
     It may sound a little complex now, we will understand it in the further examples.
    A hook can be applied in 3 ways

        forward prehook (executing before the forward pass),
        forward hook (executing after the forward pass),
        backward hook (executing after the backward pass).
    Here forward pass is the part when inputs are used to compute the values of the next hidden neurons
     using the weights and so on until it reaches the end and returns an output. Backward Pass happens
      after calculating the Loss using the output’s value and the true value,
       then the gradients of each weight and bias of every layer are calculated
    in the direction of output to input(hence backwards) using the chain rule.
     Basically, the step when Backpropagation happens
     The hook will be called every time after forward() has computed an output.
     
    '''
    # like this you extrade from the moduel the output in teh last alyer
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    # These are the 2 arrays that he is going to use to keep the data
    total_roc_auc = []
    total_pixel_roc_auc = []
    # he is looping over all the class names which is a list
    for class_name in mvtec.CLASS_NAMES:
        # for each class name he creates the dataset and dataloader
        train_dataset = mvtec.MVTecDataset(root_path=path_of_dataset,class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(root_path=path_of_dataset,class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        # he creates the ordered directory
        # this is how you define ordered dict
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        # extract train set features
        # he sets the path that he save pikl file with classes name
        # it is very interesting to know what is he saving insdie of here
        train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            # there should be no mask in here
            for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    # you get the predict mask
                    # there is something wrong in here
                    # why he is doign nothing with perd
                    # because you are are noting caring about it
                    pred = model(x.to(device))
                    # the idea in here outputs is the place where the hooks put its things
                    # check function name bellow
                    #     def hook(module, input, output):
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    # putting eveerything to its right place again
                    train_outputs[k].append(v)
                # empty the output because it has the output from last iteration
                # this is a very stupid way to code but va beeneee
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)# put alll of them allong acies 0
            # save extracted feature
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            # you already have this saved somewhere.
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)
        # he preped the picke file in something callled train_outputs
        # it is important to understand what does this do exactly
        gt_list = []
        gt_mask_list = []
        test_imgs = []
        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                pred = model(x.to(device))
            # get intermediate layer outputs
            # why are we doing that again ??
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v)
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        # THERE IS NO TRAINING IN THIS METHOD
        # THE METHOD STARTS FROM HERE
        # you looped over everything in the test set
        # now i understand
        # calculate distance matrix
        # at the average pooling the size is 160,2024,1,1
        dist_matrix = calc_dist_matrix(torch.flatten(test_outputs['avgpool'], 1),
                                       torch.flatten(train_outputs['avgpool'], 1))

        # select K nearest neighbor and take average
        # topk is equal to 5 in ere
        # you have matrix of size 160,320
        # cominbation ofevery training set with every testing set
        topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
        # it returns 160 by 5
        # 160 is teh size of test set
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()
        # this is the mean score of the topk 5  values
        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        score_map_list = []
        for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
                test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

                # calculate distance matrix
                dist_matrix_list = []
                for d_idx in range(feat_gallery.shape[0] // 100):
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    dist_matrix_list.append(dist_matrix)
                dist_matrix = torch.cat(dist_matrix_list, 0)

                # k nearest features from the gallery (k=1)
                score_map = torch.min(dist_matrix, dim=0)[0]
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                          mode='bilinear', align_corners=False)
                score_maps.append(score_map)

            # average distance between the features
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            score_map_list.append(score_map)

        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, vis_num=5)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    # you have vectors of size 160 by 2048
    n = x.size(0)#160 batch size of the test set
    m = y.size(0)#160 batch size for the training set
    d = x.size(1) # size of the vecotr
    # you unsqueeze as postion 1
    # X IS ORINGALLLY N,1,2048
    #IT become  N,M,2048
    x = x.unsqueeze(1).expand(n, m, d)
    # you unsqueeze as postion 0
    # n PyTorch, the expand function is used to "expand" the dimensions of a tensor to a larger size without
    # actually copying the data.
    # This is achieved by creating a new view of the original tensor with the expanded size,
    # where the new dimensions are broadcasted.
    # SIMPLE EXAMPLE OF tensor
    # Original tensor of shape (1, 3)
    # x = torch.tensor([[1, 2, 3]])
    # # Expanding the tensor to shape (4, 3)
    # y = x.expand(4, 3)
    # y is orginally 1,M,D
    # it becomes N,M,D
    y = y.unsqueeze(0).expand(n, m, d)

    # so the size of x is (N,M,2048) you repeat allong M dimenison
    # the size of y is (N,M,2048) you repseat along N dimension
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    # calculate the distance matrix like that it is a square matrix
    return dist_matrix


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num=5):

    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    # the code starts from here
    main()
