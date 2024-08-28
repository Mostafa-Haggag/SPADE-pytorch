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
    model = wide_resnet50_2(pretrained=True, progress=True)
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
        '''
        Size of dist_matrix:
        You mentioned that the matrix size is 160 x 320, meaning there are 160 test samples and 320 training samples
        . Each entry in this matrix represents the distance between a test sample and a training sample.
        '''
        # select K nearest neighbor and take average
        # topk is equal to 5 in ere
        # you have matrix of size 160,320
        # cominbation ofevery training set with every testing set
        topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
        # By selecting the top_k=5 nearest neighbors for each test sample,
        # you effectively reduce the focus to the 5 most
        # similar training samples for each test instance.
        # for each test test you choose the top 5 that are super close to it in the training
        # the distance is very close
        # This is a common strategy in tasks like k-NN classification, where you might average the results or
        # features from these nearest neighbors to make predictions or perform further analysis.
        '''
        torch.topk is a PyTorch function that returns the k smallest or largest elements of a tensor
        along a specified dimension.
        In this case, it's applied to dist_matrix to find the k nearest neighbors for each test sample,
        based on the distances.
        # Specifies that the function should find the nearest neighbors along the second dimension of dist_matrix,
        i.e., for each test sample across all training samples
        # largest=False: This indicates that the smallest values 
        (i.e., the nearest neighbors) should be selected, as we are interested in the smallest distances.
        ##############
        
        '''
        # it returns 160 by 5
        # 160 is teh size of test set
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()
        # this is the mean score of the topk 5  values
        # calculate image-level ROC AUC score
        # gt_list are for the test set
        # scores are the score for this specific category
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        score_map_list = []
        '''
        The code extracts features from different layers of a neural network,
        compares these features between test data and training data, and generates a "score map" for each test example.
        The score map highlights the areas in the test image that are most similar to the training data,
        likely indicating regions of interest for tasks like localization or anomaly detection.
        The code includes memory management strategies
        to avoid running out of CUDA memory, such as processing data in smaller batches,
        deleting unnecessary variables, and using torch.cuda.empty_cache()
        '''
        # looping on the number of test ouptus
        for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
            # you are looping over the full batch size
            score_maps = []
            # score_maps is initialized as an empty list to store the score maps generated for
            # different layers (layer1, layer2, layer3).
            # we have 3 layers
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer
                # This inner loop iterates over three different layers (layer1, layer2, layer3)
                # from the neural network. These layers likely correspond to feature maps extracted
                # from the model at different depths.
                # you are iterating over the layers, the layer_name is like the index
                # YOU ARE SKIPPING THE LAYERS OF AVGPOOLING
                # YOU NEED TO BE CAREFUL IN HERE

                # construct a gallery of features at all pixel locations of the K nearest neighbors
                # working on cpu to avoid the crashing
                #
                topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
                # you have topk_indexes the size of the TEST Set
                # For the current layer, the code extracts the feature maps of the top k nearest neighbors from the
                # training data for this current test sample t_idx (indexed by topk_indexes[t_idx]).
                test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
                # 1,256,56,56
                # you get the size of [1,the shape of the feature map at this layer]
                # The corresponding feature map of the current test example is extracted.
                # why am i i doing this thing of t_idx:t_idx + 1 :
                    # The main idea is that
                    # You are not slicing two items;
                        # you are slicing out a single item but preserving the dimension or structure of the data.
                        # This approach is commonly used when you want to maintain the original structure
                        # (like keeping it as a list or tensor) even when selecting a single item.
                # torch.cuda.empty_cache()
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)
                # topk feature map the idea in here that you have size of 5
                # these are 5 feature map that are similar to this specific test set
                # so you tranpose so that your feature map is [5,56,56,256]
                # you change teh number of channels with spatiail dimension
                # uou flatten from 0 to 2 so you have (5*spatial_dim*spatial_dim,number of channels)
                # you then unsquae in last 2 dimensions so you have
                # (5*spatial_dim*spatial_dim,number of channels,1,1)
                # feat_gallery: The extracted feature map (topk_feat_map) is transposed, flattened, and reshaped into
                # a format suitable for distance computation with the test feature map.
                # This creates a "gallery" of features across all spatial locations.

                # calculate distance matrix
                dist_matrix_list = []
                # dist_matrix_list: Initializes an empty list to store the distance matrices
                # yo uhave very big size .
                # (5*spatial_dim*spatial_dim,number of channels,1,1)
                # you iterate over 5*spatial_dim*spatial_dim
                # in batches
                # feature_gally 15680,256,1,1
                # we drop 80 features in this way
                for d_idx in range(feat_gallery.shape[0] // 100):
                    # The loop processes the gallery in smaller batches (of size 100 in this case) to
                    # compute the pairwise distance between features from the gallery
                    # and the test feature map using torch.pairwise_distance
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    # the feat gally has the size of 100,256,1,1
                    # test feture ,aps has size of 1,256,56,56
                    # the output is of the size of 100,256,56
                    # the 56 changes with the different layer
                    # Computes the pairwise distance between input vectors, or between columns of input matrices.

                    dist_matrix_list.append(dist_matrix)
                # The resulting distance matrices are concatenated to form the full distance matrix (dist_matrix).
                # you concate all teh distances in here
                # the length of dist_matrix_list is (15600,256,56)
                # 56 changes with the layers
                dist_matrix = torch.cat(dist_matrix_list, 0)
                # distance matrix for specific layer

                del topk_feat_map, test_feat_map,dist_matrix_list
                # torch.cuda.empty_cache()
                # k nearest features from the gallery (k=1)
                # you get the smallest distance matrics
                # dist_matrix (15600,256,56)
                score_map = torch.min(dist_matrix, dim=0)[0]
                # score_map has size of (256,56)
                # The minimum value in the distance matrix (torch.min) is taken to identify the nearest feature from
                # the gallery for each pixel in the test feature map, forming the score_map.
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                          mode='bilinear', align_corners=False)
                # you are adding 1,1 dimensionthen you are interpoalting to size of 2224
                # so your output in here has the size of [1,1,224,224]

                # The score map is then upsampled to a size of 224x224 using bilinear interpolation to
                # match the input image dimensions.
                score_maps.append(score_map)
                # The resulting score map is added to the score_maps list for the current test example.

            # average distance between the features
            # score_maps has the size of 3 because we are working on 3 layers
            # you concate allong list dimension
            # you calculate a mean score mape
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)
            # The score maps from all three layers are concatenated and averaged to get a single score map that
            # considers information from all the layers.
            # this score map has the size of (1,224,224)
            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            # apply gaussian filter  on specific numpy array
            # The averaged score map is smoothed using a Gaussian filter (gaussian_filter)
            # to reduce noise and improve localization.
            # you ass the score map for this test example
            score_map_list.append(score_map)
            # Finally, the smoothed score map is added to score_map_list, which stores
            # the results for all test examples.

        # you are looping in here to understand some specific stuff.
        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        '''
            Flattening: ravel() returns a flattened (1D) version of the input array,
            collapsing all the dimensions into a single dimension.
            Contiguous Flattened Array: If possible, ravel() will return a flattened view of the original 
            array without copying the data. However, if the array is not contiguous in memory,
            it may return a flattened copy.
        '''
        # we do the same for the score map
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
