# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='scannet', help='Dataset: sunrgbd or scannet [default: scannet]')
parser.add_argument('--num_point', type=int, default=5000, help='Point Number [default: 5000]')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions


#自分が持っているPLYデータのためのコード
from plyfile import PlyData

def read_ply(filename):
    """Read PLY file using plyfile library and return a numpy array of points."""
    plydata = PlyData.read(filename)
    pc = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return pc


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def save_labels_to_txt(pred_map_cls, dump_dir, class2type):
    label_file = os.path.join(dump_dir, "labels.txt")
    with open(label_file, "w") as f:
        for bbox in pred_map_cls[0]:
            label_id = bbox[0]  # クラスIDを取得
            class_name = class2type[label_id]  # クラス名に変換
            vertices = bbox[1]  # バウンディングボックスの頂点座標
            centroid = np.mean(vertices, axis=0)  # バウンディングボックスの中心
            f.write(f"Label: {class_name}, Centroid: {centroid}\n")

def flip_axis_to_camera(coords):
    """
    Y軸とZ軸を入れ替え、Z軸を反転させる関数
    """
    transformed_coords = np.copy(coords)
    # Y軸とZ軸を入れ替える
    transformed_coords[:, [1, 2]] = coords[:, [2, 1]]
    # Z軸を反転
    transformed_coords[:, 2] *= -1
    return transformed_coords

def flip_axis_to_camera_unity(coords):
    """
    X,Y軸を反転させる関数
    """
    transformed_coords = np.copy(coords)
    # X,Y軸を反転
    transformed_coords[:, 0] *= -1
    transformed_coords[:, 1] *= -1
    return transformed_coords

def export_bbox_to_obj(pred_map_cls, dump_dir, class2type):
    faces = np.array([
        [0, 1, 2], [2, 3, 0],  # 下の面
        [4, 5, 6], [6, 7, 4],  # 上の面
        [0, 1, 5], [5, 4, 0],  # 前の面
        [1, 2, 6], [6, 5, 1],  # 右の面
        [2, 3, 7], [7, 6, 2],  # 後ろの面
        [3, 0, 4], [4, 7, 3]   # 左の面
    ])

    for i, bbox in enumerate(pred_map_cls[0]):
        label_id = bbox[0]  # クラスIDを取得
        class_name = class2type[label_id]  # クラス名に変換
        vertices = bbox[1]  # バウンディングボックスの頂点座標
        centroid = np.mean(vertices, axis=0)  # バウンディングボックスの中心

        # 座標を変換 (Y軸とZ軸を入れ替える例)
        vertices = flip_axis_to_camera_unity(vertices)

        # Trimeshオブジェクトを作成
        bbox_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # outputfileの名前を作成する
        output_filename = os.path.join(dump_dir, f"{i}_{class_name}_pred_nms_confident_bbox.obj")
        bbox_mesh.export(output_filename, file_type='obj')


def filter_end_points_by_confidence(end_points, pred_map_cls, conf_thresh):
    """
    Filter the end_points based on confidence threshold, updating the bounding box data.
    """
    filtered_pred_map_cls = []
    for bbox in pred_map_cls[0]:
        if bbox[2] >= conf_thresh:  # bbox[2]は信頼度を示すと仮定
            filtered_pred_map_cls.append(bbox)

    # 必要に応じてend_pointsの中の他の関連データも更新
    end_points['pred_bbox_scores'] = np.array([bbox[2] for bbox in filtered_pred_map_cls])  # 信頼度スコアを更新
    end_points['pred_map_cls'] = [filtered_pred_map_cls]  # フィルタリングされたバウンディングボックスを格納

    return end_points


if __name__=='__main__':
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')

        #targetとなるPLYファイルをいれる#############################################################################
        # pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
        pc_path = os.path.join(demo_dir, '402.ply')
    else:
        print('Unkown dataset %s. Exiting.'%(DATASET))
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.9, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    point_cloud = read_ply(pc_path)

    # Debug: print the point cloud data
    print(point_cloud)
    print(point_cloud.shape)

    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
    
    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
  
    dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))

    

    # end_pointsをフィルタリング
    end_points = filter_end_points_by_confidence(end_points, pred_map_cls, eval_config_dict['conf_thresh'])



    # 保存したいラベル情報をテキストファイルに出力
    save_labels_to_txt(pred_map_cls, dump_dir, DC.class2type)
    print('Saved labels to labels.txt')

    # ラベル情報のついたbboxを出力
    export_bbox_to_obj(pred_map_cls, dump_dir, DC.class2type)

    # # フィルタリングされたバウンディングボックスをPLYファイルに保存
    # MODEL.dump_results(end_points, dump_dir, DC, True)
    # print(f"Dumped filtered detection results to folder {dump_dir}")