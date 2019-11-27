import pickle
import numpy as np
import cv2
import csv
import pandas as pd
from  ArcfaceSshOLnet import FacialRecognition
import argparse


def convert_str2bool(arg_str, default_v):
    if isinstance(arg_str, str):
        if arg_str.lower() in {"1", "y", "yes"}:
            return True
        if arg_str.lower() in {"0", "n", "no"}:
            return False
    if isinstance(arg_str, bool):
        return arg_str
    return default_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create test embedding')
    parser.add_argument('--mtcnn-model', default='mtcnn-model',
                        help='path to load mtcnn model.')
    parser.add_argument('--arcface-model', default='model-r100-ii/model,0',
                        help='path to load face embedding model.')
    parser.add_argument('--ssh-detector', default='ssh-model-final/sshb',
                        help='path to load ssh detector model.')
    parser.add_argument('--file-output', default='v100_ssh_test_embedding.pkl',
                        help='embedding output file')
    parser.add_argument('--image-folder', default='vn_celeb_face_recognition/test/',
                        help='')
    parser.add_argument('--sample-submission-file', default='vn_celeb_face_recognition/sample_submission.csv',
                        help='')
    parser.add_argument('--gpu-index', default=-1, type=int,
                        help='-1 if use cpu')
    args = parser.parse_args()
    wr = open(args.file_output, "wb")
    dicts = []
    csv_reader = pd.read_csv(args.sample_submission_file)
    model = FacialRecognition(mtcnn_model=args.mtcnn_model, arcface_model=args.arcface_model,
                              ssh_detector=args.ssh_detector, gpu_index=args.gpu_index, mtcnn_num_worker=2)
    for i in range(len(csv_reader)):
        img_file = args.image_folder + csv_reader.loc[i][0]
        print img_file
        img = cv2.imread(img_file)
        dict = {}
        embedding = model.detect_face_and_get_embedding(img)
        if embedding is None:
            embedding = model.get_embedding(img)
            dict['features'] = embedding
            dict['imgfile'] = csv_reader.loc[i][0]
        else:
            dict['features'] = embedding
            dict['imgfile'] = csv_reader.loc[i][0]
        dicts.append(dict)
    pickle.dump(dicts, wr)
    wr.close()
