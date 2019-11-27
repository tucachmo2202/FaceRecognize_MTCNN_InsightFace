import glob
import pickle
from  ArcfaceSshOLnet import FacialRecognition
import cv2
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
    parser = argparse.ArgumentParser(description='create train embedding')
    parser.add_argument('--mtcnn-model', default='mtcnn-model',
                        help='path to load mtcnn model.')
    parser.add_argument('--arcface-model', default='model-r100-ii/model,0',
                        help='path to load face embedding model.')
    parser.add_argument('--ssh-detector', default='ssh-model-final/sshb',
                        help='path to load ssh detector model.')
    parser.add_argument('--file-output', default='train_sort_v100_ssh_v2.pkl',
                        help='embedding output file')
    parser.add_argument('--flip', default='0',
                        help='use flip image')
    parser.add_argument('--image-folder', default='mydata/',
                        help='')
    parser.add_argument('--image-file', default='sorted-train.txt',
                        help='')
    parser.add_argument('--gpu-index', default=-1, type=int,
                        help='-1 if use cpu')
    args = parser.parse_args()
    args.flip = convert_str2bool(args.flip, False)
    wr = open(args.file_output, "wb")
    #dicts = wr.read()
    dicts = []
    f = open(args.image_file, "r")
    model = FacialRecognition(mtcnn_model=args.mtcnn_model, arcface_model=args.arcface_model,
                              ssh_detector=args.ssh_detector, gpu_index=args.gpu_index, mtcnn_num_worker=2)
    error = 0
    for line in f:
        dict = {}
        line = line.replace("\n", "")
        label = line.split()[0]
        img_file = line.split()[1]
        print (img_file)
        img = cv2.imread(args.image_folder + img_file)
        embedding = model.detect_face_and_get_embedding(img)
        if embedding is None:
            error += 1
        if embedding is not None:
            dict['class'] = label
            dict['features'] = embedding
            #dict['imgfile'] = img_file
            dicts.append(dict)
        if args.flip:
            img = cv2.flip(img, 1)
            embedding = model.detect_face_and_get_embedding(img)
            if embedding is not None:
                dict['class'] = label
                dict['features'] = embedding
                #dict['imgfile'] = img_file
                dicts.append(dict)
    print ("So buc anh ko detect duoc face ", error)
    pickle.dump(dicts, wr)
    wr.close()
