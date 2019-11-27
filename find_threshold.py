import pickle
import numpy as np
import faiss
import matplotlib.pyplot as plt
from random import randint
import math
import argparse
def compute_distance(first_vector, second_vector):
    distance = 0
    for i in range(len(first_vector)):
        distance += (second_vector[i] - first_vector[i]) * (second_vector[i] - first_vector[i])
    return distance


def read_embedding_file(path):
    f = open(path, "rb")
    dict = pickle.load(f)
    X = []
    Y = []
    db_img_paths = []
    for x in dict:
        _class = (x['class'])
        Y.append(_class)
        X.append(np.array(x['features']))
        db_img_paths.append(x['imgfile'])
    X = np.array(X)
    Y = np.array(Y)
    f.close()
    return X, Y, db_img_paths


def get_final_predictions(labels):
    a=np.zeros((1001))
    for i,label in enumerate(labels):
        a[label]+=1.0/(i+1)
    a=sorted(range(len(a)), key=a.__getitem__)
    a=a[-5:]
    a=a[::-1]
    return a

def emulator():
    parser = argparse.ArgumentParser(description='select threshold')
    parser.add_argument('--train-embedding-file', default='train_sort_v100_ssh_v2.pkl',
                        help='train embedding file')
    args = parser.parse_args()
    X_data, Y_data, path = read_embedding_file(args.train_embedding_file)
    d = 512
    search_model = faiss.IndexFlatL2(d)
    search_model.add(X_data)
    import random
    thresholds = np.arange(0.8, 1.3, 0.02)
    nb_test=5
    for test_i in range(nb_test):
        unknown_class=random.sample(range(0, 1000), 20) + [7, 9, 13, 17, 22, 23, 33, 39, 49, 64, 68, 81, 92, 93, 109, 117, 120, 134, 140, 142, 152, 171, 177, 198, 203, 205, 215, 228, 252, 284, 295, 308, 312, 313, 325, 340, 344, 348, 367, 395, 410, 415, 417, 447, 450, 458, 461, 467, 488, 521, 527, 556, 599, 613, 635, 663, 677, 678, 679, 680, 704, 710, 713, 745, 754, 770, 785, 794, 798, 800, 805, 808, 809, 817, 818, 842, 859, 922, 924, 927, 933, 949, 980, 989]
        print(unknown_class)
        for threshold in thresholds:
            total_accuracy=0
            for i in range(len(X_data)):
                    label=int(Y_data[i])
                    D, I = search_model.search(np.array([X_data[i]]), 15)
                    predictions = []
                    distances=[]
                    for k in range(1,len(I[0]),1):
                        predict_class = int(Y_data[I[0][k]])
                        if (predict_class == label) and (label in unknown_class):
                            continue
                        dis = D[0][k]
                        if dis > threshold and 1000 not in predictions:
                            predictions.append(1000)
                            distances.append(threshold)
                        predictions.append(predict_class)
                        distances.append(dis)
                    predictions=get_final_predictions(predictions)
                    if label in unknown_class:
                        label=1000
                    acc_ele=0
                    for k,predict_class in enumerate(predictions):
                            if predict_class==label:
                                acc_ele=(1.0/(k+1))
                                break
                    total_accuracy+=acc_ele
            print("threshold ",threshold," ,accuracy total ",total_accuracy*1.0/len(X_data))

if __name__=="__main__":
    emulator()

