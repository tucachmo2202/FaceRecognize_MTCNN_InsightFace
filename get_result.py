import pickle
import numpy as np
import faiss
import csv
import argparse

def read_data_train(path):
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

def read_data_test(path):
    f = open(path, "rb")
    dict = pickle.load(f)
    X = []
    db_img_paths = []
    for x in dict:
        X.append(np.array(x['features']))
        db_img_paths.append(x['imgfile'])
    X = np.array(X)
    f.close()
    return X, db_img_paths

def get_final_predictions(labels):
    a=np.zeros((1001))
    for i,label in enumerate(labels):
        a[label]+=1.0/(i+1)
    a=sorted(range(len(a)), key=a.__getitem__)
    a=a[-5:]
    a=a[::-1]
    return a

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='get final submit')
    parser.add_argument('--train-embedding-file', default='train_sort_v100_ssh_v2_flip.pkl',
                        help='train embedding file')
    parser.add_argument('--test-embedding-file', default='v100_ssh_test_embedding.pkl',
                        help='test embedding file')
    parser.add_argument('--output-file', default='final-submit.csv',
                        help='path to final submit file')
    parser.add_argument('--threshold', default=1.22,type=float,
                        help='threshold')
    args = parser.parse_args()

    X_train, Y_train, db_path = read_data_train(args.train_embedding_file)
    X_test, db_test = read_data_test(args.test_embedding_file)
    d = 512
    search_model = faiss.IndexFlatL2(d)
    search_model.add(X_train)
    with open(args.output_file, mode='w') as csv_file:
        fieldnames = ['image', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(X_test)):
            print (i)
            k = 15
            D, I = search_model.search(np.array([X_test[i]]), k)
            predictions = []
            for k in range(len(I[0])):
                la = int(Y_train[I[0][k]])
                if 1==1:
                    dis = D[0][k]
                    if dis > args.threshold and 1000 not in predictions:
                        predictions.append(1000)
                    predictions.append(la)
            predictions=get_final_predictions(predictions)
            rs=""
            for jj in predictions:
                rs+=str(jj)+" "
            rs = rs.strip()
            writer.writerow({'image': db_test[i], 'label': rs})