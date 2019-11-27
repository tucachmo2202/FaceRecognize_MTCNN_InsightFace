import pickle
import time
import cv2
import mxnet as mx
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from ArcfaceSshOLnet import *
from face_preprocess import *
from imutils.video import VideoStream
import imutils


def draw(boxx, name, img):
    boxx = [int(boxx[1]), int(boxx[3]), int(boxx[0]), int(boxx[2])]
    print("box drawing")
    print(boxx)
    if (name == "unknown"):
        color = (0,155,255)
    else:
        color = (155,255,0)
    cv2.rectangle(img,
                (boxx[3], boxx[1]),
                (boxx[2], boxx[0]),
                (0,155,255),2)
    cv2.putText(img, name, (boxx[2], boxx[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

def detect_and_tracking():
    cap = cv2.VideoCapture(0)
    f = open("train_sort_v100_ssh_v2.pkl", "rb")
    model = pickle.load(f)
    Detector = FacialRecognition()
    trackers = cv2.MultiTracker_create()
    iter = 0
    check = 0
    while True:
        __, frame = cap.read()
        iter += 1
        frame = imutils.resize(frame, width = 600)
        frame = cv2.flip(frame, 1)
        if (iter-9 == 0):
            iter = 0
            trackers.clear()
            trackers = cv2.MultiTracker_create()
            try:
                em,box=Detector.detect_face_and_get_embedding(frame)
            except:
                print("Detector no one")
                cv2.imshow("Nhan dien", frame)
                continue
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break
            reses = []
            for i in range(len(box)):
                b = em[i]
                boxx = box[i]
                dist = 10
                res = ""
                for person in model:
                    a = person['features'][0]
                    dis_temp = distance.euclidean(a,b)
                    if(dis_temp < dist):
                        res = person['class']
                        dist = dis_temp
                if dist > 1.1: res = "unknown"
                print(res)
                reses.append(res)
                boxx = tuple(boxx[:4])
                print("box chua qua track:")
                print(boxx)
                draw(boxx, res, frame)
                tracker = cv2.TrackerCSRT_create()
                trackers.add(tracker, frame, boxx)
                cv2.imshow("Nhan dien", frame)

            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        else:
            (success, boxes) = trackers.update(frame)
            for i in range(len(boxes)):
                print("box sau track")
                print(boxes[i])
                draw(boxes[i], reses[i], frame)
            cv2.imshow("Nhan dien", frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def camera_detect():
    cap = cv2.VideoCapture(0)
    f = open("train_sort_v100_ssh_v2.pkl", "rb")
    model = pickle.load(f)
    Detector = FacialRecognition()
    while True:
        __, frame = cap.read()
        try:
            em,box=Detector.detect_face_and_get_embedding(frame)
        except:
            print("Detector no one")
            continue
        for i in range(len(box)):
            #b = em[len(box)-i-1]
            b = em[i]
            boxx = box[i]
            #print(b.shape)
            dist = 100
            res = ""
            for person in model:
                a = person['features'][0]
                #print("distance " + str(person['class']) + " " + str(distance.euclidean(a,b)))
                dis_temp = distance.euclidean(a,b)
                if(dis_temp < dist):
                    res = person['class']
                    dist = dis_temp
            if dist > 1.1: res = "unknown"
            print(res)
            draw(boxx, res, frame)
            cv2.imshow("Nhan dien", frame)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def detect_image(path):
    f = open("train_sort_v100_ssh_v2.pkl", "rb")
    model = pickle.load(f)
    print("len model "+ str(len(model)))
    x = FacialRecognition()
    img = cv2.imread(path)
    start = time.time()
    em,box=x.detect_face_and_get_embedding(img)
    for i in range(len(box)):
        # b = em[len(box)-i-1]
        b = em[i]
        boxx = box[i]
        dist = 100
        res = ""
        for person in model:
            a = person['features'][0]
            #print("distance " + str(person['class']) + " " + str(distance.euclidean(a,b)))
            dis_temp = distance.euclidean(a,b)
            if dis_temp < dist:
                res = person['class']
                dist = dis_temp
                if dist < 0.68:
                    break
        if dist > 1.11: res = "unknown"
        end = time.time()
        print("Thơi gian nhan dien anh la " + str(end - start))
        print(res)
        draw(boxx, res, img)
    cv2.imshow("anh", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    #detect_image("/home/manhas/Desktop/solution/77078572_2551237234953330_194496709236097024_n.jpg")

    #camera_detect()
    detect_and_tracking()