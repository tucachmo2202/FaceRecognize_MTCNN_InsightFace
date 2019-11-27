# coding=utf-8
import mxnet as mx
import numpy as np
from Arcface import ArcfaceModel
from face_preprocess import preprocess
import cv2
from mtcnn.mtcnn import MTCNN
from Sshdetector import SSHDetector
from OnetLnet import OnetLnetAlignment


class FacialRecognition():
    def __init__(self, gpu_index=-1, mtcnn_model="mtcnn-model", arcface_model="model-r100-ii/model,0",
                 image_size='112,112', ssh_detector="ssh-model-final/sshb", mtcnn_num_worker=1):
        if gpu_index >= 0:
            mtcnn_ctx = mx.gpu(gpu_index)
        else:
            mtcnn_ctx = mx.cpu()
        #self.face_detector = SSHDetector(prefix=ssh_detector, epoch=0, ctx_id=gpu_index, test_mode=True)
        self.face_recognition = ArcfaceModel(gpu=gpu_index, model=arcface_model, image_size=image_size)
        #self.landmark_detector = OnetLnetAlignment(model_folder=mtcnn_model, ctx=mtcnn_ctx, num_worker=mtcnn_num_worker,
        #                                           accurate_landmark=True, threshold=[0.6, 0.7, 0.5])
        self.face_detect = MTCNN()


    def get_scales(self, img):
        TEST_SCALES = [100, 200, 300, 400]
        target_size = 400
        max_size = 1200
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]
        return scales

    def detect_face_and_get_embedding(self, img):
        thresh = 0.2
        embeddings=[]
        scales = self.get_scales(img)
        #bboxes = self.face_detector.detect(img, threshold=thresh, scales=scales)
        result = self.face_detect.detect_faces(img)   
        bboxes = []
        points = []
        if (len(result) != 0):
            for person in result:
                bb = person['box'] #Chuyen ve dang bb[0], bb[1], bb[0] + bb[2], bb[1]+ bb[3], precision
                precision = person['confidence']
                bboxes.append([bb[0],bb[1],bb[0] +bb[2],bb[1]+ bb[3],precision])
                keypoints = person['keypoints']
                points.append([keypoints['left_eye'][0],keypoints['right_eye'][0],keypoints['nose'][0],keypoints['mouth_left'][0],keypoints['mouth_right'][0],keypoints['left_eye'][1],keypoints['right_eye'][1],keypoints['nose'][1],keypoints['mouth_left'][1],keypoints['mouth_right'][1]])
            points = np.array(points)
            bboxes = np.array(bboxes)
            if len(bboxes) <= 0:
                return None
            #rs = self.landmark_detector.detect_landmark(img, bboxes)
            #if rs is not None:
            #_, points = rs
            for i in range(len(points)):
                point = points[i, :].reshape((2, 5)).T
                #print("point: " + str(point))
                nimg = preprocess(img, bboxes[i], point, image_size='112,112')
                #nimg = cv2.resize(img[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])], (112,112))
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                
                # cv2.imshow("image",img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                x = np.transpose(nimg, (2, 0, 1))
                embeddings.append( self.face_recognition.get_feature(x))
            return embeddings,bboxes
        return None

    def get_embedding(self, img):
        nimg = cv2.resize(img, (112, 112))
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        x = np.transpose(nimg, (2, 0, 1))
        embeddings = self.face_recognition.get_feature(x)
        return embeddings