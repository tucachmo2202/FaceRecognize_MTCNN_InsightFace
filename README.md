# FaceRecognize_MTCNN_InsightFace
Nhận diện mặt người với MTCNN và InsightFace

Tải Pre_train model của insightFace về thư mục model-r100-ii: https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0
Cài đặt các gói cần thiết.
Tại thư mục FaceRecognize_MTCNN_InsightFace mở terminal lên và chạy lệnh Make

Để tạo dữ liệu mặt mới vào thư mục mydata và tạo thư mục với tên người cần nhận diện, sau đó đưa ảnh mặt người đó vào
Chạy file write_train_file.py sau đó chạy lệnh get_train_embedding.py để train khuôn mặt.
Cuối cùng chạy file test.py để nhận diện mặt người.
