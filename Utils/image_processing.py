import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import time


def draw_text(frame, text, font, text_origin, fill=(0, 0, 0), fix_org=True,text_size=30):
    t = time.time()
    image = Image.fromarray(frame)
    t = time.time()
    draw = ImageDraw.Draw(image)
    t = time.time()
    font = ImageFont.truetype(font=font,
                              size=np.floor(text_size).astype('int32'))

    # font = ImageFont.truetype(font=font,
    #                          size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    if fix_org:
        left = text_origin[0]
        top = text_origin[1]
        label_size = draw.textsize(text, font)
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
    t = time.time()
    draw.text(text_origin, text, fill=fill, font=font)
    result = np.array(image)
    return result


def merge_images(l_img, s_img, x_offset, y_offset):
    y1, y2 = y_offset ,y_offset + s_img.shape[0]
    x1, x2 = x_offset,x_offset + s_img.shape[1]
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])

    return l_img


def resize_img(img, w_new):
    h, w = img.shape[0], img.shape[1]
    h_new = w_new * h / w
    img = cv2.resize(img, (w_new, h_new))
    return img

def resize_img_2(img, max_size=700):
        h, w = img.shape[0], img.shape[1]
        if max(h, w) > max_size:
            scale = max(max(int(w / 600), int(h / 600)), 2)
            w_new = int(w / scale)
            h_new = int(h / scale)
            img = cv2.resize(img, (w_new, h_new))
            return img, scale
        return img, 1


def resize_remain_factor(img, max_size=1400):
    h, w = img.shape[0], img.shape[1]
    # print img.shape
    if max(w, h) > max_size:
        if w > h:
            w_new = max_size
            h_new = w_new * h / w
        else:
            h_new = max_size
            w_new = h_new * w / h
        img = cv2.resize(img, (int(w_new),int(h_new)))
        return img
    return img

