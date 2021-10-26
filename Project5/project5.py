import numpy as np
import cv2
import csv
import cmath
import os.path

## in RGB (0-255)
a1 = (134, 51, 143)
a2 = (131, 132, 4)
R = 30

def load_img(path):
    img = cv2.imread(path)
    img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("original", img)
    cv2.imshow("hsi", img_hsi)
    print(img.shape)
    # print(img_hsi.shape)
    # print(img_hsi)
    cv2.waitKey()
    return img, img_hsi

def img_viz(img, win):
    path = os.path.join("result", win + ".png")
    print(path)
    # print(np.max(img) - np.min(img))
    # img_normalize = ( img - np.min(img) ) * 255 / (np.max(img) - np.min(img))
    img_normalize = img
    # print(np.max(img_normalize), np.min(img_normalize))
    # print(img_normalize.shape)
    m = np.array(img_normalize, dtype=np.uint8)
    cv2.imshow(win, m)
    cv2.imwrite(path, m)
    cv2.waitKey()


def color_slicing(img, a, R):
    row, col, chanal = img.shape

    img_rgb = img[...,::-1]
    img_out = np.zeros((img.shape))
    for i in range(row):
        for j in range(col):
            dist = 0
            for color in range(chanal):
                diff = (a[color] - img_rgb[i, j, color])**2
                dist += diff
            if dist > R*R:
                img_out[i, j] = [0.5*256, 0.5*256, 0.5*256]
            else:
                img_out[i, j] = img[i, j]
    
    return img_out



if __name__ == "__main__":
    path = os.path.join("material","violet (color).tif")
    img, img_hsi = load_img(path)
    # exit(-1)
    [H, S, I] = cv2.split(img_hsi)
    print(np.max(H), np.min(H))
    img_viz(H, "H component")
    img_viz(S, "S component")
    img_viz(I, "I component")
     
    img_sliced_1 = color_slicing(img, a1, R)
    img_viz(img_sliced_1, "a1")
    img_sliced_2 = color_slicing(img, a2, R)
    img_viz(img_sliced_2, "a2")