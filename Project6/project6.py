import numpy as np
import cv2
import csv
import math
import os.path

TH = 0.10
TL = 0.04

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("original", img)
    print(img.shape)
    # cv2.waitKey()
    return img

def scale_img(img, scale_range):
    # scale to [0:scale_range]
    img_normalize = ( img - np.min(img) ) * scale_range / (np.max(img) - np.min(img))
    return img_normalize

def sobel_gradient(img):
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    print(np.max(Gx), np.min(Gx))
    # x_abs = cv2.convertScaleAbs(Gx)
    # y_abs = cv2.convertScaleAbs(Gy)
    x_abs = np.abs(Gx)
    y_abs = np.abs(Gy)
    print(np.max(x_abs), np.min(x_abs)) # 1,0
    scaled_x = scale_img(abs(x_abs), 255)
    scaled_y = scale_img(abs(y_abs), 255)
    x = scale_img(abs(x_abs), 1)
    y = scale_img(abs(y_abs), 1)
    # exit(-1)
    img_viz( scaled_x, "Gx")
    img_viz( scaled_y, "Gy")
    # print(Gx)
    # print(type(Gx), Gx.shape)
    # mag = cv2.addWeighted(scaled_x, 0.5, scaled_y, 0.5, 0)
    mag = cv2.magnitude(Gx, Gy)
    # mag = cv2.addWeighted(x, 0.5, y, 0.5, 0)
    # img_viz( mag, "Magnitude Image")
    img_viz( mag, "Magnitude Image")
    angle = cv2.phase(Gx, Gy)
    angle_mask = angle > np.pi
    angle[angle_mask] -= 2*np.pi
    # angle_2 = np.arctan2(Gy, Gx)
    img_viz(angle, "Angle Image")
    # print(np.max(angle)/np.pi) # 0 ~ 2*pi
    # print(np.min(angle)/np.pi)
    # img_viz(angle_2, "Angle_2 Image")
    return mag, angle

def nonMaximum(mag, angle):
    g_n = np.zeros(mag.shape)
    edge_type = np.full((mag.shape), -1)

    # sobel dir are +x:right +y:down
    edge_dir = [[[0,-1], [0,1]],
                    [[-1,1], [1,-1]],
                    [[-1,0], [1,0]],
                    [[1,1],[-1,-1]]
    ]
    
    # 1 : vertical
    # 2 : -45
    # 3 : horizontal
    # 4 : +45
    for row in range(angle.shape[0]):
        for col in range(angle.shape[1]):
            # deg = (angle[row][col] - np.pi) * 180 / np.pi
            deg = angle[row][col] * 180 / np.pi
            # print(deg)
          
            # search in gradient dir
            if deg < 0:
                if deg <= -157.5 or deg >= -22.5:
                    edge_type[row][col] = 2
                elif -157.5 <= deg <= -112.5:
                    edge_type[row][col] = 3
                elif -112.5 <= deg <= -67.5:
                    edge_type[row][col] = 0
                else:
                    edge_type[row][col] = 1
            else:
                if deg >= 157.5 or deg <= 22.5:
                    edge_type[row][col] = 2
                elif 22.5 <= deg <= 67.5:
                    edge_type[row][col] = 3
                elif 67.5 <= deg <= 112.5:
                    edge_type[row][col] = 0
                else:
                    edge_type[row][col] = 1
    
    for row in range(mag.shape[0]):
        for col in range(mag.shape[1]):
            neighbor = edge_dir[int(edge_type[row][col])]
            # print(neighbor)
            # deg = angle[row][col] * 180 / np.pi
            # print(deg)
            # print("At row:{}, col:{}".format(row, col))
            # print("Neighbor: ({},{}), ({},{})".format(row+neighbor[0][0], col+neighbor[0][1], row+neighbor[1][0],col+neighbor[1][1]))
            
            # if row >= 5:
            #     exit(-1)
            row_shit_1 = row+neighbor[0][1]
            row_shit_2 = row+neighbor[1][1]
            col_shit_1 = col+neighbor[0][0]
            col_shit_2 = col+neighbor[1][0]
            valid_1 = -100
            valid_2 = -100
            if (row_shit_1>=0) and (row_shit_1<mag.shape[0]) and (col_shit_1>=0) and (col_shit_1<mag.shape[1]):
                valid_1 = mag[row_shit_1][col_shit_1]
            if (row_shit_2>=0) and (row_shit_2<mag.shape[0]) and (col_shit_2>=0) and (col_shit_2<mag.shape[1]):
                valid_2 = mag[row_shit_2][col_shit_2]
           
            if (mag[row][col] >= valid_1) and (mag[row][col] >= valid_2):
                g_n[row][col] = mag[row][col]
            else :
                g_n[row][col] = 0

                
    img_viz(g_n, "g_n")
    return g_n

def hysteresis(g_n, TH, TL):
    gn_H = np.zeros(g_n.shape)
    gn_L = np.zeros(g_n.shape)
    for row in range(g_n.shape[0]):
        for col in range(g_n.shape[1]):
            if g_n[row][col] >= TH:
                gn_H[row][col] = g_n[row][col]
            elif TH > g_n[row][col] >= TL:
                gn_L[row][col] = g_n[row][col]
    img_viz(gn_H, "gn_H")
    img_viz(gn_L, "gn_L")

    edge_map = np.zeros(g_n.shape)
    for row in range(gn_H.shape[0]):
        for col in range(gn_H.shape[1]):
            if (gn_H[row][col]):
                edge_map[row][col] = 1
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        if (0 <= (row+i) < gn_H.shape[0]) and (0 <= (col+j) < gn_H.shape[1]):
                            if(gn_L[row+i][col+j]):
                                edge_map[row+i][col+j] = 1
    return edge_map
    



def img_viz(img, win):
    path = os.path.join("result", win + ".png")
    print(path)
    img_normalize = scale_img(img, 255)
    m = np.array(img_normalize, dtype=np.uint8)
    # cv2.imshow(win, m)
    cv2.imwrite(path, m)
    # cv2.waitKey()


if __name__ == "__main__":
    img_path = os.path.join("material", "image-pj6(Canny).tif")
    # 600x881x3
    img = load_img(img_path)
    img_scaled = scale_img(img, 1)
    # print(np.max(img_scaled), np.min(img_scaled))
    height, width = img_scaled.shape
    short_side = height if height <= width else width
    # 3
    sigma = int(0.005 * short_side)
    kernel_size = 6*sigma + 1  
    gauss_blur_img = cv2.GaussianBlur(img_scaled, (kernel_size, kernel_size), sigma)
    img_viz(gauss_blur_img, "Gaussian Blurred Image")
    mag, angle = sobel_gradient(gauss_blur_img)
    img_viz(mag, "Gradient Magnitude")

    g_n = nonMaximum(mag, angle)
    edge_map = hysteresis(g_n, TH = TH, TL = TL)
    img_viz(edge_map, "Edge Map")
    # print(type(gauss_blur_img[0][0]))
    canny_img = cv2.Canny(np.array(gauss_blur_img, dtype=np.uint8), TL, TH)
    cv2.imshow("canny", canny_img)
    cv2.waitKey()
    # img_viz(canny_img, "Canny Edge Map")

