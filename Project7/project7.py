import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import copy

# Ns = 400
T = 10
c_list = [1, 10]
Ns_list = [100, 400]
# c = 1

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2.imshow("original", img)
    print(img.shape)
    # cv2.waitKey()
    return img

def find_lowest_gradient(grad_img, mx, my):
    a = -1
    b = -1
    lowest = math.inf
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            if np.linalg.norm(grad_img[mx+i, my+j]) < lowest:
                lowest = np.linalg.norm(grad_img[mx+i, my+j])
                a = mx+i
                b = my+j
    return a, b

## mean = [r,g,b,x,y] combine mean & mean_loc
def initial_mean(img, s, Ns):
    mean = [np.zeros((5, ))]*Ns
    # mean = np.zeros((Ns, 5))
    fix_center = [[]]*Ns
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gradient_mag = cv2.magnitude(Gx, Gy)
    count = 0
    row, col, _ = img.shape
    for i in range(int(row/s)):
        for j in range(int(col/s)):
            # m -> (i+1)*s-1, (j+1)*s-1
            x = (i+1)*s-1
            y = (j+1)*s-1
            fix_center[count] = [x, y]
            mean[count] = np.array(img[x, y].tolist()+[x,y]) #(5,) array BGRxy
            # boundary
            if x == (row-1) or y == (col-1):
                count += 1
                continue
            mx, my = find_lowest_gradient(gradient_mag, x, y)
            mean[count] = np.array(img[mx, my].tolist()+[mx,my])
            count += 1
    return mean, fix_center


def cal_dist(img, px, py, mean, s, c):
    dc = np.linalg.norm( mean[:3] - img[px, py] )
    ds = np.linalg.norm( mean[3:] - np.array([px, py]) )
    D = math.sqrt(dc**2 + c**2 * (ds/s)**2)
    return D
    

def update_cluster(img, mean, fix_center, s, Ns, c):
    row, col, _ =img.shape
    Nt = row*col
    L = np.array([-1]*Nt).reshape((row, col))
    D = np.array([np.inf]*Nt).reshape((row, col))
    error = np.inf
    counter = 0

    while error > T:
        counter += 1
        for cluster_idx in range(Ns):
            center = fix_center[cluster_idx]
            for j in range(-s+center[0]+1, s+center[0]+1):
                for k in range(-s+center[1]+1, s+center[1]+1):
                    if j >= row or k >= col:
                        continue
                    
                    d = cal_dist(img, j, k, mean[cluster_idx], s, c)
                    if d < D[j, k]:
                        D[j, k] = d
                        L[j, k] = cluster_idx
        
        ## update mean by L record
        m_sum = np.zeros((Ns, 5))
        m_counter = np.zeros((Ns, ))
        for i in range(row):
            for j in range(col):
                m_sum[L[i,j]] += np.hstack( (img[i, j], np.array([i, j])) )
                # print(m_sum[L[i,j]])
                # print((m_sum[L[i,j]]).shape)
                # exit(-1)
                m_counter[L[i,j]] += 1
        
        update_mean = [np.zeros((5, ))]*Ns
        error = 0 
        for k in range(Ns):
            m = m_sum[k] / m_counter[k]
            update_mean[k] = np.array(m, dtype=int)
            error += np.linalg.norm( update_mean[k]-mean[k] )
            # print(update_mean[0])
            # exit(-1)

        mean = copy.deepcopy(update_mean)
    
    print("Converges!")
    print("{} superpixels, need {} iteration.".format(Ns, counter))
    return mean, L


def superPixels_img(img, mean, L):
    row, col = L.shape
    super_img = img.copy()

    for i in range(row):
        for j in range(col):
            cluster = L[i, j]
            super_img[i, j] = mean[cluster][:3]
    
    return super_img

def scale_img(img, scale_range):
    # scale to [0:scale_range]
    img = img - np.min(img)
    img = img / np.max(img) * scale_range
    return img

def get_different_img(path_original, path_super, name, Ns, c):
    # path_original = os.path.join("material","image-pj7a.tif")
    # path_super = os.path.join("result","a_superImg_400_c=10.png")
  
    img = cv2.imread(path_original, cv2.IMREAD_GRAYSCALE)
    super_img = cv2.imread(path_super, cv2.IMREAD_GRAYSCALE)
    
    diff_img = cv2.subtract(img, super_img)
    # diff_img = img - super_img
    # cv2.imshow("diff", diff_img)
    # cv2.waitKey()
    # print(np.max(diff_img), np.min(diff_img))

    # class 'numpy.float64'>
    # print(type(img_normalize[0, 0]))
    # float must be converted to m, as uint8 to show
    img_normalize = scale_img(diff_img, 255)
    m = np.array(img_normalize, dtype=np.uint8)
    viz_img(m, "diff_"+name+"_"+str(Ns)+"_c="+str(c))
    # cv2.imshow("diff_n", m)
    # cv2.waitKey()


def viz_img(img, win_name):
    # path = os.path.join("result","diff", win_name+".png")
    path = os.path.join("result", win_name+".png")
    # cv2.imshow("Super img", img)
    cv2.imwrite(path, img)
    # cv2.waitKey()
    

if __name__ == "__main__":
    name_list = ["a","b","c","d"]

    ## get different superpixel images
    for name in name_list:
        path = os.path.join("material","image-pj7"+name+".tif")
        print("Processing",name)
        print(path)
        # path_a = os.path.join("material","image-pj7a.tif")
        # a:(600,600,3)
        img = load_img(path)
        Nt = img.shape[0] * img.shape[1]
        for Ns in Ns_list:
            # s = 60,30
            s = int(math.sqrt(Nt/Ns))
            
            mean, fix_center = initial_mean(img, s, Ns)
            for c in c_list:
                mean, L = update_cluster(img, mean, fix_center, s, Ns, c)
                super_img = superPixels_img(img, mean, L)
                viz_img(super_img, name+"_superImg_"+str(Ns)+"_c="+str(c))
                # diff_img = different_img(img, super_img)
    
    
    
    ## get_different_img
    for name in name_list:
        path_original = os.path.join("material","image-pj7"+name+".tif")
        print("Processing",name)
        print(path_original)
        for Ns in Ns_list:
            for c in c_list:
                path_super = os.path.join("result",name+"_superImg_"+str(Ns)+"_c="+str(c)+".png")
                get_different_img(path_original, path_super, name, Ns, c)
