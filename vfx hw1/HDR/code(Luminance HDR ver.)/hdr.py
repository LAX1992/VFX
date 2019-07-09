import os
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

IMAGE_NUM = 5
# Images are of resolution 3968*2232 RGB
# Shutter speed 1/60, 1/125, 1/500, 1/1000, 1/2000
S_HEIGHT = 11
S_WIDTH = 20
M_HEIGHT = 1080
M_WIDTH = 1920

#delta_t = [1/60, 1/125, 1/500, 1/1000, 1/2000] #library
#delta_t = [1/60, 1/250, 1/500, 1/2000] #bridge
#delta_t = [1/15, 1/30, 1/125, 1/250, 1/500] #library2
#delta_t = [1/30, 1/60, 1/125, 1/500, 1/2000] #building
#delta_t = [1/15, 1/30, 1/60, 1/125, 1/250, 1/500] #machine
#delta_t = [1/30, 1/60, 1/125, 1/250, 1/1000] #bread
delta_t = [1/30, 1/60, 1/125, 1/250, 1/500] #statue

def downScaleImage(file, size, path):
    img = cv2.imread(file)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img)
    return

def recoverRadiance(samples, log_delta_t, l):
    b_vector = np.zeros(255 + IMAGE_NUM*S_HEIGHT*S_WIDTH)
    A_matrix = np.zeros((255 + IMAGE_NUM*S_HEIGHT*S_WIDTH, S_HEIGHT*S_WIDTH + 256))

    k = 0
    for i in range(IMAGE_NUM):
        for j in range(S_HEIGHT*S_WIDTH):
            A_matrix[k, samples[i*S_HEIGHT*S_WIDTH + j]] = 1
            A_matrix[k, 256 + j] = -1
            b_vector[k] = log_delta_t[i]
            k = k+1

    A_matrix[k, 128] = 1
    k = k+1

    # Smooth constraint
    for i in range(254):
        A_matrix[k,i] = l*hat_weighting(i); 
        A_matrix[k,i+1] = -2*l*hat_weighting(i+1)
        A_matrix[k,i+2] = l*hat_weighting(i+2)
        k = k+1

    x_least = np.matmul(np.linalg.pinv(A_matrix), b_vector.T)

    return x_least

def hat_weighting(px_value):
    p_max, p_min = 255, 0

    if px_value >= 128:
        return 256 - px_value
    else:
        return px_value + 1


this_dir, this_file = os.path.split(os.path.abspath(__file__))
#shrink the images to 11*20
shrunk_img = np.zeros((IMAGE_NUM, S_HEIGHT, S_WIDTH, 3), dtype=np.uint8)
log_delta_t = np.zeros(IMAGE_NUM)
for i in range(IMAGE_NUM):
    file = os.path.join(this_dir, "vfx", sys.argv[1], str(i+1)+".JPG")
    downScaleImage(file, (S_WIDTH, S_HEIGHT), os.path.join(this_dir, str(i+1)+"-test.jpg"))
    downScaleImage(file, (M_WIDTH, M_HEIGHT), os.path.join(this_dir, "resize", str(i+1)+".jpg"))
    file = os.path.join(this_dir, str(i+1)+"-test.jpg")
    shrunk_img[i] = cv2.imread(file)
    log_delta_t[i] = np.math.log(delta_t[i])

hdr_mapping = np.zeros((3, 256))
for i in range(3):
    single_channel_img = shrunk_img[:,:,:,i]
    x = recoverRadiance(single_channel_img.flatten(), log_delta_t, 20)
    hdr_mapping[i] = x[0:256]

full_img = np.zeros((IMAGE_NUM, M_HEIGHT, M_WIDTH,3), dtype=np.uint8)
for i in range(IMAGE_NUM):
    file = os.path.join(this_dir, "resize", str(i+1)+".jpg")
    full_img[i] = cv2.imread(file)

hdr_img = np.zeros((M_HEIGHT, M_WIDTH, 3), dtype=np.float32)
for i in range(M_HEIGHT):
    for j in range(M_WIDTH):
        for channel in range(3):
            v_sum, w_sum = 0, 0
            for k in range(IMAGE_NUM):
                w = hat_weighting(full_img[k,i,j,channel])
                w_sum += w
                v_sum += w*(hdr_mapping[channel, full_img[k,i,j,channel]] - log_delta_t[k])

            hdr_img[i,j,channel] = np.math.exp(v_sum / w_sum)

cv2.imwrite(sys.argv[1]+".hdr", hdr_img)