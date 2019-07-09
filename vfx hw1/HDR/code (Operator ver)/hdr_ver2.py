import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve2d
#此份為較大的lambda配上local operation的code
def hdr_parameter(img_list, exposure_times):
    log_exposure_times = [math.log(e,2) for e in exposure_times]
    l = 100
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]
    Z_list = [img.flatten() for img in img_list]
    Z = np.array(Z_list)
    
    return Z, log_exposure_times, l, w

def sampleIntensities(images):
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1
    num_images = len(images)
    rows = len(layer_stack[0])-1
    cols = len(layer_stack[0][0])-1
    # Intensity_values matrix即pdf上的Z
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    for i in range(z_min, z_max + 1):
        idx_1 = random.randint(0, rows)
        idx_2 = random.randint(0, cols)
        for j in range(num_images):
            intensity_values[i, j] = images[j][idx_1, idx_2]
    
    return intensity_values

def response_curve_solver(Z, B, smoothing_lambda, w):
    n = 256
    A = np.zeros(shape=(Z.shape[0]*Z.shape[1]+n+1, n+Z.shape[0]))
    b = np.zeros(shape=(A.shape[0], 1))

    k = 0
    for i in range(np.size(Z, 0)):
        for j in range(np.size(Z, 1)):
            z = Z[i][j]
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k][0] = wij*B[j]
            k += 1
    
    A[k][128] = 1
    k += 1

    for i in range(n-1):
        A[k][i]   =    smoothing_lambda*w[i+1]
        A[k][i+1] = -2*smoothing_lambda*w[i+1]
        A[k][i+2] =    smoothing_lambda*w[i+1]
        k += 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:256]
    lE = x[256:]
    
    return g, lE

def construct_hdr(img_list, response_curve, exposure_times):
    # Get image size in each channel
    img_size = img_list[0][0].shape
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]
    ln_t = np.log2(exposure_times)
    # 用在最後將ln_E取exp
    vfunc = np.vectorize(lambda x:math.exp(x))
    # Hdr_radiance_map
    hdr = np.zeros((img_size[0], img_size[1], 3), dtype='float32')

    # Construct radiance map for BGR channels
    for i in range(3):
        # Z = [b channel, g channel, r channel]
        Z = [img.flatten().tolist() for img in img_list[i]]
        ln_E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(vfunc(ln_E), img_size)

    return hdr

def construct_radiance_map(g, Z, ln_t, w):
    # Denominator分母, Numerator分子
    Denominator = [0]*len(Z[0])
    ln_E = [0]*len(Z[0])
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        Numerator = 0
        for j in range(imgs):
            z = Z[j][i]
            Denominator[i] += w[z]*(g[z] - ln_t[j])
            Numerator += w[z]
        if Numerator > 0:
            ln_E[i] = Denominator[i] / Numerator
        else: 
            ln_E[i] = Denominator[i]
        Numerator = 0
    
    return ln_E

def tonemapping_global(hdrpic, s, a):
    height, weight, channel = hdrpic.shape
    N = height * weight
    delta = 0.00001
     
    L_w = 0.2125 * hdrpic[:,:,0] + 0.7154 * hdrpic[:,:,1] + 0.0721 * hdrpic[:,:,2] 
    L_w_mean = math.exp( sum(sum( np.log(delta + L_w) ))/N ) 
    
    key_scale = a / L_w_mean 
    
    L = key_scale * hdrpic 
    sL_w = key_scale * L_w 
    L_final = sL_w / (1 + sL_w) 

    ldrpic = np.zeros(hdrpic.shape)

    for i in range(3):
        ldrpic[:,:,i] = (((hdrpic[:,:,i] * ( 1/L_w ))) ** s ) * L_final
        
    return ldrpic

def tonemapping_local(hdrpic, sat, a):
    height, weight, channel = hdrpic.shape
    N = height * weight
    delta = 0.00001
      
    L_w = 0.2125 * hdrpic[:,:,0] + 0.7154 * hdrpic[:,:,1] + 0.0721 * hdrpic[:,:,2] 
    
    level, phi = 8, 8 
    
    v1 = np.zeros((L_w.shape[0], L_w.shape[1], level), dtype='float32')
    v = np.zeros((L_w.shape[0], L_w.shape[1], level), dtype='float32')
    v1_sm = np.zeros((L_w.shape[0], L_w.shape[1]), dtype='float32')
    
    for scale in range(level): 
        mask = matlab_style_gauss2D((43, 43), 0.5)
        v1[:,:,scale] = convolve2d(L_w, mask, 'same')
        mask = matlab_style_gauss2D((43, 43), 0.5)
        v2 = convolve2d(L_w, mask, 'same')
        if scale == 0:
            v[:,:,scale] = (v1[:,:,scale] - v2) /  v1[:,:,scale]
        else:
            v[:,:,scale] = (v1[:,:,scale] - v2) / (((2 ** phi * 0.36) / (scale)**2) + v1[:,:,scale])
    
    
    sm = np.zeros((L_w.shape[0], L_w.shape[1]))-1
    tmp = np.ones((L_w.shape[0], L_w.shape[1]))
    
    for scale in range(level):
        target = tmp * (abs(v[:,:,scale]) < 0.05)
        tmp = tmp - target
        sm[target==1] = scale

    sm[sm == -1] = 0
    
    for x in range(v1.shape[0]):
        for y in range(v1.shape[1]):
            v1_sm[x,y] = v1[x, y, int(sm[x,y])]

    L_w_mean = math.exp( sum(sum( np.log(delta + v1_sm) ))/N ) 
    key_scale = a / L_w_mean 
    L = key_scale * v1_sm 
    sL_w = key_scale * L_w 
    L_d = sL_w / (1 + L)  
    
    ldrpic = np.zeros(hdrpic.shape)

    for i in range(3):
        ldrpic[:,:,i] = (((hdrpic[:,:,i] * ( 1/L_w ))) ** sat ) * L_d
    
    ldrpic[ldrpic > 1] = 1
    
    return ldrpic

def matlab_style_gauss2D(shape=(3,3), sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# Main----------------------------------------------------------------------------------------------------
ROOT = os.getcwd()
# 把目標所在資料夾加進路徑
# 換資料夾只需把所有'wall'的字串換成新資料夾名稱即可
DATA = os.path.join(ROOT, 'wall')
dataset = np.vstack([np.expand_dims(np.array(cv2.imread(os.path.join(DATA, img))), axis=0) \
	    for img in os.listdir(os.path.join(DATA)) ])
# 每張圖對應的曝光時間
exposure_times = [ 1/15, 1/30, 1/60, 1/125, 1/250, 1/500, 1/1000]
# G_function計算Debevec------------------------------------------------------------------------------------
print('Calculate g function')
g_function = []
for channel in range(3):
    layer_stack = [img[:, :, channel] for img in dataset]
    intensity_samples = sampleIntensities(layer_stack)
    intensity_samples = np.array(intensity_samples)
    Z, log_exposure_times, smoothing_lambda, w = hdr_parameter(intensity_samples, exposure_times)
    g, le = response_curve_solver(Z, log_exposure_times, smoothing_lambda, w)
    g_function.append(g)
g_function = np.array(g_function)
# 存圖於Result資料夾----------------------------------------------------------------------------------------
path = os.path.join(ROOT, 'Result')
if not os.path.isdir(path):
    os.mkdir(path)
# Response curve-------------------------------------------------------------------------------------------
print('Draw response curve')
plt.figure(figsize=(10, 10))
# fmt = '[color][marker][line]'
plt.plot(g_function[2], range(256), 'rx')
plt.plot(g_function[1], range(256), 'gx')
plt.plot(g_function[0], range(256), 'bx')
plt.title('response_curve')
plt.ylabel('pixel value Z')
plt.xlabel('log exposure X')
plt.savefig(os.path.join(path, 'response_curve_'+'wall.jpg'))
#Construct hdr--------------------------------------------------------------------------------------------
print('Construct HDR image')
img_list_b = dataset[:, :, :, 0]
img_list_g = dataset[:, :, :, 1]
img_list_r = dataset[:, :, :, 2]
hdr_image = construct_hdr([img_list_b, img_list_g, img_list_r], g_function, exposure_times)
#cv2.imwrite(os.path.join(path, 'hdr_'+'wall.hdr'), hdr_image)
# Radiance map--------------------------------------------------------------------------------------------
print('Draw radiance map')
plt.figure(figsize=(12, 12))
plt.imshow(np.log(cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)), cmap='jet') 
plt.colorbar()
plt.savefig(os.path.join(path, 'radiance_map_'+'wall.png'))
# Tone mapping--------------------------------------------------------------------------------------------
'''
print('Tone mapping global')
ldrDrago = tonemapping_global(hdr_image, 0.5, 1)
tonemapping_globa = np.clip(ldrDrago*255, 0, 255).astype('uint8')
cv2.imwrite(os.path.join(path, 'tonemap_global_'+'wall.png'), tonemapping_globa)
'''
print('Tone mapping')
ldrDrago = tonemapping_local(hdr_image, 0.5, 1)
tonemapping = np.clip(ldrDrago*255, 0, 255).astype('uint8')
cv2.imwrite(os.path.join(path, 'tonemap_'+'wall.png'), tonemapping)
print('Done')