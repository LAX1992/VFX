import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
#程式直接執行即可
#會print出從最底到最上層共五次計算出的位移方向+圖
def changeToExbitmap(img, threshold):
	img_array = img.copy()
	for index1, row in enumerate(img_array):
		for index2, ele in enumerate(row):
			if ele > threshold - 10 and ele < threshold + 10:
				img_array[index1][index2] = 0
			else:
				img_array[index1][index2] = 1
	return img_array

def changeToBinary(img, threshold):
	img_array = img.copy()
	for index1, row in enumerate(img_array):
		for index2, ele in enumerate(row):
			if ele < threshold:
				img_array[index1][index2] = 0
			else:
				img_array[index1][index2] = 1
	return img_array

def shift(matrix, dx, dy):
	matrix_copy = matrix.copy()
	for index1, row in enumerate(matrix_copy):
		for index2, ele in enumerate(row):
			if(index1+dx < matrix_copy.shape[0]-1 and index1+dx >= 0 and index2+dy < matrix_copy.shape[1]-1 and index2+dy >= 0):
				matrix_copy[index1][index2] = matrix_copy[index1+dx][index2+dy]
			else :
				continue
	return matrix_copy

#讀圖1移動到2
img_array1 = cv2.imread('alignment01.jpg', 0)
img_array2 = cv2.imread('alignment02.jpg', 0)
#最後要轉的圖
img_1 = cv2.imread('alignment01.jpg')
#找中位數
median_1 = np.median(img_array1)
median_2 = np.median(img_array2)
#找exclusive map
ExBitmap_1 = changeToExbitmap(img_array1, median_1)
ExBitmap_2 = changeToExbitmap(img_array2, median_2)

#thresholding
ThresBitmap_1 = changeToBinary(img_array1, median_1)
ThresBitmap_2 = changeToBinary(img_array2, median_2)

#開始建金字塔
Bitmap_resize_1 = []
Bitmap_resize_2 = []
ExBitmap_resize_1 = []
ExBitmap_resize_2 = []

for i in range(5):
	if i == 0:
		Bitmap_resize_1.append(cv2.resize(ThresBitmap_1,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
		Bitmap_resize_2.append(cv2.resize(ThresBitmap_2,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
		ExBitmap_resize_1.append(cv2.resize(ExBitmap_1,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
		ExBitmap_resize_2.append(cv2.resize(ExBitmap_2,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
	else:
		Bitmap_resize_1.append(cv2.resize(Bitmap_resize_1[i-1],None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
		Bitmap_resize_2.append(cv2.resize(Bitmap_resize_2[i-1],None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
		ExBitmap_resize_1.append(cv2.resize(ExBitmap_resize_1[i-1],None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
		ExBitmap_resize_2.append(cv2.resize(ExBitmap_resize_2[i-1],None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
Bitmap_resize_1 = np.array(Bitmap_resize_1)
Bitmap_resize_2 = np.array(Bitmap_resize_2)
ExBitmap_resize_1 = np.array(ExBitmap_resize_1)
ExBitmap_resize_2 = np.array(ExBitmap_resize_2)


record = np.zeros(2)
result = []
for photo_number in reversed(range(5)):
	error_count = sys.maxsize
	destBitmap = np.array(Bitmap_resize_2[photo_number])
	destEXmap = np.array(ExBitmap_resize_2[photo_number])
	Bitmap_mask = np.array(Bitmap_resize_1[photo_number])
	EXmap_mask = np.array(ExBitmap_resize_1[photo_number])
	if photo_number < 4:
		move_x = int(record[0])
		move_y = int(record[1])
		Bitmap_mask = shift(Bitmap_mask, move_x, move_y)
		EXmap_mask = shift(EXmap_mask, move_x, move_y)
	max_i = 0
	max_j = 0
	for i in range(-1, 2):
		for j in range(-1, 2):
			shiftBit_res = shift(Bitmap_mask, i, j)
			shiftExBit_res = shift(EXmap_mask, i, j)
			BitmapXOR = cv2.bitwise_xor(shiftBit_res, destBitmap)
			BitmapAND_1 = cv2.bitwise_and(BitmapXOR, destEXmap)
			BitmapAND_2 = cv2.bitwise_and(BitmapAND_1, shiftExBit_res)
			if(error_count > (np.sum(BitmapAND_2))):
				error_count = np.sum(BitmapAND_2)
				max_i = i
				max_j = j
	record[0] = record[0]*2 + max_i*2
	record[1] = record[1]*2 + max_j*2
	print("%d %d"%(max_i,max_j))
result = shift(img_1, int(record[0]), int(record[1]))
cv2.imshow("ya",result)
cv2.waitKey()
cv2.imwrite('mtb_output.jpg', result)