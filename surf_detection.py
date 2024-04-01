'''
    Tarea 6.1 - Color and Filtering
        En esta primera parte, buscamos implementar la determinacion que definen Ohta et al.
'''

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import cv2

# Funcion para determinar la imagen integral
def integral(img):

	output_image = np.zeros((img.shape[0], img.shape[1])) # Inicializando la imagen integral
	output_image[0,:], output_image[:,0] = img[0,:], img[:,0]

	# Rellenando imagen integral
	for i in range(1, img.shape[0]):
		for j in range(1, img.shape[1]):
			output_image[i,j] = img[i,j] + output_image[i-1,j] + output_image[i,j-1] - output_image[i-1,j-1]

	return output_image

# Obtenemos el Hessiano
def hessianDet(img, sigma):

	size = int(3*sigma)
	height = img.shape[0]
	width = img.shape[1]
	s_2 = int((size-1) / 2)
	s_3 = int(size / 3)
	hessian_det = np.zeros_like(img, dtype=np.double)
	w_i = 1.0 / size / size

	if size %2 == 0:
		size += 1

	for i in range(height):
		for j in range(width):

			top_left = integ_window(img, i-s_3, j-s_3, s_3, s_3) # Esquina superior izquierda
			bottom_right = integ_window(img, i+1, j+1, s_3, s_3) # Esquina inferior derecha
			bottom_left = integ_window(img, i-s_3, j+1, s_3, s_3) # Esquina inferior izquierda
			top_right = integ_window(img, i+1, j-s_3, s_3, s_3) # Esquina superior derecha

			d_xy = bottom_left + top_right - top_left - bottom_right
			d_xy = -d_xy * w_i

			middle = integ_window(img, i-s_3+1, j-s_2, 2*s_3-1, size) # En medio de la region o caja (eje x)
			side = integ_window(img, i-s_3+1, j-s_3/2, 2*s_3-1, s_3) # En los costados

			d_xx = middle - 3*side
			d_xx = -d_xx * w_i

			middle = integ_window(img, i-s_2, j-s_3+1, size, 2*s_3-1) # En medio de la region o caja (eje y)
			side = integ_window(img, i-s_3/2, j-s_3+1, s_3, 2*s_3-1) # En los costados

			d_yy = middle - 3*side
			d_yy = -d_yy * w_i

			hessian_det[i,j] = d_xx * d_yy - (0.912 * d_xy)**2

	return hessian_det

# Integracion sobre ventana 
def integ_window(img, row, col, row_l, col_l):

	row = int(limits(row, 0, img.shape[0]-1))
	col = int(limits(col, 0, img.shape[1]-1))

	row_l = int(limits(row+row_l, 0, img.shape[0]-1))
	col_l = int(limits(col+col_l, 0, img.shape[1]-1))

	ans = img[row,col] + img[row_l,col_l] - img[row,col_l] - img[row_l,col]

	if ans < 0:

		return 0

	return ans

# Limites de seccion
def limits(x, low, upper):

	if x > upper:

		return upper

	elif x < low:

		return low

	return x

# Funcion para obtener los maximos
def getMax(img, threshold, footprint):

	image_max = ndi.maximum_filter(img, footprint=footprint, mode='constant')
	
	out = img == image_max
	out &= img < threshold

	mask = out

	coordinates = np.nonzero(mask)
	intensities = img[coordinates]
	idx_maxsort = np.argsort(-intensities)
	coordinates = np.transpose(coordinates)[idx_maxsort]

	return coordinates

# Deteccion por SURF
def surf_detection(img, min_sigma, max_sigma, step_sigma):

	sigmas = np.linspace(min_sigma, max_sigma, step_sigma)

	# Primero obtenemos la imagen integral
	int_img = integral(img)

	# Ahora obtenemos el determinante de la matriz Hessian
	hessian = [hessianDet(int_img, sigma) for sigma in sigmas]
	# hessian = hessianDet(int_img, 3)

	# Obtenemos los picos maximo
	img2max = np.dstack(hessian)
	maximus = getMax(img2max, threshold=0.1, footprint=np.ones((3,)*img2max.ndim))

	key_points = maximus

	return key_points

# Inicio del programa...
if __name__ == "__main__":

	# Leyendo la imagen
	img = cv2.imread('images/1.jpeg', cv2.IMREAD_GRAYSCALE).astype(float)
	img2 = cv2.imread('images/1.jpeg')
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

	key_points = surf_detection(img, 1, 31, 5)

	plt.figure()
	plt.imshow(img2)
	plt.plot([p[1] for p in key_points],[p[0] for p in key_points],'*')
	plt.show()