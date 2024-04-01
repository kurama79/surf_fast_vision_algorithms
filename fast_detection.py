'''
    Tarea 6.1 - Color and Filtering
        En esta primera parte, buscamos implementar la determinacion que definen Ohta et al.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Funcion para obtener el circulo alrededor de pixel
def circle(x, y):

    p1 = (x+3, y)
    
    p3 = (x+3, y-1)
    
    p5 = (x+1, y+3)
    
    p7 = (x-1, y+3)
    
    p9 = (x-3, y)
    
    p11 = (x-3, y-1)
    
    p13 = (x+1, y-3)
    
    p15 = (x-1, y-3)

    current_circle = [p1, p3, p5, p7, p9, p11, p13, p15]
    
    return current_circle;

# Funcion para determinar si una region es una esquina 
def is_corner(img, row, col, current_circle, threshold):

    intensity = int(img[row,col])

    row_top, col_top = current_circle[0]
    row_bottom, col_bottom = current_circle[4]
    row_right, col_right = current_circle[2]
    row_left, col_left = current_circle[6]

    intensity_top = int(img[row_top,col_top])
    intensity_bottom = int(img[row_bottom,col_bottom])
    intensity_right = int(img[row_right,col_right])
    intensity_left = int(img[row_left,col_left])

    count = 0
    if abs(intensity_top - intensity) > threshold:
        count += 1 

    if abs(intensity_bottom - intensity) > threshold:
        count += 1

    if abs(intensity_right - intensity) > threshold:
        count += 1

    if abs(intensity_left - intensity) > threshold:
        count += 1

    return count >= 3

# Funcion para ver si dos puntos estan muy cercanos
def tooClose(p1, p2, val=4):

    x1, y1 = p1
    x2, y2 = p2

    x = x2 - x1
    y = y2 - y1

    return np.sqrt(x**2 + y**2) <= val

# Calculo de V para Non-maximal suppression
def get_V(img, p, current_circle):

    row, col = p
    intensity = int(img[row,col])

    row_top, col_top = current_circle[0]
    row_topR, col_topR = current_circle[1]
    row_right, col_right = current_circle[2]
    row_bottomR, col_bottomR = current_circle[3]
    row_bottom, col_bottom = current_circle[4]
    row_bottomL, col_bottomL = current_circle[5]
    row_left, col_left = current_circle[6]
    row_topL, col_topL = current_circle[7]

    intensity_top = int(img[row_top,col_top])
    intensity_topR = int(img[row_topR,col_topR])
    intensity_right = int(img[row_right,col_right])
    intensity_bottomR = int(img[row_bottomR,col_bottomR])
    intensity_bottom = int(img[row_bottom,col_bottom])
    intensity_bottomL = int(img[row_bottomL,col_bottomL])
    intensity_left = int(img[row_left,col_left])
    intensity_topL = int(img[row_topL,col_topL])  

    V = abs(intensity - intensity_top) + abs(intensity - intensity_topL) + \
        abs(intensity - intensity_right) + abs(intensity - intensity_bottomR) + \
        abs(intensity - intensity_bottom) + abs(intensity - intensity_bottomL) + \
        abs(intensity - intensity_left) + abs(intensity - intensity_topL)

    return V

# Funcion Non-maximal suppression
def non_maxima(img, corners, current_circle):

    i = 1
    while i < len(corners):

        currPoint = corners[i]
        prevPoint = corners[i-1]

        if tooClose(prevPoint, currPoint):

            currScore = get_V(img, currPoint, current_circle)
            prevScore = get_V(img, prevPoint, current_circle)

            if (currScore > prevScore):
                del(corners[i-1])

            else:
                del(corners[i])

        else:        	
            i += 1
            # continue

    return corners

# Inicio del algoritmo FAST
def fast_detection(image, threshold=50):

    corners = []
    height, width = image.shape[0:2]

    # Buscamos por pixel delimitado
    for i in range(int(height/9), int(8*height/9)):
        for j in range(int(width/9), int(8*width/9)):

            current_circle = circle(i,j) 

            if is_corner(image, i, j, current_circle, threshold):
                corners.append((i,j))

    corners = non_maxima(image, corners, current_circle) 

    return corners;

# Inicio del programa...
if __name__ == "__main__":

	# Leyendo la imagen
	img = cv2.imread('images/cinves.jpeg', cv2.IMREAD_GRAYSCALE).astype(float)
	img2 = cv2.imread('images/cinves.jpeg')
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

	key_points = fast_detection(img)

	plt.figure()
	plt.imshow(img2)
	plt.plot([p[1] for p in key_points],[p[0] for p in key_points],'*')
	plt.show()