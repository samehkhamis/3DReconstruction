from pylab import *
import scipy.misc as misc

def save_image(image, filename):
	im = where(image > 255, 0, image)
	misc.imsave(filename, im)
	
def draw_image(image):
	im = where(image > 255, 0, image)
	
	figure()
	imshow(im, cmap = cm.gray, extent = [0, im.shape[1], im.shape[0], 0])
	
def draw_image_features(image, all, inliers):
	im = where(image > 255, 0, image)
	
	figure()
	plot(all[:, 0], all[:, 1], 'm.')
	plot(inliers[:, 0], inliers[:, 1], 'r.')
	imshow(im, cmap = cm.gray, extent = [0, im.shape[1], im.shape[0], 0])
	
def draw_image_epipolar(image1, features2, F12, H = eye(3, 3), offset = (0, 0)):
	im = where(image1 > 255, 0, image1)
	
	Hinv = inv(H)
	figure()
	x = array([0, 10, im.shape[1] - 1])
	for i in range(features2.shape[0]):
		l = dot(features2[i], F12)
		l = dot(l, Hinv)				# l1_new = Hinv.T * (m2.T * F) = Hinv.T * l1
		y = (-l[2] - l[0] * x) / l[1]
		plot(x - offset[0], y - offset[1], 'r-')
		
	imshow(im, cmap = cm.gray, extent = [0, im.shape[1], im.shape[0], 0])
	