from pylab import *
from misc import *
import scipy.optimize as opt

def planar_homography(F_ft, xy_from, xy_to):
	# Find the orienting homography
	epipole = homogeneous(kernel(F_ft, both = True)[1])
	
	amin, ier = opt.leastsq(lambda a: planar_homography_error(a, F_ft, epipole, xy_from, xy_to), array([0, 0, 0]),
		xtol = 1e-12, ftol = 1e-12, maxfev = 15000)
	
	H = dot(skew_symmetric(epipole), F_ft) - dot(epipole.reshape(3, 1), amin.reshape(1, 3))
	
	return H, epipole
	
def planar_homography_error(a, F, e2, xy1, xy2):
	H = dot(skew_symmetric(e2), F) - dot(e2.reshape(3, 1), a.reshape(1, 3))
	fit = homogeneous(dot(xy1, H.T))
	return sqrt(sum((fit - xy2)**2, axis = 1))
	
def setup_rectification(images, F, xy1, xy2):
	w, h = images.shape[2], images.shape[1]
	e1, e2 = kernel(F, both = True)
	
	# Choose second image as reference
	F_ft, xy_from, xy_to = F, xy1, xy2
	reference_image = 1
	rectified = False

	if at_infinity(e2):
		if at_infinity(e1):
			# If both images are rectified, do nothing
			rectified = True
		else:
			# If only the second epipole is infinite, swap the images
			F_ft, xy_from, xy_to = F_ft.T, xy_to, xy_from
			reference_image = 0
			
	if not at_infinity(e1) and not at_infinity(e2):
		# Transform to the image with the epipole closer to the image center
		e1, e2 = homogeneous(e1), homogeneous(e2)
		dist1 = (e1[0] - w / 2)**2 + (e1[1] - h / 2)**2
		dist2 = (e2[0] - w / 2)**2 + (e2[1] - h / 2)**2
		if dist2 > dist1:
			F_ft, xy_from, xy_to = F_ft.T, xy_to, xy_from
			reference_image = 0
	
	return reference_image, F_ft, xy_from, xy_to, rectified
	
def homography_transform(image, H, emptyflag = 1e3, samplingfunc = bilinear_color):
	w, h = image.shape[1], image.shape[0]
	bounds = array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]])
	bounds_H = homogeneous(dot(bounds, H.T))
	
	minx, miny = int(floor(min(bounds_H[:, 0]))), int(floor(min(bounds_H[:, 1])))
	maxx, maxy = int(ceil(max(bounds_H[:, 0]))), int(ceil(max(bounds_H[:, 1])))
	
	Hinv = inv(H)
	i, j = mgrid[minx:maxx, miny:maxy]
	ij = vstack((i.flatten(), j.flatten(), ones(i.shape).flatten())).T
	ij_H = homogeneous(dot(ij, Hinv.T))
	
	# Transform the second image to be oriented with the first
	image_H = tile(emptyflag, (maxy - miny, maxx - minx))
	for i in range(ij.shape[0]):
		if point_in_quad(bounds, ij_H[i]):
			image_H[ij[i, 1] - miny, ij[i, 0] - minx] = samplingfunc(image, ij_H[i, 0], ij_H[i, 1])
	
	del i, j, ij, ij_H
	return image_H, bounds, bounds_H
	
def common_region(epipole, bounds_from, bounds_to):
	# Find oriented epipolar lines to the corners of both images and their angles
	lines1 = cross(epipole, bounds_from)
	lines2 = cross(epipole, bounds_to)
	angles1 = array([arctan2(-lines1[i, 0], lines1[i, 1]) for i in range(4)])
	angles2 = array([arctan2(-lines2[i, 0], lines2[i, 1]) for i in range(4)])
	minangle1, maxangle1 = min(angles1), max(angles1)
	minangle2, maxangle2 = min(angles2), max(angles2)

	# Find the common region
	if point_in_quad(bounds_from, epipole):
		if point_in_quad(bounds_to, epipole):
			minangle, maxangle = -pi, pi
		else:
			minangle, maxangle = minangle2, maxangle2
	elif point_in_quad(bounds_to, epipole):
		minangle, maxangle = minangle1, maxangle1
	else:
		if abs(minangle1 - minangle2) <= pi:
			minangle = max(minangle1, minangle2)
		else:
			minangle = min(minangle1, minangle2)
		if abs(maxangle1 - maxangle2) <= pi:
			maxangle = min(maxangle1, maxangle2)
		else:
			maxangle = max(maxangle1, maxangle2)

	flip = False
	if argmin(angles1) >= 2 and argmin(angles2) >= 2:
		flip = True
	
	if maxangle - minangle > pi and maxangle - minangle < 2 * pi:
		minangle, maxangle = maxangle, minangle + 2 * pi
		flip = True
		
	return minangle, maxangle, flip
	
def rectification_table(H, epipole, w, h):
	# Prepare the rectification table
	bounds_to = array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]])
	bounds_from = homogeneous(dot(bounds_to, H.T))
	box1 = cross(bounds_from, bounds_from[r_[1:4, 0]])
	box2 = cross(bounds_to, bounds_to[r_[1:4, 0]])
	
	# Find the common region in both images
	minangle, maxangle, flip = common_region(epipole, bounds_from, bounds_to)
	
	# Set the rectification table up
	rectify = array([])
	mindist, maxdist = 1e50, -1e50
	angle = minangle
	step = 1e50
	while angle <= maxangle:
		rectify = append(rectify, angle)
		
		# Find the epipolar line
		m = tan(angle)
		l = array([m, -1, epipole[1] - m * epipole[0]])		# mx - y - b = 0
		
		# Intersection with first image and their distances
		points = homogeneous(cross(box1, l))
		intersection = array([point_in_quad(bounds_from, xy) for xy in points])
		dist = sqrt(sum((points[intersection] - epipole)**2, axis = 1))
		
		if dist.shape[0] > 0:
			mindist = min(mindist, min(dist))
			maxdist = max(maxdist, max(dist))
			step = arctan2(1.0, max(dist))
		
		# Intersection with second image and their distances
		points = homogeneous(cross(box2, l))
		intersection = array([point_in_quad(bounds_to, xy) for xy in points])
		dist = sqrt(sum((points[intersection] - epipole)**2, axis = 1))
		
		if dist.shape[0] > 0:
			mindist = min(mindist, min(dist))
			maxdist = max(maxdist, max(dist))
			step = min(step, arctan2(1.0, max(dist)))
		
		angle = angle + step
	
	return rectify, flip, mindist, maxdist
	
def rectify_image(image, Hinv, epipole, rect_table, emptyflag = 1e3, samplingfunc = bilinear_color):
	rectify, flip, mindist, maxdist = rect_table
	xsize = int(ceil(maxdist - mindist))
	ysize = rectify.shape[0]
	
	rect = tile(emptyflag, (ysize, xsize))
	
	for y in range(ysize):
		# Find the epipolar line angle
		angle = rectify[y]
		dx = cos(angle)
		dy = sin(angle)
		
		# Fill the current scanline
		pixel0 = array([epipole[0] + dx * mindist, epipole[1] + dy * mindist, 1])
		
		for x in range(xsize):
			pixel = homogeneous(dot(Hinv, pixel0))
			
			if pixel[0] >= 0 and pixel[0] < image.shape[1] and pixel[1] >= 0 and pixel[1] < image.shape[0]:
				rect[y, x] = samplingfunc(image, pixel[0], pixel[1])
			
			pixel0[0] += dx
			pixel0[1] += dy
	
	if flip:
		rect = flipud(fliplr(rect))
	
	return rect
	
def unrectify_image(image, H, epipole, w, h, rect_table, emptyflag = 1e3, samplingfunc = bilinear_color):
	rectify, flip, mindist, maxdist = rect_table
	
	if flip:
		image = flipud(fliplr(image))
	
	# Fill in the unrectified image
	unrect = tile(emptyflag, (h, w))
	
	for y in range(h):
		for x in range(w):
			p = homogeneous(dot(H, array([x, y, 1])))
			if p[0] >= 0 and p[0] < w and p[1] >= 0 and p[1] < h:
				l = cross(epipole, p)
				angle = arctan2(-l[0], l[1])
				if flip and angle < 0:
					angle += 2 * pi
				
				# Find the distance to the boundary in the direction of the epipole
				dist = sqrt(sum((p - epipole)**2)) - mindist
				
				# Find the linearly interpolated y from the angle
				y_1 = argmin(abs(rectify - angle))
				angle_1 = rectify[y_1]
				if angle > angle_1: y_2 = min(y_1 + 1, rectify.shape[0] - 1)
				else: y_2 = max(0, y_1 - 1)
				
				angle_2 = rectify[y_2]
				if y_1 == y_2: yrect = y_1
				else: yrect = (abs(angle - angle_1) * y_2 + abs(angle_2 - angle) * y_1) / abs(angle_2 - angle_1)
				
				unrect[y, x] = samplingfunc(image, dist, yrect)
	
	return unrect
	
def mapping_homography(self, xy1, xy2):
	A = vstack([array(
		[[xy1[i, 0], xy1[i, 1], 1, 0, 0, 0, -xy1[i, 0] * xy2[i, 0], -xy1[i, 1] * xy2[i, 0], -xy2[i, 0]],
		[0, 0, 0, xy1[i, 0], xy1[i, 1], 1, -xy1[i, 0] * xy2[i, 1], -xy1[i, 1] * xy2[i, 1], -xy2[i, 1]]])
		for i in range(xy1.shape[0])])
		
	return kernel(A).reshape(3, 3)
	