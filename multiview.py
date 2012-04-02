import sift
from pylab import *
from misc import *
from calibration import *
from triangulation import *
from fmatrix import fmatrix
from rectification import *
from visualization import *
from stereo import *

# Estimate the fundamental matrix between image pairs
#files = array(['TsukubaL.pgm', 'TsukubaR.pgm'])
files = array(['a1.pgm', 'a2.pgm'])
images = array([flipud(imread(f)) for f in files])
w, h = images.shape[2], images.shape[1]

# Detect SIFT features and match the images
sift.detect(files[0], '0.key')
sift.detect(files[1], '1.key')
xy1, xy2 = sift.match('0.key', '1.key')
xy1 = hstack((xy1, ones((len(xy1), 1)))).round()
xy2 = hstack((xy2, ones((len(xy2), 1)))).round()

# Estimate the fundamental matrix
F, inliers = fmatrix(xy1, xy2)

# Correct the projection of the features to sub-pixel accuracy
xy1_corrected, xy2_corrected = correct_projection(F, xy1[inliers],  xy2[inliers])

# Polar rectification
reference_image, F_ft, xy_from, xy_to, rectified = setup_rectification(images, F, xy1_corrected, xy2_corrected)
image_from, image_to = images[1 - reference_image], images[reference_image]

if rectified:
	# Images are already rectified, do nothing
	rect_from, rect_to = image_from, image_to
else:
	# Orient the epipolar lines, create the rectification table, and use it
	H, epipole = planar_homography(F_ft, xy_from, xy_to)
	rect_table = rectification_table(H, epipole, w, h)
	rect_from = rectify_image(image_from, inv(H), epipole, rect_table)
	rect_to = rectify_image(image_to, eye(3, 3), epipole, rect_table)

# Self-calibrate the cameras
E, K = self_calibrate(F_ft, w, h)
P1, P2 = normalized_cameras_from_ematrix(E, xy_from[0], xy_to[0])
P1 = dot(K, P1)
P2 = dot(K, P2)

# Estimate the disparity ramge using the features
if rectified:
	dispdiff = xy_from[:, 0] - xy_to[:, 0]
else:
	xy_from_H = homogeneous(dot(xy_from, H.T))
	dispdiff = sqrt(sum((xy_from_H - epipole)**2, axis = 1)) - sqrt(sum((xy_to - epipole)**2, axis = 1))

disprange = arange(int(floor(min(dispdiff) / 1.25)), int(ceil(max(dispdiff) * 1.25)))

# Estimate the disparity map
rect_disparity, mrf_params, dummy = disparity_map(rect_from, rect_to, disprange)
if rectified:
	disparity = rect_disparity
else:
	disparity = unrectify_image(rect_disparity, H, epipole, w, h, rect_table, emptyflag = rect_disparity.min(), samplingfunc = nearest_color)

# Model building
f = K[0, 0]
b = alg.norm(P1[:, 3] - P2[:, 3])
Z = f * b / (disparity + b * 0.5)
Y, X = mgrid[0:h, 0:w]
X = X * Z / f
Y = Y * Z / f

from enthought.tvtk.tools import mlab
mlab.figure(browser = False).add(mlab.Surf(X, Y, Z, image_from))

#mlab.figure(browser = False).add(mlab.ImShow(disparity, show_scalar_bar = True))

#from scipy.io.matlab import mio
#mio.savemat('disparity.mat', {'disparity': disparity, 'F_ft': F_ft, 'rect_from': rect_from, 'rect_to': rect_to, 'Z': Z, 'X': X, 'Y': Y, 'image': image_from})
#MATLAB: warp(X, Y, -Z, image)

#draw_image_features(image_from, xy1, xy_from)
#draw_image_features(image_to, xy2, xy_to)
#draw_image_epipolar(image_from, xy_to, F_ft, H, (minx, miny))
#draw_image_epipolar(image_to, xy_from, F_ft.T)
#draw_image(rect_from)
#draw_image(rect_to)
#draw_image(disparity)
