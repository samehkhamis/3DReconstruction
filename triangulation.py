from pylab import *
from misc import *

def correct_projection(F, xy1_inliers, xy2_inliers):
	xy1_corrected = zeros(xy1_inliers.shape)
	xy2_corrected = zeros(xy2_inliers.shape)
	
	for i in range(xy1_inliers.shape[0]):
		xy1_corrected[i], xy2_corrected[i] = correct_projection_for_point(F, xy1_inliers[i], xy2_inliers[i])
	
	return xy1_corrected, xy2_corrected
	
def correct_projection_for_point(F, xy1, xy2):
	T1 = array([[1, 0, -xy1[0]], [0, 1, -xy1[1]], [0, 0, 1]])
	T2 = array([[1, 0, -xy2[0]], [0, 1, -xy2[1]], [0, 0, 1]])
	FT = dot(dot(inv(T2.T), F), inv(T1))
	e1T, e2T = kernel(FT, both = True)
	e1T = e1T / sqrt(1 - e1T[2]**2)
	e2T = e2T / sqrt(1 - e2T[2]**2)
	R1 = array([[e1T[0], e1T[1], 0], [-e1T[1], e1T[0], 0], [0, 0, 1]])
	R2 = array([[e2T[0], e2T[1], 0], [-e2T[1], e2T[0], 0], [0, 0, 1]])
	FR = dot(dot(R2, FT), R1.T)
	
	# Set up the polynomial g(t) whose roots are extremas of s(t)
	a, b, c, d, f1, f2 = FR[1, 1], FR[1, 2], FR[2, 1], FR[2, 2], e1T[2], e2T[2]
	rs = roots([-(a * d - b * c) * f1**4 * a * c,
		((a**2 + f2**2 * c**2)**2 - (a * d - b * c) * f1**4 * b * c - (a * d - b * c) * f1**4 * a * d),
		(2 * (2 * b * a + 2 * f2**2 * d * c) * (a**2 + f2**2 * c**2) - 2 * (a * d - b * c) * f1**2 * a * c - (a * d - b * c) * f1**4 * b * d),
		(-2 * (a * d - b * c) * f1**2 * b * c - 2 * (a * d - b * c) * f1**2 * a * d + 2 * (b**2 + f2**2 * d**2) * (a**2 + f2**2 * c**2) + (2 * b * a + 2 * f2**2 * d * c)**2),
		(-(a * d - b * c) * a * c - 2 * (a * d - b * c) * f1**2 * b * d + 2 * (b**2 + f2**2 * d**2) * (2 * b * a + 2 * f2**2 * d * c)),
		((b**2 + f2**2 * d**2)**2 - (a * d - b * c) * b * c-(a * d - b * c) * a * d),
		-(a * d - b * c) * b * d])
		
	rs = real(rs[abs(imag(rs)) < 1e-6])
	st = lambda t: t**2 / (1 + f1**2 * t**2) + (c * t + d)**2 / ((a * t + b)**2 + f2**2 * (c * t + d)**2)
	stval = st(rs)
	stmin = min(stval)
	stinf = 1 / f1**2 + c**2 / (a**2 + f2**2 * c**2)
	tmin = rs[argmin(stval)]
	
	# Is the minimum at t = inf?
	if stmin > stinf:
		l1 = array([f1, 0, -1])
		l2 = array([-f2 * c, a, c])
	else:
		l1 = array([tmin * f1, 1, -tmin])
		l2 = array([-f2 * (c * tmin + d), a * tmin + b, c * tmin + d])
	
	# Compute the corrected coordinates and find X = xyz
	xhat1 = array([-l1[0] * l1[2], -l1[1] * l1[2], l1[0]**2 + l1[1]**2])
	xhat2 = array([-l2[0] * l2[2], -l2[1] * l2[2], l2[0]**2 + l2[1]**2])
	xhat1 = dot(dot(inv(T1), R1.T), xhat1)
	xhat2 = dot(dot(inv(T2), R2.T), xhat2)
	xhat1 = xhat1 / xhat1[2]
	xhat2 = xhat2 / xhat2[2]
	return xhat1, xhat2
	
def triangulate(P1, P2, xy1, xy2):
	xy1 = homogeneous(xy1)
	xy2 = homogeneous(xy2)
	return kernel(vstack([
		[xy1[0] * P1[2] - P1[0]],
		[xy1[1] * P1[2] - P1[1]],
		[xy2[0] * P2[2] - P2[0]],
		[xy2[1] * P2[2] - P2[1]]]))
	