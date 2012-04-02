from pylab import *
import scipy.linalg as alg

def matrix_rank(A, tol = 1e-6):
	s = svd(A, compute_uv = 0)
	return sum(abs(s / max(s)) > tol)
	
def kernel(A, both = False):
	U, s, Vh = svd(A)
	if both: return Vh.T[:, -1], U[:, -1]
	else: return Vh.T[:, -1]
	
def reduce_rank(A, n = 1):
	U, s, Vh = svd(A)
	return dot(dot(U, diag(hstack((s[:-n], zeros(n))))), Vh)
	
def skew_symmetric(x):
	return array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
	
def bilinear_color(image, x, y):
	x1, x2 = floor(x), min(ceil(x), image.shape[1] - 1)
	y1, y2 = floor(y), min(ceil(y), image.shape[0] - 1)
	
	wx1, wx2, wy1, wy2 = 0.5, 0.5, 0.5, 0.5
	if x1 != x2:
		wx1, wx2 = x2 - x, x - x1
	if y1 != y2:
		wy1, wy2 = y2 - y, y - y1
	
	return int(round(sum([
		image[y1, x1] * wy1 * wx1,
		image[y1, x2] * wy1 * wx2,
		image[y2, x1] * wy2 * wx1,
		image[y2, x2] * wy2 * wx2])))
	
def nearest_color(image, x, y):
	x1, x2 = floor(x), min(ceil(x), image.shape[1] - 1)
	y1, y2 = floor(y), min(ceil(y), image.shape[0] - 1)
	
	if x - x1 < 0.5: fx = x1
	else: fx = x2
	
	if y - y1 < 0.5: fy = y1
	else: fy = y2
	
	return image[fy, fx]
	
def point_in_quad(bounds, point):
	diff = bounds - point
	s = array([diff[0, 0] * diff[1, 1] - diff[1, 0] * diff[0, 1],
		diff[1, 0] * diff[2, 1] - diff[2, 0] * diff[1, 1],
		diff[2, 0] * diff[3, 1] - diff[3, 0] * diff[2, 1],
		diff[3, 0] * diff[0, 1] - diff[0, 0] * diff[3, 1]])
	s = s[abs(s) > 1e-6]
	return abs(sum(sign(s))) == len(s)
	
def normalize_angle(a):
	while a < -pi:
		a += 2 * pi
	while a > pi:
		a -= 2 * pi
	return a
	
def homogeneous(v):
	if len(v.shape) == 1:
		return v / v[-1]
	else:
		return v / v[:, -1].reshape(v.shape[0], 1)
	
def stabilize(x, tol = 1e-6):
	xs = x.copy()
	xs[abs(xs) < tol] = 0
	return xs
	
def normalize_norm(A):
	return A / alg.norm(A)
	
def at_infinity(p, tol = 1e-6):
	return abs(p[-1] / max(abs(p))) < tol
	