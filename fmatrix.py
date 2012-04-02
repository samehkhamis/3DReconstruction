from pylab import *
from misc import *
from ransac import *
from fundamental import *
import scipy.optimize as opt

def fmatrix(xy1, xy2):
	# Normalize the coordinates of the features
	T1 = normalization_matrix(xy1)
	T2 = normalization_matrix(xy2)
	xy1_T1 = dot(xy1, T1.T)		# T1 * X
	xy2_T2 = dot(xy2, T2.T)		# T2 * X'

	# RANSAC to reject outliers and LM initialized with RANSAC output
	F0_T, inliers = ransac(FundamentalMatrixModel(), xy1_T1, xy2_T2, 8, 1e-3)
	F0 = dot(T2.T, dot(F0_T, T1))
	
	Fp0, P1, P2 = params_from_fmatrix(F0)
	
	xy1_inliers = dot(xy1, P1.T)[inliers]
	xy2_inliers = dot(xy2, P2.T)[inliers]
	PZPT1 = dot(P1, dot(diag([1, 1, 0]), P1.T))
	PZPT2 = dot(P2, dot(diag([1, 1, 0]), P2.T))
	Fp, ier = opt.leastsq(lambda params: fundamental_matrix_error(fmatrix_from_params(params), PZPT1, PZPT2, xy1_inliers, xy2_inliers), Fp0,
		xtol = 1e-6, ftol = 1e-6, maxfev = 15000)
	F = dot(P2.T, dot(fmatrix_from_params(Fp), P1))
	
	del xy1_T1, xy2_T2
	return stabilize(normalize_norm(F)), inliers
	
def normalization_matrix(xy):
	x = xy[:, 0]
	y = xy[:, 1]
	xm = mean(x)
	ym = mean(y)
	denom = sum(sqrt((x - xm)**2 + (y - ym)**2))
	s = sqrt(2) * len(x) / denom
	return array([[s, 0, -s * xm], [0, s, -s * ym], [0, 0, 1]])
	
def fundamental_matrix_error(Fhat, PZPT1, PZPT2, xy1, xy2):
	# Zhang's method
	xy2_fit = dot(xy1, Fhat.T)					# F * m
	xy1_fit = dot(xy2, Fhat)					# F.T * m'
	xy2_f_xy1 = sum(xy1_fit * xy1, axis = 1)	# m'.T * F * m
	denom = sum(xy2_fit * dot(xy2_fit, PZPT1.T), axis = 1) + sum(xy1_fit * dot(xy1_fit, PZPT2.T), axis = 1)
	return sqrt(xy2_f_xy1**2 / denom)
	
def fmatrix_from_params(params):
	a, b, c, x1, y1, x2, y2 = params
	
	return array([
		[1, a, - x1 - a * y1],
		[b, c, -b * x1 - c * y1],
		[-x2 - b * y2, -a * x2 - c * y2, (x1 + a * y1) * x2 + (b * x1 + c * y1) * y2]])
	
def params_from_fmatrix(F):
	# Zhang's method
	P1 = r_[:3]
	P2 = r_[:3]
	e1, e2 = kernel(F, both = True)
	m = argmax(abs(F))
	i0, j0 = m / 3, m % 3
	
	if j0 != 0:
		P1[j0], P1[0] = P1[0], P1[j0]
		e1[j0], e1[0] = e1[0], e1[j0]
	if i0 != 0:
		P2[i0], P2[0] = P2[0], P2[i0]
		e2[i0], e2[0] = e2[0], e2[i0]
	if abs(e1[1]) > abs(e1[2]):
		P1[2], P1[1] = P1[1], P1[2]
		e1[2], e1[1] = e1[1], e1[2]
	if abs(e2[1]) > abs(e2[2]):
		P2[2], P2[1] = P2[1], P2[2]
		e2[2], e2[1] = e2[1], e2[2]
	
	P1 = eye(3)[P1]
	P2 = eye(3)[P2]
	Fhat = dot(inv(P2.T), dot(F, inv(P1)))
	params = array([Fhat[0, 1] / Fhat[0, 0], Fhat[1, 0] / Fhat[0, 0], Fhat[1, 1] / Fhat[0, 0], e1[0] / e1[2], e1[1] / e1[2], e2[0] / e2[2], e2[1] / e2[2]])
	return params, P1, P2
	