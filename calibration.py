from pylab import *
from misc import *
from triangulation import triangulate

def self_calibrate(F, w, h):
	# Compute the semi-calibrated fundamental matrix
	K = array([[2 * (w + h), 0, w / 2], [0, 2 * (w + h), h / 2], [0, 0, 1]])
	G = normalize_norm(dot(K.T, dot(F, K)))

	# Self-calibration using the Kruppa equations (Sturm's method)
	U, s, Vh = svd(G)
	fp = array([s[0]**2 * (1 - U[2, 0]**2) * (1 - Vh[0, 2]**2) - s[1]**2 * (1 - U[2, 1]**2) * (1 - Vh[1, 2]**2),
		s[0]**2 * (U[2, 0]**2 + Vh[0, 2]**2 - 2 * U[2, 0]**2 * Vh[0, 2]**2) - s[1]**2 * (U[2, 1]**2 + Vh[1, 2]**2 - 2 * U[2, 1]**2 * Vh[1, 2]**2),
		s[0]**2 * U[2, 0]**2 * Vh[0, 2]**2 - s[1]**2 * U[2, 1]**2 * Vh[1, 2]**2])

	rs = roots(fp)
	rs = real(rs[abs(imag(rs)) < 1e-6])
	rs = rs[rs > 0]
	
	f = 2 * (w + h)
	if any(abs(fp) > 1e-6) and len(rs) > 0:
		f = 2 * (w + h) * sqrt(rs[0])
	
	K = array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
	E = dot(K.T, dot(F, K))								# E = K.T * F * K
	return stabilize(normalize_norm(E)), K
	
def normalized_cameras_from_ematrix(E, feature1, feature2):
	# Extract the camera matrices
	W = array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	U, s, Vh = svd(E)
	P1 = eye(3, 4)										# P1 = I | 0
	P2s = array([										# P2 = [e2] * F | e2
		hstack((dot(U, dot(W, Vh)), U[:, -1].reshape(3, 1))),
		hstack((dot(U, dot(W, Vh)), -U[:, -1].reshape(3, 1))),
		hstack((dot(U, dot(W.T, Vh)), U[:, -1].reshape(3, 1))),
		hstack((dot(U, dot(W.T, Vh)), -U[:, -1].reshape(3, 1)))])

	# The second camera has positive depth for the points along with the first one
	xyz_test = homogeneous(array([triangulate(P1, P2i, feature1, feature2) for P2i in P2s]))
	xyz_P1 = array([dot(xyz_test[i], P1.T) for i in range(4)])
	xyz_P2 = array([dot(xyz_test[i], P2s[i].T) for i in range(4)])
	
	P2 = P2s[(xyz_P1[:, 2] > 0) & (xyz_P2[:, 2] > 0)][0]
	
	return P1, P2
	