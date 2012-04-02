from pylab import *
from misc import kernel, reduce_rank

class FundamentalMatrixModel:
	def fit(self, xy1, xy2):
		A = vstack([array(
			[xy1[i, 0] * xy2[i, 0], xy1[i, 1] * xy2[i, 0], xy2[i, 0], xy1[i, 0] * xy2[i, 1], xy1[i, 1] * xy2[i, 1], xy2[i, 1], xy1[i, 0], xy1[i, 1], 1])
			for i in range(xy1.shape[0])])
		
		return reduce_rank(kernel(A).reshape(3, 3))
		
	def error(self, model, xy1, xy2):
		# Sampson distance (first-order approximation to geometric error)
		xy2_fit = dot(xy1, model.T)					# F * m
		xy1_fit = dot(xy2, model)					# F.T * m'
		xy2_f_xy1 = sum(xy1_fit * xy1, axis = 1)	# m'.T * F * m
		return sqrt(xy2_f_xy1**2 / (xy2_fit[:, 0]**2 + xy2_fit[:, 1]**2 + xy1_fit[:, 0]**2 + xy1_fit[:, 1]**2))
		