from pylab import *
from gcopt import *
import scipy.optimize as opt

def energy_terms(rect_from, rect_to, disprange):
	w, h = rect_from.shape[1], rect_from.shape[0]
	nlabels = disprange.shape[0]
	
	# Data term: abs(rect_from[p] - rect_from[p - d])
	ys, xs, ls, i, xr = range(h), range(w), range(nlabels), 0, 0
	data0 = tile(255, w * h * nlabels)
	for y in ys:
		for x in xs:
			for l in ls:
				xr = x - disprange[l]
				if xr >= 0 and xr < w and rect_from[y, x] < 256 and rect_to[y, xr] < 256:
					data0[i] = abs(rect_from[y, x] - rect_to[y, xr])
				i += 1
	
	# Smoothness term: abs(l1 - l2)
	i = 0
	smooth0 = empty(nlabels * nlabels)
	for l1 in ls:
		for l2 in ls:
			smooth0[i] = abs(disprange[l1] - disprange[l2])
			i += 1
	
	#flag = rect_from.max()
	#if flag > 255:
	#	if flag == rect_to.max():
	#		rect_to[find(flag)] = flag * 2
	
	#before = minimum(disprange, 0)
	#after = maximum(disprange, 0)
	#dummy = tile(1e3, (h, 1))
	
	# Data term: abs(rect_from[p] - rect_to[p - l])
	#data0 = abs(repeat(rect_from.flatten(), nlabels).flatten() -
	#	vstack([hstack((repeat(dummy, after[i], axis = 1), rect_to[:, -before[i]:w - after[i]], repeat(dummy, -before[i], axis = 1))).flatten()
	#	for i in range(nlabels)]).T.flatten())
	
	# Smoothness term: delta(l1 - l2)
	#l1, l2 = mgrid[0:nlabels, 0:nlabels]
	#smooth0 = abs(disprange[l1.flatten()] - disprange[l2.flatten()])
	
	return data0, smooth0, w, h, nlabels

def disparity_map(rect_from, rect_to, disprange, tol = 2.5e-2, maxiter = 20):
	# Calculate initial data and smoothness terms
	data0, smooth0, w, h, nlabels = energy_terms(rect_from, rect_to, disprange)
	
	# Parameter estimation for the disparity map (Zhang and Seitz)
	mixture_params = array([0.9, 0.1, 256, 0.9, 0.1, nlabels])
	
	# Mixture model parameters to regularized energy parameters
	new_mrf_params = mrf_from_mixture(*mixture_params)
	mrf_params = new_mrf_params + new_mrf_params
	iter = 0

	while any(abs(new_mrf_params - mrf_params) / mrf_params > tol) and iter < maxiter:
		# Generate the depth map from the rectified images
		mrf_params = new_mrf_params
		disparity = disparity_map_params(data0, smooth0, w, h, disprange, *mrf_params)
		
		# New estimates using the disparity map
		mixture_params = mixture_from_disparity(rect_from, rect_to, disparity, *mixture_params)
		new_mrf_params = mrf_from_mixture(*mixture_params)
		
		iter += 1
		print iter, mixture_params, new_mrf_params
	
	del data0, smooth0
	return disparity, mrf_params, iter

def disparity_map_params(data0, smooth0, w, h, disprange, maxdata, maxsmooth, weighting):
	g = GridGraph(w, h, disprange.shape[0])
	
	data = minimum(data0, maxdata)
	smooth = weighting * minimum(smooth0, maxsmooth)
	
	g.set_data_cost(data)
	g.set_smooth_cost(smooth)
	g.expansion()
	disparity = disprange[g.labels()].reshape(h, w)
	
	del data, smooth, g
	return disparity
	
def mrf_from_mixture(match_prob, match_decay, match_max, disp_prob, disp_decay, disp_max):
	match_norm = (1 - exp(-match_decay)) / (1 - exp(-match_decay * match_max))
	sd = match_prob * match_norm * match_decay / (match_prob * match_norm + (1 - match_prob) / match_max)
	td = log(1 + match_prob * match_norm * match_max / (1 - match_prob))
	
	disp_norm = (1 - exp(-disp_decay)) / (1 - exp(-disp_decay * disp_max))
	sp = disp_prob * disp_norm * disp_decay / (disp_prob * disp_norm + (1 - disp_prob) / disp_max)
	tp = log(1 + disp_prob * disp_norm * disp_max / (1 - disp_prob))

	maxdata, maxsmooth, weighting = td / sd, tp / sp, sp / sd
	return array([maxdata, maxsmooth, weighting])
	
def mixture_from_disparity(rect_from, rect_to, disparity, match_prob, match_decay, match_max, disp_prob, disp_decay, disp_max):
	w, h = rect_from.shape[1], rect_from.shape[0]
	match_norm = (1 - exp(-match_decay)) / (1 - exp(-match_decay * match_max))
	disp_norm = (1 - exp(-disp_decay)) / (1 - exp(-disp_decay * disp_max))
	
	# The conditional probability of matching errors
	new_match_max, wd_sum, wd_weighted_sum, wd_count, xr = 0.0, 0.0, 0.0, 0.0, 0
	for y in range(h):
		for x in range(w):
			xr = x - disparity[y, x]
			if xr >= 0 and xr < w and rect_from[y, x] < 256 and rect_to[y, xr] < 256:
				match_delta = abs(rect_from[y, x] - rect_to[y, xr])
				if match_delta > new_match_max:
					new_match_max = match_delta
				tmp = match_prob * match_norm * exp(-match_decay * match_delta)
				wd = tmp / (tmp + (1 - match_prob) / match_max)
				wd_sum += wd
				wd_weighted_sum += wd * match_delta
				wd_count += 1

	match_max = new_match_max + 1
	match_prob = wd_sum / wd_count
	match_y = wd_weighted_sum / wd_sum
	match_decay = opt.newton(lambda decay: 1 / (exp(decay) - 1) - 1 / (exp(decay * match_max) - 1) - match_y, log(1 / match_y + 1))
	
	# The conditional probability of disparity differences
	new_disp_max, wp_sum, wp_weighted_sum, wp_count = 0.0, 0.0, 0.0, 0.0
	for y in range(h):
		for x in range(w):
			xr = x - disparity[y, x]
			if xr >= 0 and xr < w and rect_from[y, x] < 256 and rect_to[y, xr] < 256:
				if x != w - 1:
					disp_delta = abs(disparity[y, x] - disparity[y, x + 1])
					if disp_delta > new_disp_max:
						new_disp_max = disp_delta
					tmp = disp_prob * disp_norm * exp(-disp_decay * disp_delta)
					wp = tmp / (tmp + (1 - disp_prob) / disp_max)
					wp_sum += wp
					wp_weighted_sum += wp * disp_delta
					wp_count += 1
				if y != h - 1:
					disp_delta = abs(disparity[y, x] - disparity[y + 1, x])
					if disp_delta > new_disp_max:
						new_disp_max = disp_delta
					tmp = disp_prob * disp_norm * exp(-disp_decay * disp_delta)
					wp = tmp / (tmp + (1 - disp_prob) / disp_max)
					wp_sum += wp
					wp_weighted_sum += wp * disp_delta
					wp_count += 1
	
	disp_max = new_disp_max + 1
	disp_prob = wp_sum / wp_count
	disp_y = wp_weighted_sum / wp_sum
	disp_decay = opt.newton(lambda decay: 1 / (exp(decay) - 1) - 1 / (exp(decay * disp_max) - 1) - disp_y, log(1 / disp_y + 1))
	
	return array([match_prob, match_decay, match_max, disp_prob, disp_decay, disp_max])
	