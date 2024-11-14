'''
Visualize the spatial and temporal kernels learned by a trained EEGNet. User specifies an 
EEGNet model and the kernel weights are visualized.

Brandon McMahan
September 8, 2021
'''
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
from EEGNet import EEGNet

# get PyTorch model name from user input
#parser = argparse.ArgumentParser(description="Visualizes the features in a trained EEGNet")
#parser.add_argument('model_name', help="name of PyTorch model")
#parser.add_argument('latency', help="latency of pytorch model")
#args = parser.parse_args()
#F1 = int(args.model_name[-3])
#D = int(args.model_name[-1])
#n_tsteps = int(args.latency)

#print("F1:", F1)
#print("D:", D)
# load trained PyTorch model

name = "delete__EEGNet-8-2__lr0.001__latency2000__n_classes4__batch_size128__datasetBM_dataset_2__lossmodified"
pytorch_model = EEGNet(F1=8, D=2, output_dim=4, n_tsteps=2000)
pytorch_model.load_state_dict(torch.load("trained_models/" + name))

# extract the relevant weights
def extract_temporal_weights(pytorch_model):
	'''
	extracts the temporal weights learned by the model. Returns a 
	NumPy array of size (n_filters, 500) that contain each of the 
	filter weights
	'''
	n_filters = pytorch_model._conv1.weight.data.shape[0]  # number of temporal filters learned by model
	filter_weights = np.zeros((n_filters, 50))
	for kernel in range(n_filters):
		filter_weights[kernel, :] = pytorch_model._conv1.weight.data[kernel, 0, 0, :].detach().cpu().numpy()
	return filter_weights

def extract_spatial_weights(pytorch_model):
	'''
	Extracts the spatial weights from a PyTorch model. Returns a
	NumPy array with shape (F1, D, 59)
	where F1 is the number of temporal filters and D is the number 
	of spatial filters per temporal filter.
	'''

	# extract the spatial weights from the model
	spatial_kernel = pytorch_model._depthwise.weight.data.detach().cpu().numpy()  # (F1*D, 1, 59, 1)
	spatial_kernel = spatial_kernel[:, 0, :, 0]   # (F1*D, 59)

	# extract F1 and D from weight shapes
	F1 = pytorch_model._conv1.weight.data.shape[0]
	D = int(spatial_kernel.shape[0] / F1)

	# format the spatial weights
	spatial_filters = np.zeros((F1, D, 64))
	for temporal_filter_ix in range(F1):  # loop over temporal filters
		for spatial_filter_ix in range(D):  # loop over all spatial filters that correspond to this temporal filter
			spatial_filters[temporal_filter_ix, spatial_filter_ix] = spatial_kernel[D*temporal_filter_ix + spatial_filter_ix]

	return spatial_filters

# visualize the weights
def visualize_spatial_filters(spatial_kernel):
	''' 
	Draws the spatial filter on the current plot

	spatial_kernel is a NumPy array of weights with shape (59,)
	'''
	inds_original = np.array([
		[0, 4], [0, 5], [0, 6], [2, 1], [2, 3], [2, 5], [2, 7], [2, 9],
		[3, 2], [3, 4], [3, 6], [3, 8], [4, 0], [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [4, 10], 
		[5, 2], [5, 4], [5, 6], [5, 8], [6, 1], [6, 3], [6, 5], [6, 7], [6, 9],
		[7, 5], [8, 4], [8, 6], [0, 0], [1, 1], [1, 3], [1, 7], [1, 9],
		[2, 2], [2, 4], [2, 6], [2, 8], [3, 3], [3, 5], [3, 7],
		[4, 2], [4, 4], [4, 6], [4, 8], [5, 3], [5, 7],
		[6, 2], [6, 4], [6, 6], [6, 8],
		[7, 2], [7, 3], [7, 7], [7, 8],
		[3, 1], [3, 9], [5, 1], [5, 9], [7, 1], [7, 9], [8, 5]
	])
	grid = np.zeros((9, 11))
	for electrode_ix in range(64):  # i think the first two electrodes and final three electrodes get cropped off
		x_pos = inds_original[electrode_ix][0]
		y_pos = inds_original[electrode_ix][1]
		grid[x_pos, y_pos] = spatial_kernel[electrode_ix]
	plt.imshow(grid)

def visualize_temporal_filters(filter_weights):
	plt.plot(np.linspace(0, 0.5, 50), filter_weights)

def visualize_features(pytorch_model):
	filter_weights = extract_temporal_weights(pytorch_model)  # (F1, 59)
	spatial_filters = extract_spatial_weights(pytorch_model)  # (F1, D, 59)

	# extract EEGNet hyper-parameters
	F1, D, n_electrodes = spatial_filters.shape

	# set up the image
	n_rows = D+1  # row for each spatial filter plus the temporal filter
	n_cols = F1   # column for each temporal filter

	# loop over temporal filters
	plt.figure(figsize=[12.4, 9.6])
	for temporal_filter_ix in range(F1):
		# plot the temporal filter
		plt.subplot(n_rows, n_cols, temporal_filter_ix+1)
		plt.title("Filter #" + str(temporal_filter_ix))
		visualize_temporal_filters(filter_weights[temporal_filter_ix, :])
		# loop over spatial filters
		for spatial_filter_ix in range(D):
			plt.subplot(n_rows, n_cols, (spatial_filter_ix+1)*F1+temporal_filter_ix+1)
			visualize_spatial_filters(spatial_filters[temporal_filter_ix, spatial_filter_ix, :])
	plt.savefig("trained_models/" + name + "_features.png") 
	plt.show()




visualize_features(pytorch_model)
