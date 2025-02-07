#############################################
# This script samples the conductivity of a mesh on some predetermined slice.
# The sampled values are then visualized and used to extract tissue boundaries.

# NOTE: This script assumes that you have already run the "visualize_potential_on_slice.py"
#       script first. The output generated by that script is needed for this script. Furthermore,
#       this script is tailored to the volume conductor shown in Figure 4.2 in the paper,
#       since we explicitely specify the map from conductivities to colors.
#############################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import matplotlib.cm
import matplotlib.gridspec
import math
import skimage
import os.path

#######################################
# Visualization parameters
#######################################

# This script assumes that the script "visualize_potential_on_slice.py" as already been run.
# Please specify the path to the npz archive generated by this script below.
slice_data_path = 'TO_FILL_IN'

# If the flag below is set to "True", this script estimates the tissue boundaries and adds
# them to the plot.
add_boundary = True

# Depending on the sampling resolution, the boundary will be too thin to be seen in the visualization. To make the boundary visible
# in the image, we perform a number of binary dilations. The parameter below specifies the number of dilations performed.
# If you change the sampling resolution in "visualize_potential_on_slice.py", you might have to adapt this parameter.
nr_dilations = 2

#######################################
# Create image
#######################################

# load data
slice_data = np.load(slice_data_path)
position_coordinates = slice_data['position_coordinates']
nr_positions = len(position_coordinates)
grid_indices = slice_data['grid_indices']
grid_delta = slice_data['grid_delta']
lower_left_bbox = slice_data['lower_left_bbox']
upper_right_bbox = slice_data['upper_right_bbox']
sigma_values = slice_data['sigma_values']

# compute image boundaries
x_extent = upper_right_bbox[0] - lower_left_bbox[0]
y_extent = upper_right_bbox[1] - lower_left_bbox[1]

nr_x_cells = math.floor(x_extent / grid_delta[0]) + 1
nr_y_cells = math.floor(y_extent / grid_delta[1]) + 1

lower_left_corner = lower_left_bbox[:2] - 0.5 * grid_delta
upper_right_corner = lower_left_corner[:2] + np.array([nr_x_cells * grid_delta[0], nr_y_cells * grid_delta[1]])
potential_image_extent = [lower_left_corner[0], upper_right_corner[0], lower_left_corner[1], upper_right_corner[1]]

# set up image storage
head_mask = np.full((nr_y_cells, nr_x_cells, 4), True)

sigma_image = np.empty((nr_y_cells, nr_x_cells), dtype=np.float64)
sigma_image[:] = 0.0

sigma_rgba = np.empty((nr_y_cells, nr_x_cells, 4), dtype = np.float64)
sigma_rgba[:] = 0.0

# create custom colormaps
conductivities = np.array([8.000e-06,
                           2.500e-05,
                           1.260e-04,
                           2.750e-04,
                           4.650e-04,
                           6.000e-04,
                           0.0016539999999999999])

# the order here is important. We assume that conductivity[i] corresponds to tissue_labels[i], which corresponds to tissue_colors[i]
tissue_labels = ['Compacta',
                 'Spongiosa',
                 'WM',
                 'GM',
                 'Scalp',
                 'Blood',
                 'CSF']

conductivity_to_index = { \
  8.000e-06 : 0,
  2.500e-05 : 1,
  1.260e-04 : 2,
  2.750e-04 : 3,
  4.650e-04 : 4,
  6.000e-04 : 5,
  0.0016539999999999999 : 6
}

tissue_colors = np.array([[204.0, 255.0, 255.0, 1.0],
                          [255.0, 170.0, 0.0, 1.0],
                          [255.0, 255.0, 255.0, 1.0],
                          [155.0, 150.0, 147.0, 1.0],
                          [253.0, 245.0, 226.0, 1.0],
                          [255.0, 255.0, 0.0, 1.0],
                          [210.0, 4.0, 45.0, 1.0]])

tissue_colors[:, :3] /= 255.0

# assigne scalar values per pixel
print('Assigning scalar field')
for i in range(nr_positions):
  current_x_index = grid_indices[i][0]
  current_y_index = grid_indices[i][1]
  
  head_mask[current_y_index, current_x_index, :] = False
  sigma_image[current_y_index, current_x_index] = sigma_values[i]
  sigma_rgba[current_y_index, current_x_index, :] = tissue_colors[conductivity_to_index[sigma_values[i]]]

if add_boundary:
  print('Grafting boundaries on top of image')
  
  # extract boundary from image, and perform a dilation to increase visability
  boundaries = skimage.segmentation.find_boundaries(sigma_image, mode='thick')
  for i in range(nr_dilations):
    boundaries = skimage.morphology.binary_dilation(boundaries)
  
  # graft boundary onto images
  for i in range(nr_y_cells):
    for j in range(nr_x_cells):
      if boundaries[i, j]:
        head_mask[i, j, :] = False
        sigma_rgba[i, j, :] = np.array([0.0, 0.0, 0.0, 1.0])

print('Masking of positions outside the head')
masked_sigma = np.ma.masked_where(head_mask, sigma_rgba)

#######################################
# Create figure
#######################################

# configure latex
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 4
mpl.rcParams['ytick.labelsize'] = 4


####################################
## compute figure size to fit figure into latex column without scaling
## credit for this function goes to 
## https://jwalton.info/Embed-Publication-Matplotlib-Latex/
####################################
# params :
#			- width 							:				float, textwidth or columnwidth in pts.
#																		you can get this value by inserting \showthe\textwidth
#																		or \showthe\columnwidth into your latex document,
#																		depending on if your text is organized into columns or not.
#																		For IEEE Trans. on Medical Imaging we have a columnwidth of
#																		252 pt
#			- fraction						:				Fraction of the width which the figure is supposed to occupy
#			- subplot_structure		:				give the subplot structure, meaning a tuple (n, m) where
#																		n is the number of subplot rows and m is the number of
#																		subplot columns
#
# returns :
#			fig_dim : tuple containing the dimensions of the figure in inches
def figure_size(width, fraction = 1.0, subplot_structure = (1, 1)):
	fig_width_pt = width * fraction
	inches_per_latex_pt = 1.0/72.27
	golden_ratio = (np.sqrt(5) - 1)/2.0
	figure_width_in = inches_per_latex_pt * fig_width_pt
	figure_height_in = figure_width_in * golden_ratio * (subplot_structure[0] / subplot_structure[1])
	return (figure_width_in, figure_height_in)


# set up figure
columnwidth_pt = 370.0
figure = plt.figure(figsize = figure_size(columnwidth_pt), dpi=300)
grid_spec = matplotlib.gridspec.GridSpec(24, 24, figure=figure)

# add axes for plot of conductivity on slice, and plot image
axes_sigma = figure.add_subplot(grid_spec[2:18, :])
image_handle = axes_sigma.imshow(masked_sigma, extent = potential_image_extent, origin = 'lower', aspect = 'equal')

# set axis labels 
xaxis_x_pos = 0.92
xaxis_y_pos = 0.06
yaxis_x_pos = 0.075
yaxis_y_pos = 0.935
axes_sigma.set_xlabel(r'{\footnotesize mm}')
axes_sigma.set_ylabel(r'{\footnotesize mm}')
axes_sigma.xaxis.set_label_coords(xaxis_x_pos, xaxis_y_pos)
axes_sigma.yaxis.set_label_coords(yaxis_x_pos, yaxis_y_pos)

# set title
axes_sigma.set_title(r'\textbf{\small (b) Axial slice of mesh}', pad = 10.3)

# add colorbar
nr_tissues = len(conductivities)
tissue_colormap = matplotlib.colors.ListedColormap(tissue_colors)
boundary_norm = matplotlib.colors.BoundaryNorm([i - 0.5 for i in range(nr_tissues + 1)], ncolors=nr_tissues)
tissue_color_scalar_mapper = matplotlib.cm.ScalarMappable(norm=boundary_norm, cmap = tissue_colormap)
colorbar_axes = figure.add_subplot(grid_spec[-1, 4:20])
colorbar_handle = figure.colorbar(tissue_color_scalar_mapper, orientation = 'horizontal', ax=axes_sigma, format = "{x:.0E}", cax = colorbar_axes)
colorbar_handle.ax.set_xticks(np.array([i for i in range(nr_tissues)]))
colorbar_handle.ax.set_xticklabels(tissue_labels)
colorbar_handle.ax.tick_params(size = 0)
colorbar_handle.ax.set_title(r'{\small Tissue}')

# save figure
plt.savefig('realistic_mesh_visualization.pdf', format = 'pdf', bbox_inches = 'tight', pad_inches = 0)

# show figure
plt.show()
