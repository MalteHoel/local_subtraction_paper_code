#############################################
# This script computes the forward solution for a dipole using either the local subtraction or
# the analytical subtraction approach. This result is then sampled on a slice, and the result
# of this sampling is then visualized. Figures 5.4 and 5.5 in the paper were generated this way.
#############################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import matplotlib.cm
import matplotlib.gridspec
import math
import skimage
import sys

#######################################
# Simulation parameters
#######################################

# Please specify the path to the volume conductor below. The volume conductor is assumed to be given as an npz archive.
# Note that this script expects the volume conductor to be given in RAS coordinates (Right, Anterior, Superior)
volume_conductor_path = 'TO_FILL_IN'

# Please specify the path to the DUNEuro installation below
duneuropy_path='TO_FILL_IN'

# Definition of the dipole used for forward simulation and visualization.
# The dipole below is a tangential dipole on the anterior wall of the post-central gyrus.
# We interprete the dipole moment as having units of nAm. If the mesh coordinates are given in mm, and the conductivities
# are given in Siemens/mm, the units of the potential will be uV.
dipole_position = np.array([-38.42198332, 
                            -1.76773663,
                             58.75119099])
dipole_direction = np.array([0.50523976,
                             0.80518911, 
                             0.31048878])
dipole_strength = 20.0
dipole_moment = dipole_strength * dipole_direction

# Please specify the potential approach to be used for the numerical simulations. In this script, we assume
# the potential approach to be either 'local_subtraction' or 'analytical_subtraction'
potential_approach = 'analytical_subtraction'

#######################################
# Visualization parameters
#######################################

# After computing the values of the potentials on a slice through the mesh, the results are exported to an npz archive.
# On subsequent runs, e.g. if one wants to change parameters in the visualzation, these results can be re-used, which
# saves computation time. If the flag below is set to "True", the script will try to open the results from a previous run.
only_visualize = False

# To visualize how the tissue interfaces impact the head potential, we add the option to estimate the compartment boundaries
# and overlay them onto the potential plot. If the flag below is set to "True", this estimation and overlaying is performed.
add_boundary = True

# z-Coordinate of the slice used for visualization.
slice_height = 60.0

# Resolution used while sampling the slice
slice_sampling_resolution = 0.1

# Depending on the sampling resolution, the boundary will be too thin to be seen in the visualization. To make the boundary visible
# in the image, we perform a number of binary dilations. The parameter below specifies the number of dilations performed.
# If you change the sampling resolution, you might have to adapt this parameter.
nr_dilations = 2

# The parameters below control the limits used for colormapping the scalar potential values to colors.
vmin_potential = -5.0
vmax_potential = 5.0

#######################################
# Compute or load forward solution
#######################################

if not ((potential_approach == 'local_subtraction') or (potential_approach == 'analytical_subtraction')): 
  raise NotImplementedError(f'Potential approach {potential_approach} not supported, please use either "local_subtraction" or "analytical_subtraction".')

# create meeg driver
if not only_visualize:
  print('Performing forward simulation')
  
  sys.path.append(duneuropy_path)
  import duneuropy as dp

  volume_conductor_data = np.load(volume_conductor_path)
  nodes = volume_conductor_data['nodes']
  elements = volume_conductor_data['elements']
  labels = volume_conductor_data['labels']
  conductivities = volume_conductor_data['conductivities']

  # create driver
  print('Creating driver')
  mesh_cfg = {'nodes' : nodes, 'elements' : elements}
  tensor_cfg = {'labels' : labels, 'conductivities' : conductivities}
  volume_conductor_cfg = {'grid' : mesh_cfg, 'tensors' : tensor_cfg}
  # In the following, we have set 'post_process' to false. With this setting, DUNEuro will only compute the correction potential u^c, and not u^c + u^\infty
  driver_cfg = {'type' : 'fitted', 'solver_type' : 'cg', 'element_type' : 'tetrahedron', 'grainSize' : '100', 'post_process' : 'false', 'post_process_meg' : 'false', 'subtract_mean' : 'true'}
  solver_cfg = {'reduction' : '1e-16', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly'}
  driver_cfg['solver'] = solver_cfg
  driver_cfg['volume_conductor'] = volume_conductor_cfg

  meeg_driver = dp.MEEGDriver3d(driver_cfg)
  print('Driver created')

  # set potential approach
  potential_approach_config_analytical_subtraction = \
  {
    'type' : 'subtraction',
    'intorderadd' : 0,
    'intorderadd_lb' : 0
  }
  potential_approach_config_local_subtraction = \
  {
    'type' : 'localized_subtraction',
    'restrict' : False,
    'initialization' : 'single_element',
    'intorderadd_eeg_patch' : 0,
    'intorderadd_eeg_boundary' : 0,
    'intorderadd_eeg_transition' : 0,
    'extensions' : 'vertex vertex'
  }
  potential_approach_config_database = \
  {
    'analytical_subtraction' : potential_approach_config_analytical_subtraction,
    'local_subtraction' : potential_approach_config_local_subtraction
  }

  driver_cfg['source_model'] = potential_approach_config_database[potential_approach]

  # solve for the correction potential
  dipole = dp.Dipole3d(dipole_position, dipole_moment)

  print('Solving forward problem')
  correction_potential_storage = meeg_driver.makeDomainFunction()
  meeg_driver.solveEEGForward(dipole, correction_potential_storage, driver_cfg)
  print('Forward problem solved')

  #############################
  # prepare slice
  #############################
  
  # Place positions at the specified height, with the specified resolution.
  # The function below returns all positions on this slice that lie inside the FEM mesh.
  print('Placing positions for sampling on slice')
  placed_positions = meeg_driver.placePositionsZ(slice_sampling_resolution, slice_height)
  print('Positions placed')

  position_coordinates_raw = placed_positions['positions']
  nr_positions = len(position_coordinates_raw)
  grid_indices_raw = placed_positions['grid_indices']

  position_coordinates = np.empty((nr_positions, 3), dtype=np.float64)
  grid_indices = np.empty((nr_positions, 2), dtype=np.int32)
  for i in range(nr_positions):
    position_coordinates[i] = np.array(position_coordinates_raw[i])
    grid_indices[i] = np.array(grid_indices_raw[i])

  print('Evaluating correction potential on slice')
  correction_potential_values = np.array(meeg_driver.evaluateFunctionAtPositionsInsideMesh(correction_potential_storage, position_coordinates_raw))
  print('Correction potential on slice evaluated')

  print('Evaluating infinity potential on slice')
  infinity_potential_values = np.array(meeg_driver.evaluateUInfinityAtPositions(dipole, position_coordinates_raw))
  print('Infinity potential on slice evaluated')
  
  print('Evaluation chi on slice')
  if potential_approach == 'local_subtraction':
    chi_values = np.array(meeg_driver.evaluateChiAtPositions(dipole, position_coordinates_raw, driver_cfg['source_model'], driver_cfg['solver']))
  elif potential_approach == 'analytical_subtraction':
    chi_values = np.ones(infinity_potential_values.shape, dtype=np.float64)
  else:
    raise NotImplementedError(f'Potential approach {potential_approach} not supported, please use either "local_subtraction" or "analytical_subtraction".')
  print('Chi evaluated')
  
  print('Evaluating sigma on slice')
  sigma_values = np.array(meeg_driver.evaluateSigmaAtPositions(position_coordinates_raw))
  print('Sigma evaluated')
  
  grid_delta = np.array(placed_positions['grid_delta'])
  lower_left_bbox = np.array(placed_positions['lower_left_corner'])
  upper_right_bbox = np.array(placed_positions['upper_right_corner'])
  
  print('Forward simulation finished')
  
  print('Saving results')
  np.savez('potentials_on_slice.npz', position_coordinates=position_coordinates, grid_indices=grid_indices, grid_delta=grid_delta, lower_left_bbox=lower_left_bbox, upper_right_bbox=upper_right_bbox, correction_potential_values=correction_potential_values, infinity_potential_values=infinity_potential_values, chi_values=chi_values, sigma_values=sigma_values)
  print('Results saved')
else:
  print('Using precomputed values')
  print('Loading data')
  slice_data = np.load('potentials_on_slice.npz')
  position_coordinates = slice_data['position_coordinates']
  nr_positions = len(position_coordinates)
  grid_indices = slice_data['grid_indices']
  grid_delta = slice_data['grid_delta']
  lower_left_bbox = slice_data['lower_left_bbox']
  upper_right_bbox = slice_data['upper_right_bbox']
  correction_potential_values = slice_data['correction_potential_values']
  infinity_potential_values = slice_data['infinity_potential_values']
  chi_values = slice_data['chi_values']
  sigma_values = slice_data['sigma_values']
  print('Data loaded')

#######################################
# Create images from data
#######################################

print('Preparing data for visualization')
# compute image extent
x_extent = upper_right_bbox[0] - lower_left_bbox[0]
y_extent = upper_right_bbox[1] - lower_left_bbox[1]

nr_x_cells = math.floor(x_extent / grid_delta[0]) + 1
nr_y_cells = math.floor(y_extent / grid_delta[1]) + 1

lower_left_corner = lower_left_bbox[:2] - 0.5 * grid_delta
upper_right_corner = lower_left_corner[:2] + np.array([nr_x_cells * grid_delta[0], nr_y_cells * grid_delta[1]])
potential_image_extent = [lower_left_corner[0], upper_right_corner[0], lower_left_corner[1], upper_right_corner[1]]

# create storage for images
head_mask = np.full((nr_y_cells, nr_x_cells, 4), True)  # "True" is supposed to signal positions outside of the head.

correction_potential_image = np.empty((nr_y_cells, nr_x_cells), dtype=np.float64)
correction_potential_image[:] = np.nan

infinity_potential_image = np.empty((nr_y_cells, nr_x_cells), dtype=np.float64)
infinity_potential_image[:] = np.nan

total_potential_image = np.empty((nr_y_cells, nr_x_cells), dtype=np.float64)
total_potential_image[:] = np.nan

sigma_image = np.empty((nr_y_cells, nr_x_cells), dtype=np.float64)
sigma_image[:] = 0.0

# set up colormap
potential_normalizer = matplotlib.colors.Normalize(vmin = vmin_potential, vmax = vmax_potential, clip = False)
potential_colormap = plt.colormaps['coolwarm'].copy()
potential_colormap.set_bad(color = 'white')
potential_rgba_mapper = matplotlib.cm.ScalarMappable(norm=potential_normalizer, cmap = potential_colormap)

# assigne scalar values per pixel
print('Assigning scalar fields')
for i in range(nr_positions):
  current_x_index = grid_indices[i][0]
  current_y_index = grid_indices[i][1]
  
  head_mask[current_y_index, current_x_index, :] = False
  correction_potential_image[current_y_index, current_x_index] = correction_potential_values[i]
  infinity_potential_image[current_y_index, current_x_index] = chi_values[i] * infinity_potential_values[i]
  total_potential_image[current_y_index, current_x_index] = correction_potential_values[i] + chi_values[i] * infinity_potential_values[i]
  sigma_image[current_y_index, current_x_index] = sigma_values[i]
  

# transform to RGB images
print('Mapping to RGBA values')
correction_potential_rgba = potential_rgba_mapper.to_rgba(correction_potential_image)
infinity_potential_rgba = potential_rgba_mapper.to_rgba(infinity_potential_image)
total_potential_rgba = potential_rgba_mapper.to_rgba(total_potential_image)

if add_boundary:
  print('Grafting boundaries on top of images')
  
  # extract boundary from image, and perform a dilation to increase visability
  boundaries = skimage.segmentation.find_boundaries(sigma_image, mode='thick')
  for i in range(nr_dilations):
    boundaries = skimage.morphology.binary_dilation(boundaries)
  
  # graft boundary onto images
  for i in range(nr_y_cells):
    for j in range(nr_x_cells):
      if boundaries[i, j]:
        head_mask[i, j, :] = False
        correction_potential_rgba[i, j, :] = np.array([0.0, 0.0, 0.0, 1.0])
        infinity_potential_rgba[i, j, :] = np.array([0.0, 0.0, 0.0, 1.0])
        total_potential_rgba[i, j, :] = np.array([0.0, 0.0, 0.0, 1.0])

print('Masking of positions outside the head')
masked_correction_potential = np.ma.masked_where(head_mask, correction_potential_rgba)
masked_infinity_potential = np.ma.masked_where(head_mask, infinity_potential_rgba)
masked_total_potential = np.ma.masked_where(head_mask, total_potential_rgba)

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
grid_spec = matplotlib.gridspec.GridSpec(24, 24, figure = figure)

# add axes
axes_correction = figure.add_subplot(grid_spec[:20, :8])
axes_infinity = figure.add_subplot(grid_spec[:20, 8:16])
axes_total_potential = figure.add_subplot(grid_spec[:20, 16:])

# plot images onto axes
correction_plot = axes_correction.imshow(masked_correction_potential, extent = potential_image_extent, origin = 'lower', aspect = 'equal')
infinity_plot = axes_infinity.imshow(masked_infinity_potential, extent = potential_image_extent, origin = 'lower', aspect = 'equal')
total_plot = axes_total_potential.imshow(masked_total_potential, extent = potential_image_extent, origin = 'lower', aspect = 'equal')

# set axis labels
xaxis_x_pos = 0.92
xaxis_y_pos = 0.06

yaxis_x_pos = 0.075
yaxis_y_pos = 0.935

axes_correction.set_xlabel(r'{\footnotesize mm}')
axes_correction.set_ylabel(r'{\footnotesize mm}')
axes_correction.xaxis.set_label_coords(xaxis_x_pos, xaxis_y_pos)
axes_correction.yaxis.set_label_coords(yaxis_x_pos, yaxis_y_pos)
correction_xticks = axes_correction.get_xticks()
correction_xticklabels = axes_correction.get_xticklabels()
axes_correction.get_xaxis().set_ticks(correction_xticks[1:-1])
axes_correction.get_xaxis().set_ticklabels(correction_xticklabels[1:-1])

axes_infinity.set_xlabel(r'{\footnotesize mm}')
axes_infinity.set_ylabel(r'{\footnotesize mm}')
axes_infinity.xaxis.set_label_coords(xaxis_x_pos, xaxis_y_pos)
axes_infinity.yaxis.set_label_coords(yaxis_x_pos, yaxis_y_pos)
axes_infinity.get_yaxis().set_ticks([])
axes_infinity.get_yaxis().set_ticklabels([])
infinity_xticks = axes_infinity.get_xticks()
infinity_xticklabels = axes_infinity.get_xticklabels()
axes_infinity.get_xaxis().set_ticks(infinity_xticks[1:-1])
axes_infinity.get_xaxis().set_ticklabels(infinity_xticklabels[1:-1])

axes_total_potential.set_xlabel(r'{\footnotesize mm}')
axes_total_potential.set_ylabel(r'{\footnotesize mm}')
axes_total_potential.xaxis.set_label_coords(xaxis_x_pos, xaxis_y_pos)
axes_total_potential.yaxis.set_label_coords(yaxis_x_pos, yaxis_y_pos)
axes_total_potential.get_yaxis().set_ticks([])
axes_total_potential.get_yaxis().set_ticklabels([])
total_potential_xticks = axes_total_potential.get_xticks()
total_potential_xticklabels = axes_total_potential.get_xticklabels()
axes_total_potential.get_xaxis().set_ticks(total_potential_xticks[1:-1])
axes_total_potential.get_xaxis().set_ticklabels(total_potential_xticklabels[1:-1])

# add titles to figure
if potential_approach == 'local_subtraction':
  figure.suptitle(r'\textbf{Potentials involved in the local subtraction approach}')
  axes_correction.set_title(r'$u^c$')
  axes_infinity.set_title(r'$\chi \cdot u^\infty$')
  axes_total_potential.set_title(r'$u^c + \chi \cdot u^\infty$')
elif potential_approach == 'analytical_subtraction':
  figure.suptitle(r'\textbf{Potentials involved in the analytical subtraction approach}')
  axes_correction.set_title(r'$u^c$')
  axes_infinity.set_title(r'$u^\infty$')
  axes_total_potential.set_title(r'$u^c + u^\infty$')
else:
  raise NotImplementedError(f'Potential approach {potential_approach} not supported, please use either "local_subtraction" or "analytical_subtraction".')

# add colorbar to figure
colorbar_ticks = np.linspace(vmin_potential, vmax_potential, num = 5)
colorbar_axes = figure.add_subplot(grid_spec[-1, 6:18])
figure.colorbar(mappable=potential_rgba_mapper, cax=colorbar_axes, orientation = 'horizontal', fraction = 1.0, ticks = colorbar_ticks, format = "{x:.2f}")
colorbar_axes.set_title(r'{\small Potential ($\mu V$)}')
colorbar_ticklabels = colorbar_axes.get_xticklabels()
colorbar_ticklabels[0] = f'$\\leq {vmin_potential:.2f}$'
colorbar_ticklabels[-1] = f'$\\geq {vmax_potential:.2f}$'
colorbar_axes.get_xaxis().set_ticklabels(colorbar_ticklabels)

# save figure
plt.savefig('potentials_on_slice.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)

# show figure
plt.show()
