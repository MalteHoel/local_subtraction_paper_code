#!/bin/python3

############################################################################################################
#
# Script for the validation of source models in multilayer sphere models. This script uses duneuro to 
# compute numerical solutions for the EEG or MEG forward problem and compares these results against the analytic solutions. 
# This comparison is performed for tangential and radial dipoles at different eccentricities.
#
#############################################################################################################

# import libraries
import numpy as np
import os
import configparser
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

#########################################
# read configs
#########################################
configs = configparser.ConfigParser()
configs.optionxform = str
configs.read('configs.ini')

# read library paths
duneuro_path = configs.get('libraries', 'duneuro')
duneuro_analytic_solution = configs.get('libraries', 'duneuro_analytic_solution')
simbiosphere = configs.get('libraries', 'simbiosphere')

#read io information
input_folder = configs.get('io', 'input_folder')
output_foldfer = configs.get('io', 'output_folder')

#read sphere parameters
radii = [float(x) for x in configs.get('multilayer_sphere', 'radii').split()]
sphere_center = [float(x) for x in configs.get('multilayer_sphere', 'center').split()]
conductivities = [float(x) for x in configs.get('multilayer_sphere', 'conductivities').split()]
eccentricities = np.array([float(x) for x in configs.get('multilayer_sphere', 'eccentricities').split()])

# import duneuro and simbiosphere
sys.path.append(duneuro_path)
sys.path.append(duneuro_analytic_solution)
sys.path.append(simbiosphere)
import duneuropy as dp
import duneuroAnalyticSolutionPy as dp_analytic
import simbiopy as sp

# set io information
filename_grid = os.path.join(input_folder, 'volume_conductor', 'mesh.msh')
filename_tensors = os.path.join(input_folder, 'volume_conductor', 'conductivities.txt')
volume_conductor = {'grid.filename' : filename_grid, 'tensors.filename' : filename_tensors, 'refine_brain' : False, 'refine_skin' : True}
folder_dipoles = os.path.join(input_folder, 'dipoles')

# read section into dict
# params :
#	- configs 		: ConfigParser object
#	- section_name 	: Name of the section to be read into dict
def read_section_into_dict(configs, section_name):
	dict_storage = {}
	for key, value in configs.items(section_name):
		dict_storage[key] = value
	return dict_storage

# read driver config
driver_config = read_section_into_dict(configs, 'driver_config')
driver_config['volume_conductor'] = volume_conductor
driver_config['solver'] = read_section_into_dict(configs, 'solver_config')

# read electrode config
electrode_config = read_section_into_dict(configs, 'electrode_config')
use_projected_electrodes = configs["testing_parameters"].getboolean('use_projected_electrodes')

# read meg config
driver_config['meg']  = read_section_into_dict(configs, 'meg_config')

# read source models
source_model_list = configs.get('source_models', 'types').split()
source_model_config_database = {}
for source_model in source_model_list:
	source_model_config_database[source_model] = read_section_into_dict(configs, f"{source_model}_config")
print("Source models to validate:")
for source_model in source_model_list:
	print(source_model)

source_models_for_numerical_simulation = []
for source_model in source_model_list:
	if configs[f"{source_model}_config"].getboolean('skip_numerical_simulation', False):
		print(f"Skipping numerical simulation for {source_model} source model")
	else:
		print(f"Performing numerical simulation for {source_model} source model")
		source_models_for_numerical_simulation.append(source_model)

numerical_simulation_needed = len(source_models_for_numerical_simulation) > 0

source_models_for_comparison = []
for source_model in source_model_list:
	if configs[f"{source_model}_config"].getboolean('skip_comparison', False):
		print(f"Skipping comparison of numerical and analytical solutions for {source_model} source model")
	else:
		print(f"Performing comparison of numerical and analytical solutions for {source_model} source model")
		source_models_for_comparison.append(source_model)

comparisons_needed = len(source_models_for_comparison) > 0

source_models_for_boxplot = []
for source_model in source_model_list:
	if configs[f"{source_model}_config"].getboolean('skip_boxplot', False):
		print(f"Skipping {source_model} source model for boxplot")
	else:
		print(f"Showing {source_model} in result boxplots")
		source_models_for_boxplot.append(source_model)

# read transfer matrix configs
transfer_configs = {}
transfer_configs['solver'] = read_section_into_dict(configs, 'transfer_config.solver')
force_transfer_recomputation = configs['transfer_config.solver'].getboolean('force_recomputation', False)

# read testing parameters
do_eeg = configs['testing_parameters'].getboolean('do_eeg')
do_meg = configs['testing_parameters'].getboolean('do_meg')

skip_analytical_eeg = configs['testing_parameters'].getboolean('skip_analytical_eeg')
skip_analytical_meg = configs['testing_parameters'].getboolean('skip_analytical_meg')

# read boxplot configs
eccentricity_sets = [selection.strip().split(' ') for selection in configs.get('boxplot_config', 'eccentricities_per_boxplot').split('|')]

boxplots_needed = len(eccentricity_sets) > 0 and len(source_models_for_boxplot) > 0

#########################################
# create output folder structure
#########################################

Path('output').mkdir(exist_ok=True)
Path('output/transfermatrices').mkdir(exist_ok=True)
Path('output/results').mkdir(exist_ok=True)
Path('output/results/eeg').mkdir(exist_ok=True)
Path('output/results/meg').mkdir(exist_ok=True)
Path('output/results/eeg/radial').mkdir(exist_ok=True)
Path('output/results/eeg/tangential').mkdir(exist_ok=True)

Path('output/results/eeg/radial/analytical_solution').mkdir(exist_ok=True)
Path('output/results/eeg/radial/boxplots').mkdir(exist_ok=True)
Path('output/results/eeg/radial/lnMAG').mkdir(exist_ok=True)
Path('output/results/eeg/radial/numerical_solution').mkdir(exist_ok=True)
Path('output/results/eeg/radial/rdm').mkdir(exist_ok=True)
Path('output/results/eeg/radial/relative_error').mkdir(exist_ok=True)
Path('output/results/eeg/radial/dataframes').mkdir(exist_ok=True)

Path('output/results/eeg/tangential/analytical_solution').mkdir(exist_ok=True)
Path('output/results/eeg/tangential/boxplots').mkdir(exist_ok=True)
Path('output/results/eeg/tangential/lnMAG').mkdir(exist_ok=True)
Path('output/results/eeg/tangential/numerical_solution').mkdir(exist_ok=True)
Path('output/results/eeg/tangential/rdm').mkdir(exist_ok=True)
Path('output/results/eeg/tangential/relative_error').mkdir(exist_ok=True)
Path('output/results/eeg/tangential/dataframes').mkdir(exist_ok=True)

Path('output/results/meg/analytical_solution').mkdir(exist_ok=True)
Path('output/results/meg/boxplots').mkdir(exist_ok=True)
Path('output/results/meg/lnMAG').mkdir(exist_ok=True)
Path('output/results/meg/numerical_solution').mkdir(exist_ok=True)
Path('output/results/meg/rdm').mkdir(exist_ok=True)
Path('output/results/meg/relative_error').mkdir(exist_ok=True)
Path('output/results/meg/dataframes').mkdir(exist_ok=True)

#########################################
# compute transfer matrices
#########################################

# create driver
if numerical_simulation_needed:
	meeg_driver = dp.MEEGDriver3d(driver_config)

# check if transfer matrices need to be (re)computed
eeg_dependencies_change_time = max(os.path.getctime('input/sensors/electrodes.txt'), os.path.getctime('input/volume_conductor/conductivities.txt'), os.path.getctime('input/volume_conductor/mesh.msh'))
compute_eeg_transfer = do_eeg and (not os.path.exists('output/transfermatrices/eeg_transfer_matrix.npy') or os.path.getctime('output/transfermatrices/eeg_transfer_matrix.npy') < eeg_dependencies_change_time)

meg_dependencies_change_time = max(os.path.getctime('input/sensors/coils.txt'), os.path.getctime('input/sensors/projections.txt'), os.path.getctime('input/volume_conductor/conductivities.txt'), os.path.getctime('input/volume_conductor/mesh.msh'))
compute_meg_transfer = do_meg and (not os.path.exists('output/transfermatrices/meg_transfer_matrix.npy') or os.path.getctime('output/transfermatrices/meg_transfer_matrix.npy') < meg_dependencies_change_time)

# if EEG simulations are going to be performed we need to set the electrode positions. Note that we need to do this even if we have already computed the transfer matrices, since we might need
# to postprocess the solution, which requires knowledge about the electrode positions
if do_eeg and (numerical_simulation_needed or not skip_analytical_eeg):
	print(f"Setting electrodes")
	if use_projected_electrodes:
		print("Using projected electrodes")
		electrodes = dp.read_field_vectors_3d('input/sensors/projected_electrodes.txt')
	else:
		print("Not using projected electrodes")
		electrodes = dp.read_field_vectors_3d('input/sensors/electrodes.txt')
	if numerical_simulation_needed:
		meeg_driver.setElectrodes(electrodes, electrode_config)
	print(f"Electrodes set")

# potentially compute EEG transfer matrix
if (compute_eeg_transfer or force_transfer_recomputation) and numerical_simulation_needed:
	print(f"Computing EEG transfer matrix")
	# compute transfer matrix
	start_time = time.time()
	transfer_matrix_raw, computation_information = meeg_driver.computeEEGTransferMatrix(transfer_configs)
	transfer_matrix = np.array(transfer_matrix_raw)
	print(f"Computing the EEG transfer matrix took {time.time() - start_time} seconds")
	print(f"EEG transfer matrix computed")
	print(f"Saving EEG transfer matrix")
	np.save('output/transfermatrices/eeg_transfer_matrix.npy', transfer_matrix)
	print(f"Transfer matrix saved")
elif do_eeg:
	print("EEG transfer matrix already computed or not needed")

# if MEG simulations are going to be performed we need to set the coil positions and their corresponding projections. Similar to the EEG case, we potentially need to postprocess the solution, which requires 
# knowledge about the sensor locations
if do_meg and (numerical_simulation_needed or not skip_analytical_meg):
	print(f"Setting coils and projections")
	coils = dp.read_field_vectors_3d('input/sensors/coils.txt')
	projections = dp.read_projections_3d('input/sensors/projections.txt')
	if numerical_simulation_needed:
		meeg_driver.setCoilsAndProjections(coils, projections)
	print(f"Coils and projections set")

# potentially compute MEG transfer matrix
if (compute_meg_transfer or force_transfer_recomputation) and numerical_simulation_needed:
	print(f"Computing MEG transfer matrix")
	start_time = time.time()
	transfer_matrix_raw, computation_information = meeg_driver.computeMEGTransferMatrix(transfer_configs)
	transfer_matrix = np.array(transfer_matrix_raw)
	print(f"Computing the MEG transfer matrix took {time.time() - start_time} seconds")
	print(f"Transfer matrix computed")
	print(f"Saving transfer matrix")
	np.save('output/transfermatrices/meg_transfer_matrix.npy', transfer_matrix)
	print(f"Transfer matrix saved")
elif do_meg:
	print("MEG transfer matrix already computed or not needed")


#########################################
# compute numerical solutions
#########################################


# compute numerical eeg solutions
if do_eeg and numerical_simulation_needed:
	print("Computing numerical EEG solutions")
	transfer_matrix = np.load('output/transfermatrices/eeg_transfer_matrix.npy')
	# radial dipoles
	print("Simulating radial dipoles")
	radial_dipole_filenames = os.listdir('input/dipoles/radial')
	for filename in radial_dipole_filenames:
		dipoles = dp.read_dipoles_3d(f"input/dipoles/radial/{filename}")
		for source_model in source_models_for_numerical_simulation:
			driver_config['source_model'] = source_model_config_database[source_model]
			start_time = time.time()
			numerical_solution, computation_information = meeg_driver.applyEEGTransfer(transfer_matrix, dipoles, driver_config)
			print(f"Computing forward solutions for {source_model} took {time.time() - start_time} seconds")
			np.savetxt(f"output/results/eeg/radial/numerical_solution/{source_model}_numerical_solution_{filename}", np.array(numerical_solution), fmt='%.18f', delimiter = ' ', newline='\n')
	# tangential dipoles
	print("Simulating tangential dipoles")
	tangential_dipole_filenames = os.listdir('input/dipoles/tangential')
	for filename in tangential_dipole_filenames:
		dipoles = dp.read_dipoles_3d(f"input/dipoles/tangential/{filename}")
		for source_model in source_models_for_numerical_simulation:
			driver_config['source_model'] = source_model_config_database[source_model]
			start_time = time.time()
			numerical_solution, computation_information = meeg_driver.applyEEGTransfer(transfer_matrix, dipoles, driver_config)
			print(f"Computing forward solutions for {source_model} took {time.time() - start_time} seconds")
			np.savetxt(f"output/results/eeg/tangential/numerical_solution/{source_model}_numerical_solution_{filename}", np.array(numerical_solution), fmt='%.18f', delimiter = ' ', newline='\n')
	print("Numerical EEG solutions computed")


# compute numerical meg solutions
if do_meg and numerical_simulation_needed:
	print("Computing numerical MEG solutions")
	transfer_matrix = np.load('output/transfermatrices/meg_transfer_matrix.npy')
	# tangential dipoles
	print("Simulating tangential dipoles")
	tangential_dipole_filenames = os.listdir('input/dipoles/tangential')
	for filename in tangential_dipole_filenames:
		dipoles = dp.read_dipoles_3d(f"input/dipoles/tangential/{filename}")
		for source_model in source_models_for_numerical_simulation:
			driver_config['source_model'] = source_model_config_database[source_model]
			driver_config['post_process'] = False
			start_time = time.time()
			numerical_solution, computation_information = meeg_driver.applyMEGTransfer(transfer_matrix, dipoles, driver_config)
			print(f"Computing forward solutions for {source_model} took {time.time() - start_time} seconds")
			np.savetxt(f"output/results/meg/numerical_solution/{source_model}_numerical_solution_{filename}", np.array(numerical_solution), fmt='%.18f', delimiter = ' ', newline='\n')
	print("Numerical MEG solutions computed")


#########################################
# compute analytical solutions
#########################################

# eeg
if do_eeg and not skip_analytical_eeg:
	print("Computing analytical EEG solutions")
	electrodes_simbio = [np.array(electrode).tolist() for electrode in electrodes]
	#radial dipoles
	print(f"Computing solutions for radial dipoles")
	radial_dipole_filenames = os.listdir('input/dipoles/radial')
	for filename in radial_dipole_filenames:
		dipoles = dp.read_dipoles_3d(f"input/dipoles/radial/{filename}")
		with open(f"output/results/eeg/radial/analytical_solution/analytical_solution_{filename}", mode='w') as outputfile:
			for count, dipole in enumerate(dipoles):
				analytical_solution = sp.analytic_solution(radii, sphere_center, conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
				mean = sum(analytical_solution) / float(len(analytical_solution))
				analytical_solution = [x - mean for x in analytical_solution]
				outputfile.write(" ".join([str(x) for x in analytical_solution]))
				if count < len(dipoles)  - 1:
					outputfile.write("\n")
	# tangential dipoles
	print(f"Computing solutions for tangential dipoles")
	tangential_dipole_filenames = os.listdir('input/dipoles/tangential')
	for filename in tangential_dipole_filenames:
		dipoles = dp.read_dipoles_3d(f"input/dipoles/tangential/{filename}")
		with open(f"output/results/eeg/tangential/analytical_solution/analytical_solution_{filename}", mode='w') as outputfile:
			for count, dipole in enumerate(dipoles):
				analytical_solution = sp.analytic_solution(radii, sphere_center, conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
				mean = sum(analytical_solution) / float(len(analytical_solution))
				analytical_solution = [x - mean for x in analytical_solution]
				outputfile.write(" ".join([str(x) for x in analytical_solution]))
				if count < len(dipoles)  - 1:
					outputfile.write("\n")
	print(f"Analytical solutions computed")
elif do_eeg and skip_analytical_eeg:
	print(f"Skipping computation of analytical EEG solution")


# meg
if do_meg and not skip_analytical_meg:
	print("Computing analytical MEG solutions")
	MEGSolver = dp_analytic.AnalyticSolutionMEG(dp.FieldVector3D(sphere_center))
	tangential_dipole_filenames = os.listdir('input/dipoles/tangential')
	for filename in tangential_dipole_filenames:
		dipoles = dp.read_dipoles_3d(f"input/dipoles/tangential/{filename}")
		with open(f"output/results/meg/analytical_solution/analytical_solution_{filename}", mode='w') as outputfile:
			for count, dipole in enumerate(dipoles):
				MEGSolver.bind(dipole)
				analytic_vals = []
				for coil_index, coil in enumerate(coils):
					for projection in projections[coil_index]:
						analytic_vals.append(str(MEGSolver.secondaryField(coil, projection)))
				outputfile.write(' '.join(analytic_vals))
				if count < len(dipoles) -1:
					outputfile.write("\n")
	print(f"Analytical solutions computed")
elif do_meg and skip_analytical_meg:
	print(f"Skipping computation of analytical MEG solution")


#########################################
# compare numerical and analytical solutions
#########################################

# first define error measures

# relative error
# params : 
#			- numerical_solution 	: 1-dimensional numpy array
#			- analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
def relative_error(numerical_solution, analytical_solution):
	assert len(numerical_solution) == len(analytical_solution)
	return np.linalg.norm(numerical_solution - analytical_solution) / np.linalg.norm(analytical_solution)

# rdm error
# params :
#			- numerical_solution 	: 1-dimensional numpy array
#			- analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
def rdm(numerical_solution, analytical_solution):
	assert len(numerical_solution) == len(analytical_solution)
	return np.linalg.norm(numerical_solution/np.linalg.norm(numerical_solution) - analytical_solution/np.linalg.norm(analytical_solution))

# lnMAG
# params :
#			- numerical_solution 	: 1-dimensional numpy array
#			- analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
def lnMAG(numerical_solution, analytical_solution):
	assert len(numerical_solution) == len(analytical_solution)
	return np.log(np.linalg.norm(numerical_solution)/np.linalg.norm(analytical_solution))

# compare analytical solutions and numerical solutions inside folder and write the results to files
#	params :
#		- source_model					: source model for which we want to compute comparisons
#		- basefolder						:	basefolder for the current testing modality, e.g. 'output/results/eeg/radial'
#		- orientation_tag				: 'radial' or 'tangential'
def create_comparisons(source_model, basefolder, orientation_tag):
	basename_files = os.listdir(f"input/dipoles/{orientation_tag}")
	local_dataframes = [None] * len(basename_files)
	current_offset = 0
	for count, basename in enumerate(basename_files):
		analytical_solutions = np.loadtxt(f"{basefolder}/analytical_solution/analytical_solution_{basename}")
		numerical_solutions = np.loadtxt(f"{basefolder}/numerical_solution/{source_model}_numerical_solution_{basename}")
		assert analytical_solutions.shape == numerical_solutions.shape
		relative_errors = [relative_error(numerical_solution, analytical_solution) for numerical_solution, analytical_solution in zip(numerical_solutions, analytical_solutions)]
		rdms = [rdm(numerical_solution, analytical_solution) for numerical_solution, analytical_solution in zip(numerical_solutions, analytical_solutions)]
		lnMAGs = [lnMAG(numerical_solution, analytical_solution) for numerical_solution, analytical_solution in zip(numerical_solutions, analytical_solutions)]
		
		# write error measures to file
		with open(f"{basefolder}/relative_error/{source_model}_relative_error_{basename}", mode='w') as outputfile:
			outputfile.write('\n'.join([str(val) for val in relative_errors]))
		with open(f"{basefolder}/rdm/{source_model}_rdm_{basename}", mode='w') as outputfile:
			outputfile.write('\n'.join([str(val) for val in rdms]))
		with open(f"{basefolder}/lnMAG/{source_model}_lnMAG_{basename}", mode='w') as outputfile:
			outputfile.write('\n'.join([str(val) for val in lnMAGs]))
		
		#extract eccentricity from file name. For this we assume the filename to contain the substring "ecc_0.x_", where x is some number, e.g. 1 or 95
		base_index = basename.find('ecc_0.')
		base_offset = len('ecc_')
		end_offset = basename[base_index+base_offset:].find('_')
		eccentricity = basename[base_index+base_offset:base_index+base_offset+end_offset]
		
		local_dataframes[count] = pd.DataFrame({'source_model' : source_model, 'ecc' : eccentricity, 'relative_error' : relative_errors, 'rdm' : rdms, 'lnMAGS' : lnMAGs}, index=range(current_offset, current_offset + len(analytical_solutions)))
		current_offset += len(analytical_solutions)
		
	total_dataframe = pd.concat(local_dataframes)
	total_dataframe.to_csv(f"{basefolder}/dataframes/{source_model}_dataframe.csv")
	
	return

#														source_model					ecc					relative_error					rdm					lnMAG

# we now iterate over all source models and compare their respective numerical solutions with the analytical solution
if comparisons_needed:
	print("Comparing numerical and analytical solutions")
	for source_model in source_models_for_comparison:
		print(f"Performing comparisons for {source_model} source model")
		# eeg
		if do_eeg:
			print("Comparing EEG solutions")
			
			# radial
			print("Comparing radial solutions")
			create_comparisons(source_model, 'output/results/eeg/radial', 'radial')
					
			# tangential
			print("Comparing tangential solutions")
			create_comparisons(source_model, 'output/results/eeg/tangential', 'tangential')
			
		# meg
		if do_meg:
			print("Comparing MEG solutions")
			
			#tangential
			print("Comparing tangential solutions")
			create_comparisons(source_model, 'output/results/meg', 'tangential')
		
		print(f"Comparisons for {source_model} source model finished")
	print("All comparisons finished")
else:
	print("No comparisons need to be computed")

#########################################
# create boxplots
#########################################

# create pandas dataframe for later creation of boxplots via seaborn. For the structure of the constructed dataframe look at the parameter description of the function "create_boxplots"
# params	:
#		- eccentricity_selection			: list of strings, where each string represents an eccentricity to include in the boxplot, e.g. ['0.7', '0.8', '0.9']
#		- source_model_selection			: list of strings, where each string represents a source model to include in the boxplot, e.g. ['venant', 'localized_subtraction']
#		- basefolder									:	basefolder for the current testing modality, e.g. 'output/results/eeg/radial'
#		- orientation_tag							: 'radial' or 'tangential'
def create_dataframe(eccentricity_selection, source_model_selection, basefolder, orientation_tag):
	number_of_eccentricities = len(eccentricity_selection)
	
	# we first create a dataframe for every source model. These will be concatenated later on
	number_of_source_models = len(source_models_for_boxplot)
	source_model_data_frames = [None] * number_of_source_models
	
	for i, source_model in enumerate(source_models_for_boxplot):
		local_data_frames = [None] * number_of_eccentricities
		for j, eccentricity in enumerate(eccentricity_selection):
			relative_errors = np.loadtxt(f"{basefolder}/relative_error/{source_model}_relative_error_ecc_{eccentricity}_{orientation_tag}.txt")
			rdms = np.loadtxt(f"{basefolder}/rdm/{source_model}_rdm_ecc_{eccentricity}_{orientation_tag}.txt")
			lnMAGs = np.loadtxt(f"{basefolder}/lnMAG/{source_model}_lnMAG_ecc_{eccentricity}_{orientation_tag}.txt")
			local_data_frames[j] = pd.DataFrame({'source_model' : source_model, 'ecc' : eccentricity, 'relative_error' : relative_errors, 'rdm' : rdms, 'lnMAG' : lnMAGs})
		source_model_data_frames[i] = pd.concat(local_data_frames)
		
	total_dataframe = pd.concat(source_model_data_frames)
	
	return total_dataframe


# create boxplots for relative_error, rdm and lnMAG for different source models
# params :
#			- dataframe 			: pandas dataframe containing the information to be plotted inside the boxplot. The structure of this dataframe is supposed to be
#														
#														source_model					ecc					relative_error					rdm					lnMAG
#
#												0		partial_integration		0.1					0.04										0.03				-0.001
#																																	.
#																																	.
#																																	.
#			- basefolder			: basefolder for the current testing modality, e.g. 'output/results/eeg/radial'
#			- modality_tag		: 'eeg' or 'meg'
#			- orientation_tag : 'radial' or 'tangential'	 
def create_boxplots(dataframe, basefolder, modality_tag, orientation_tag):
		# relative error
	fig_relative_error, ax_relative_error = plt.subplots()
	ax_relative_error.set_ylim(0, 0.1)
	sns.boxplot(x='ecc', y='relative_error', hue='source_model', data=dataframe, ax=ax_relative_error)
	plt.savefig(f"{basefolder}/boxplots/relative_error_{modality_tag}_{orientation_tag}_{'_'.join(source_models_for_boxplot)}_{'_'.join(eccentricity_selection)}.png")
	plt.clf()
	
	# rdm
	fig_rdm, ax_rdm = plt.subplots()
	ax_rdm.set_ylim(0, 0.1)
	sns.boxplot(x='ecc', y='rdm', hue='source_model', data=dataframe, ax=ax_rdm)
	plt.savefig(f"{basefolder}/boxplots/rdm_{modality_tag}_{orientation_tag}_{'_'.join(source_models_for_boxplot)}_ecc_{'_'.join(eccentricity_selection)}.png")
	plt.clf()
	
	# lnMAG
	fig_lnMAG, ax_lnMAG = plt.subplots()
	ax_lnMAG.set_ylim(-0.1, 0.1)
	sns.boxplot(x='ecc', y='lnMAG', hue='source_model', data=dataframe, ax=ax_lnMAG)
	plt.savefig(f"{basefolder}/boxplots/lnMAG_{modality_tag}_{orientation_tag}_{'_'.join(source_models_for_boxplot)}_ecc_{'_'.join(eccentricity_selection)}.png")
	plt.clf()
	
	return

if boxplots_needed:
	print("Creating boxplots")
	# eeg
	if do_eeg:
		for eccentricity_selection in eccentricity_sets:
			# radial
			dataframe_eeg_radial = create_dataframe(eccentricity_selection, source_models_for_boxplot, 'output/results/eeg/radial', 'radial')
			create_boxplots(dataframe_eeg_radial, 'output/results/eeg/radial', 'eeg', 'radial')
			
			# tangential
			dataframe_eeg_tangential = create_dataframe(eccentricity_selection, source_models_for_boxplot, 'output/results/eeg/tangential', 'tangential')
			create_boxplots(dataframe_eeg_tangential, 'output/results/eeg/tangential', 'eeg', 'tangential')

	# meg
	if do_meg:
		for eccentricity_selection in eccentricity_sets:
			#tangential
			dataframe_meg_tangential = create_dataframe(eccentricity_selection, source_models_for_boxplot, 'output/results/meg', 'tangential')
			create_boxplots(dataframe_meg_tangential, 'output/results/meg', 'meg', 'tangential')
	print("Boxplots created")
else:
	print("No boxplots need to be created")

print("The program didn't crash!")
