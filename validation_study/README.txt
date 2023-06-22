The folder containing this README file should contain the following files and folders:
  - validate_source_models.py
  - configs.ini
  - input_data_example
  - README.txt
In the following, we describe what needs to be done to run the validation script.

1) Build DUNEuro
  - For this we refer to the "installation.txt" file in the "duneuro_installation" folder.

2) Prepare the input data
  - The script assumes that all input data is stored inside a folder, which in the following we call INPUT_FOLDER. We assume this folder to contain the following files in the following subdirectories:
  
    INPUT_FOLDER/volume_conductor/mesh.msh
    INPUT_FOLDER/volume_conductor/conductivities.txt
    
    INPUT_FOLDER/sensors/electrodes.txt                   [EEG ONLY]
    INPUT_FOLDER/sensors/coils.txt                        [MEG ONLY]
    INPUT_FOLDER/sensors/projections.txt                  [MEG ONLY]
    
    INPUT_FOLDER/dipoles/radial/...
    INPUT_FOLDER/dipoles/tangential/...
    
  Here, ... stands for files named "ecc_0.x_[radial|tangential].txt", where x is some number. The "input_data_example" directory illustrates the described format.

3) Edit the config file
  - Open the "configs.ini" file. In this file you should add the paths in the "libraries" section and the "io" section. E.g. if BUILD_FOLDER is the folder containing your DUNEuro build, you should set
  
    duneuro=BUILD_FOLDER/duneuro-py/src
    duneuro_analytic_solution=BUILD_FOLDER/duneuro-analytic-solution/src
  
    and similarly for the path to the simbiosphere build.

Now you can run the "validate_source_models.py" script. After this script finished, the designated output folder contains the results of the numerical simulation.
