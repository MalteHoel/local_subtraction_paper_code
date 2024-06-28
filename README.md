This repository contains the scripts used in the numerical evaluation of the Local Subtraction Approach for EEG and MEG forward modeling. 
This approach is introduced and investigated in the accompanying paper (Preprint available at: https://arxiv.org/abs/2302.12785).

More concretely, the repository contains three top-level folders, which are described in the following.

1) duneuro_installation
   
  The Local Subtraction Approach has been implemented into the DUNEuro toolbox, an open-source C++ toolbox for neuroscience applications based on the DUNE framework.
  (See Schrader et. al., "DUNEuro—A software toolbox for forward modeling in bioelectromagnetism", 2021, DOI: 10.1371/journal.pone.0252431, for an introduction to DUNEuro.)
  To run the validation scripts, you need an installation of DUNEuro. To this end, the duneuro_installation folder contains a clone script that selects compatible versions of the necessary DUNE- and DUNEuro modules. We refer to the file duneuro_installation/installation.txt for a detailed description of the installation process.

2) multilayer_sphere_validation_study

  This folder contains the script to perform the numerical validation in multilayer sphere models, a small example illustrating the expected form of the input data, and a config file. We refer to the README.txt file in the mulitlayer_sphere_validation_study folder for a detailed description of how to run the validation script.
  
3) realistic_volume_conductor_investigation

  This folder contains the scripts to reproduce the results for the realistic mesh shown in the paper. We refer to the README.txt file in the realistic_volume_conductor_investigation folder for a detailed description of how to run the scripts.
