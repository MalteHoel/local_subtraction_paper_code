This repository contains the scripts used in the numerical evaluation of the Local Subtraction Approach for EEG and MEG forward modeling. 
This approach is introduced and investigated in the accompanying paper (Preprint available at: https://arxiv.org/abs/2302.12785).

More concretely, the repository contains two top-level folders, which are described in the following.

1) duneuro_installation
   
  The Local Subtraction Approach has been implemented into the DUNEuro toolbox, an open-source C++ toolbox for neuroscience applications based on the DUNE framework.
  (See Schrader et. al., "DUNEuro—A software toolbox for forward modeling in bioelectromagnetism", 2021, DOI: 10.1371/journal.pone.0252431, for an introduction to DUNEuro.)
  To run the validation scripts, you need an installation of DUNEuro. To this end, the duneuro_installation folder contains a clone script that selects compatible versions of the necessary DUNE- and DUNEuro modules. We refer to the file duneuro_installation/installation.txt for a detailed description of the installation process.

2) validation_study

  This folder contains the script to perform the numerical validation, a small example illustrating the expected form of the input data, and a config file. We refer to the README.txt file in the validation_study folder for a detailed description of how to run the validation script.
