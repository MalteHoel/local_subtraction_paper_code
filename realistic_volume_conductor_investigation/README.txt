The folder containing this README file should contain the following files:

  - visualize_potential_on_slice.py
  - visualize_head_model.py

These scripts can be used to reproduce the results for realistic meshes shown in the local subtraction paper. We describe what needs to be done to run the scripts.

1) Build DUNEuro
  - For this, we refer to the "installation.txt" file in the "duneuro_installation" folder.

2) Adapt the necessary paths and flags.
  - In "visualize_potential_on_slice.py" the paths to the volume conductor and the path to the duneuro-py library have to be added. Additionally, the flag "os_is_ubuntu_24" needs to be set to either True or False.
  - The script "visualize_head_model.py" assumes that the script "visualize_potential_on_slice.py" has already been run.
    The script "visualize_potential_on_slice.py" produces a npz archive as output. The path to this archive has to be added to the script "visualize_head_model.py"

3) Install dependencies

  On Ubuntu 24.04, you can do
  sudo apt install python3-numpy
  sudo apt install python3-skimage
  sudo apt install python3-matplotlib
  
  On older Ubuntu distributions the version of skimage in the package repository is
  not recent enough. In this case, you can use pip to install a more recent 
  version of scikit-image that fits to your environment. The plots in the paper were
  created using version 0.22.0 of scikit-image.
  
  Furthermore, by default the images are created from Matplotlib using LaTeX. If you 
  want to exactly reproduce the figures from the paper, please install the following.
  
  sudo apt install texlive
  sudo apt install texlive-latex-extra
  sudo apt install cm-super
  sudo apt install dvipng

4) Run the scripts
  - For "visualize_potential_on_slice.py", you have to specify whether you want to investigate the analytical or the local subtraction approach by setting the "potential_appraoch" flag. We initially set this flag to the analytical subtraction approach.
