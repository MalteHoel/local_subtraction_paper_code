The folder containing this README file should contain the following files:

  - visualize_potential_on_slice.py
  - visualize_head_model.py

These scripts can be used to reproduce the results for realistic meshes shown in the local subtraction paper. We describe what needs to be done to run the scripts.

1) Build DUNEuro
  - For this, we refer to the "installation.txt" file in the "duneuro_installation" folder.

2) Adapt the necessary paths
  - In "visualize_potential_on_slice.py" the paths to the volume conductor and the path to the duneuro-py library have to be added.
  - The script "visualize_head_model.py" assumes that the script "visualize_potential_on_slice.py" has already been run.
    The script "visualize_potential_on_slice.py" produces a npz archive as output. The path to this archive has to be added to the script "visualize_head_model.py"

3) Run the scripts
  - For "visualize_potential_on_slice.py", you have to specify whether you want to investigate the analytical or the local subtraction approach by setting the "potential_appraoch" flag. We initially set this flag to the analytical subtraction approach.
