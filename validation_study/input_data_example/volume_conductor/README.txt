We assume the files do contain the data in the following format.

mesh.msh:
  A Gmsh mesh file describing the geometry of the volume conductor. The reader currently only works for version 2 msh-files.

conductivities.txt:
  A list of conductivity values. In the .msh-file, each element is given a physical entity number i. DUNEuro then interpretes the i-th entry in conductivities.txt as the conductivity of the corresponding element. 
