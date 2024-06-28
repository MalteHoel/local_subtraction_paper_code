We assume the files to be organized as follows.

mesh.msh:
  A Gmsh mesh file describing the geometry of the volume conductor. The reader currently only works for version 2 msh-files.

conductivities.txt:
  A list of conductivity values. In the .msh-file, each element is given a physical entity number i. DUNEuro then interprets the i-th entry in conductivities.txt as the conductivity of the corresponding element.

The mesh was constructed using gmsh software, see

Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities
(DOI : https://doi.org/10.1002/nme.2579)

or directly at https://gmsh.info/.
