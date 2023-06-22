We assume the files do contain the data in the following format.

electroces.txt:
  Each row contains the three cartesian coordinates of the electrodes. The number of rows corresponds to the number of electrodes.

coils.txt:
  Each row contains the three cartesian coordinates of the coil position. The number of rows corresponds to the number of coils.

projections.txt:
  DUNEuro computes values of the form <B_S(x), v>, where x is the coil position and v is a so called projection. If now x1, x2, x3 is the i-th row of coils.txt, v11, v12, v13, ...., vn1, vn2, vn3 is the i-th row of projections.txt, and we set x = (x1, x2, x3) and vi = (vi1, vi2, vi3), DUNEuro will compute the values <B_S(x), v1>, ..., <B_S(x), vn>.
