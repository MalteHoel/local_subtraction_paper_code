#!/bin/bash
for i in common geometry localfunctions grid istl; do
	git clone --branch releases/2.8 https://gitlab.dune-project.org/core/dune-$i.git;
done;

for i in functions typetree uggrid; do
	git clone --branch releases/2.8 https://gitlab.dune-project.org/staging/dune-$i.git;
done;
for i in pdelab; do
	git clone --branch releases/2.7 https://gitlab.dune-project.org/pdelab/dune-$i.git;
done;

for i in alugrid; do
	git clone --branch releases/2.8 https://gitlab.dune-project.org/extensions/dune-$i.git;
done;

git clone --branch vtu_writer_dune_grid https://github.com/MalteHoel/duneuro_localized_subtraction_cg.git

git clone --branch reworked_visualization https://github.com/MalteHoel/duneuro-py.git

git clone --branch reworked_visualization https://github.com/MalteHoel/duneuro-matlab-dev.git

# download and build simbiosphere
for i in simbiosphere; do
        git clone --recursive https://gitlab.dune-project.org/duneuro/$i.git
done;

cd simbiosphere
mkdir build
cd build
cmake ..
make
cd ..
cd ..

# clone for analytic MEG solution
git clone --branch duneuro-py-dependent https://github.com/MalteHoel/duneuro-analytic-solution.git

# clone for quick EEG forward solution test
git clone --branch visualization_update https://github.com/MalteHoel/duneuro_eeg_forward_test.git

# clone for quick MEG forward solution test
git clone --branch visualization_update https://github.com/MalteHoel/duneuro_meg_forward_test.git
