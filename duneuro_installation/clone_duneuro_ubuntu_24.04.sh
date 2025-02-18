#!/bin/bash

git clone --branch ubuntu_24.04_update https://github.com/MalteHoel/duneuro_local_subtraction.git

python3 duneuro_local_subtraction/ci_information/resolve_dune_dependencies.py --outputbasedir .
python3 duneuro_local_subtraction/ci_information/get_python_bindings.py --outputbasedir .

git clone --branch ubuntu_24.04_update --recursive https://gitlab.dune-project.org/duneuro/simbiosphere.git

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
git clone --branch ubuntu_24.04_update https://github.com/MalteHoel/duneuro_eeg_forward_test.git

# clone for quick MEG forward solution test
git clone --branch visualization_update https://github.com/MalteHoel/duneuro_meg_forward_test.git
