# CBMLB: Circulating biomarker liquid biopsy

Python package to compute the shedding of a biomarker from cancer cells into the bloodstream and its analysis in liquid biopsies (small blood samples). 
See preprint Avanzini et al. (2020) for further details: https://doi.org/10.1101/2020.02.12.946228

### <a name="releases"> Releases
* CBMLB 0.1.0 2020-05-06: Initial release of CBMLB package.


### <a name="installation"> Installation and Setup
1. Easiest is to install Mini anaconda and create a new python environment in a terminal window with ```conda create --name py36 python=3.6``` and activate it with ```conda activate py36```
2. Clone the repository from GitHub with ```git clone https://github.com/johannesreiter/cbmlb.git``` 
3. If you want to have system-wide access, create distribution packages by going into the main folder with ```cd <CBMLB_DIRECTORY>```, run ```python3 setup.py clean sdist bdist_wheel``` and install CBMLB to your python environment by executing ```pip3 install -e <CBMLB_DIRECTORY>```
4. Test installation with ```python3 -c 'import cbmlb'``` and ```python3 -m unittest discover <CBMLB_DIRECTORY>/tests/```
5. To uninstall the package use ```pip3 uninstall cbmlb``` or ```conda remove cbmlb```


### <a name="examples"> Examples
1. Simulate tumor growth and ctDNA shedding dynamics of 10 cancers: ```cbmlb dynamics -b 0.14 -d 0.13 -M 1e10 -n 10```
2. Simulate ctDNA at a given tumor size for 100 subjects: ```cbmlb distribution -b 0.14 -d 0.13 -n 100 -M 1e8 --q_d=1.4e-4```
3. Simulate monthly relapse testing for previously simulated tumor growth and shedding dynamics: ```cbmlb detection monthly -b 0.14 -d 0.13 -M 5e11 --panel_size 20 --n_muts 20 --pval_th 8.3e-04 --seq_eff 0.5 --imaging_det_size 1e9```
4. Simulate annual screening for previously simulated tumor growth and shedding dynamics: ```cbmlb detection annually -b 0.14 -d 0.136 -M 1e11 --panel_size 300000 --n_muts 10 --pval_th 1.5e-7 --seq_eff 0.5 --diagnosis_size 2.25e10```

See ```<CBMLB_DIRECTORY>/cbmlb/settings.py``` for default parameter values.

Authors: Stefano Avanzini & Johannes Reiter, Stanford University, https://reiterlab.stanford.edu
