![build](https://github.com/reiterlab/ctdna/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/reiterlab/ctdna/branch/master/graph/badge.svg)](https://codecov.io/gh/reiterlab/ctdna)
![PyPI](https://github.com/reiterlab/ctdna/workflows/PyPI/badge.svg)

# ctDNA: Circulating tumor DNA

Python package to compute the shedding of a biomarker from cancer cells into the bloodstream and its analysis in liquid biopsies (small blood samples). 
The package can be run from the command line or various methods can be important for customized calculations. 
See examples below.
See preprint Avanzini et al. (2020) for further technical details: https://doi.org/10.1101/2020.02.12.946228

### <a name="releases"> Releases
* ctdna 0.1.0 2020-05-06: Initial release of package.
* ctdna 0.1.1 2020-08-18: Added various examples and unittests. Added methods to calculate detection probabilities.


### <a name="installation"> Installation and Setup
1. Easiest is to install Mini anaconda and create a new python environment in a terminal window with ```conda create --name py36 python=3.6``` and activate it with ```conda activate py36```
2. Clone the repository from GitHub with ```git clone https://github.com/reiterlab/ctdna.git``` 
3. If you want to have system-wide access, create distribution packages by going into the main folder with ```cd <CTDNA_DIRECTORY>```, run ```python setup.py clean sdist bdist_wheel``` and install ```ctdna``` to your python environment by executing ```pip install -e <CTDNA_DIRECTORY>```
4. Test installation with ```python -c 'import ctdna'``` and ```pytest <CTDNA_DIRECTORY>/tests/```
5. To uninstall the package use ```pip uninstall ctdna``` or ```conda remove ctdna```


### <a name="examples"> Manual

```ctdna``` can be run in the following modes: 
1. ```dynamics```: simulates the evolutionary dynamics of a tumor and its biomarkers over time
2. ```distribution```: simulates the biomarker distribution at a given tumor size or tumor age
3. ```detection```: simulates the detection size of a growing tumor for repeated testing at a desired annual false positive rate or a specified p-value threshold 
4. ```roc```: computes the ROC (receiver operating characteristic) 


See ```<CTDNA_DIRECTORY>/ctdna/settings.py``` for default parameter values.


### <a name="examples"> Examples
- [example.ipynb](example.ipynb): Calculate the probability to detect a specific actionable mutation in tumors with diameters of 1, 1.2, 1.5, and 2 cm with a specificity of 99%

- Simulate tumor growth and ctDNA shedding dynamics of 10 cancers: ```ctdna dynamics -b 0.14 -d 0.13 -M 1e10 -n 10```
- Simulate ctDNA at a given tumor size for 100 subjects: ```ctdna distribution -b 0.14 -d 0.13 -n 100 -M 1e8 --q_d=1.4e-4```
- Simulate monthly relapse testing for previously simulated tumor growth and shedding dynamics: ```ctdna detection monthly -b 0.14 -d 0.13 -M 1e10 --panel_size 20 --n_muts 20 --annual_fpr 0.05 --seq_eff 0.5 --imaging_det_size 1e9```
- Simulate annual screening with CancerSEEK for previously simulated tumor growth and shedding dynamics: ```ctdna detection annually -b 0.14 -d 0.136 -M 1e11 --panel_size 2000 --n_muts 1 --annual_fpr 0.01 --seq_eff 0.5 --diagnosis_size 2.25e10```
- Simulate annual screening with CAPPSeq for previously simulated tumor growth and shedding dynamics: ```ctdna detection annually -b 0.14 -d 0.136 -M 1e11 --panel_size 300000 --n_muts 10 --pval_th 1.5e-7 --seq_eff 0.5 --diagnosis_size 2.25e10```

<!-- 
Fig. S6B,C: ```ctdna detection monthly -b 0.14 -d 0.13 -M 5e11 --panel_size 20 --n_muts 20 --annual_fpr 0.06 --seq_eff 1.0 --imaging_det_size 1e9 --n_replications 10```
Fig. S6B,C: ```ctdna detection quarterly -b 0.14 -d 0.13 -M 5e11 --panel_size 20 --n_muts 20 --annual_fpr 0.02 --seq_eff 1.0 --imaging_det_size 1e9 --n_replications 10```
Fig. S6D,E: ```ctdna detection monthly -b 0.14 -d 0.13 -M 5e11 --panel_size 20 --n_muts 20 --annual_fpr 0.06 --seq_eff 0.5 --imaging_det_size 1e9 --n_replications 10```
Fig. S10A: ```ctdna detection annually -b 0.14 -d 0.13 -M 5e11 --panel_size 2000 --n_muts 1 --annual_fpr 0.01 --seq_eff 0.5 --diagnosis_size 2.25e10 --n_replications 10```
Fig. S10B: ```ctdna detection annually -b 0.14 -d 0.13 -M 5e11 --panel_size 300000 --n_muts 5 --annual_fpr 0.01 --seq_eff 0.5 --diagnosis_size 2.25e10 --n_replications 10```
Fig. S10C: ```ctdna detection annually -b 0.14 -d 0.13 -M 5e11 --panel_size 300000 --n_muts 10 --annual_fpr 0.01 --seq_eff 0.5 --diagnosis_size 2.25e10 --n_replications 10```
-->

### <a name="examples"> Arguments
- ```-b <>``` or ```--birth_rate <>```: birth rate of cancer cells
- ```-d <>``` or ```--death_rate <>```: death rate of cancer cells
- ```--q_d <>```: ctDNA shedding probability per cell death
- ```--q_b <>```: ctDNA shedding probability per cell birth
- ```--lambda1 <>```: ctDNA shedding probability per cell per day
- ```--t12_mins <>```: cfDNA half-life time in minutes
- ```--tube_size <>```: size of blood sample tube (liters; default 0.015 l)

- ```--panel_size <>```: sequencing panel size
- ```--seq_err <>```: sequencing error rate per base-pair (default: 1e-5)
- ```--seq_eff <>```: sequencing efficiency, ie. fraction of the sampled molecules that are actually successfully sequenced (default: 0.5)
- ```--n_muts <>```: number of clonal mutations in the cancer that are covered by the sequencing panel


-  ```-M <>``` or ```--det_size <>```: tumor detection size where biomarker level is evaluated or size where dynamics simulations are stopped
-  ```-T <>``` or ```--sim_time <>```: simulations end when cancer reaches the given time


- ```--exact_th <>```: approximate growth of tumor after this given threshold is reached
-  ```-o <>``` or ```--output_dir <>```: output directory for files (default is defined in ```<CTDNA_DIRECTORY>/ctdna/settings.py```)

- ```--biomarker_wt_freq_ml <>``` Optional argument to fix the wildtype haploid genome equivalents (hGE) per plasma ml to the given number instead of sampling the plasma DNA concentration from a Gamma distribution with parameters specified in ```<CTDNA_DIRECTORY>/ctdna/settings.py```

#### Detection mode
- ```--annual_fpr <>```: Specifies desired annual false positive rate (1 - specificity) if test is repeated at the given frequency over a year
- ```--pval_th <>```: Instead of the desired annual false positive rate ```--annual_fpr```, one can directly provide a p-value threshold that calls somatic point mutations in ctDNA

Authors: Stefano Avanzini & Johannes Reiter, Stanford University, https://reiterlab.stanford.edu
