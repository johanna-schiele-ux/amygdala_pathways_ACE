# amygdala_pathways_ACE
Amygdala pathway group difference and dimensional analysis of exposure, personality and symptoms in adults with adverse childhood experience (ACE) with and without PTSD 

## Workflow

## 1) Preprocessing
- standard preprocessing using tools from MRtrix3, FSl and ANTs
- steps:
    - denoising
    - Gibbs ringing removal
    - motion & distortion correction
    - bias field correction
    - resampling to 1mm isotropic voxel size
    - brain extraction

## 2) Fiber bundle definition
- reconstruction of three amygdala pathways in MNI2009b space
- Uncinate fasciculus: standard TractSeg-derived version
- Cingulum Bundle: TractSeg-derived version complimented with manual tracking using tckgen to reconstruct amygdala connections
- Stria Terminalis: Manual tracking using tckgen
- reconstructed tract files in MNI2009b space are provided in _data/tracts_
- masks used for tracking are provided in  _data/masks_

## 3) Fixel-based analysis
- FBA using tools from MRtrix3 and MRtrix3tissue
- registration of DWIs to a population template based on 40 subjects
- extraction of metrics of fiber density (FD) and fiber cross-section (FC) for each subject and fiber bundle
  
## 4) Group difference of tract metrics
- assessment of group differences of FD and FC in the fiber bundles
- conducts assumption checks and ANCOVA per bundle and metric
    - age included as covariate for all models, FC model additionally contain brain volume
    - option to add total CTQ as additional covariate for all models
- script _group_difference.py_ provided in _code_
- input:
    - folder with one .csv file per metric containing hemisphere-averaged values for each bundle and covariates for each subject
- model outputs containing full results are provided in _results/results_group_analysis.odt_

## 5) Dimensional Analysis:
## a) Robust regression to predict clinical scores from tract metrics
- assessment of association between tract metrics and exposure (CTQ), personality (PID5), and symptoms (PCL5)
- conducts assumption checks and robust Huber regression
    - age included as covariate, FC values are residualized on brain volume 
- script _dimensional_analyses.py_ provided in _code_
- option to choose group and target clinical score 
- input:
    - six different .csv files, script chooses the appropriate two (one for each metric) according to chosen group and target
          - files contain tract values, covariates and clinical scores for each subject
          - 2 files for group without PTSD
          - 2 files for group with PTSD including all 51 subjects
          - 2 files for group with PTSD including only 50 subjects used for PID5 analyses, since PID5 data was not available for one
- model outputs containing full results for both groups are provided in _results/results_dimensional_analyses_NoPTSD.odt_ and  _results/results_dimensional_analyses_PTSD.odt_

## b) Spearman's rank correlation between tract metrics and clinical scores
- assessment of direct linear correlation between tract metrics and exposure, personality, and symptoms
