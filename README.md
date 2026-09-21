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

## 3) Fixel-based analysis
- FBA using tools from MRtrix3 and MRtrix3tissue
- registration of DWIs to a population template based on 40 subjects
- extraction of metrics of fiber density (FD) and fiber cross-section (FC) for each subject and fiber bundle
  
## 4) Group difference of tract metrics
- assessment of group differences of FD and Fc in the fiber bundles
- script _group_difference.py_ provided in _code_
- input:
    - folder with one .csv file per metric containing hemisphere-averaged values for each bundle and covariates for each subject

