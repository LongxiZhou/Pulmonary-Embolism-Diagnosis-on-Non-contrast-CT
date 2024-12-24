# Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT
Code availability for project of "Accurate Diagnosis of Pulmonary Embolism on Non-contrast CT via Verifiable Learning and Sub-visual Analysis"

## Overview
This repository provides Code availability for:
```
Longxi Zhou, et al. "Accurate Diagnosis of Pulmonary Embolism on Non-contrast CT via Verifiable Learning and Sub-visual Analysis".
```

## Description
This project proposes the Sub-visual Pulmonary Emboli Analysis (SPEA) for diagnosing pulmonary embolism (PE) on non-contrast CT scans. 

PE, caused by a blockage of blood flow in the lung arteries, results in millions of deaths annually. Notably, 93% of fatal PE cases occur within 2.5 hours of the initial symptoms, but the current diagnostic workflow for PE remains slow and inefficient. Consequently, the mortality rate of PE in the United States has remained unchanged for over three decades.

The symptoms of PE are often non-specific, and in most cases, PE is not initially suspected. When general tests, such as non-contrast CT, fail to explain the symptoms, further evaluations may reveal evidence suggestive of PE. Directly diagnosing PE on non-contrast CT can considerably reduce PE mortality, but it has long been considered as impossible for human experts due to the faint visibility of emboli.

Trained with 43,841 scans, SPEA can accurately diagnose PE on non-contrast CT scans with area-under-the-curve (AUC) scores of 0.895, 0.877, and 0.887 in internal, external, and real clinical translation, respectively. 

SPEA utilized an interpretable training strategy, and proposed a verifiable metric that achieved accurate posterior probability calibration with Brier score of 0.093 during real clinical translation.



## SPEA Development and Evaluations
<div align="center">
  <img src="./github_resources/Figure 1.png" width=1200>
</div>

## Run SPEA Method
- Step 1): Dowload the source codes from github (note in github, folder ./Data_and_Models is empty).
- Step 2): Download the file: "data_and_models.zip" from [Google Drive](https://drive.google.com/file/d/1QqhQwuZklHq2sY3fs3F7hgbriHCOJ6m2/view?usp=drive_link).
- Step 3): Move "data_and_models.zip" into ./Data_and_Models, then decompress.
- Step 4): Establish the python environment with './torch_2022-20240519.yml'.
- Step 5): Open './pulmonary_embolism_final/inference/predict_clot_from_raw.py', modify the top_dir_models to your local directory of folder "./Data_and_Models"
- Step 6): Run 'predict_clot_from_raw.py', to see the example data.
- Step 7): Change the path for .dcm files to predict your own data.

## Time and Memory Complexity
SPEA is light and can be run on most gaming laptop.
- SPEA requires GPU RAM >= 8GB and CPU RAM >= 24 GB
- Inference needs about 35 seconds on one V100 GPU + 175 seconds (single thread) on CPU + 400 seconds (multi-threading) on CPU

## Contact
If you requrest our training data, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa

## License
[Apache License 2.0](https://github.com/LongxiZhou/Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT/blob/main/LICENSE)
