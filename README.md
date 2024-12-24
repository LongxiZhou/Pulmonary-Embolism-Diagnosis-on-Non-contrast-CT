# Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT
Code availability for project of "Accurate Diagnosis of Pulmonary Embolism on Non-contrast CT via Verifiable Learning and Sub-visual Analysis"

## Overview
This repository provides Code availability for:
```
Longxi Zhou, et al. "Accurate Diagnosis of Pulmonary Embolism on Non-contrast CT via Verifiable Learning and Sub-visual Analysis".
```

## Description
This project propose the Sub-visual Pulmonary Emboli Analysis (SPEA) for diagnosing pulmonary embolism (PE) on non-contrast CT scans. PE, caused by a blockage of blood flow in the lung arteries, results in millions of deaths annually. Notably, 93% of fatal PE cases occur within 2.5 hours of the initial symptoms, but the current diagnostic workflow for PE remains slow and inefficient. Consequently, the mortality rate of PE in the United States has remained unchanged for over three decades.

Using non-contrast CT for PE diagnosis offers the potential to detect the condition at the onset of symptoms. However, identifying blockages on non-contrast CT is extremely challenging for human experts due to subtle visual cues. Our study demonstrates the successful application of artificial intelligence in accurately detecting these blockages on non-contrast CT, which may considerably reduce PE mortality.

## SPEA Development and Evaluations
<div align="center">
  <img src="./github_resources/Figure 1.png" width=1200 height=1200>
</div>

## Run SPEA Method
- Step 1): Dowload the source codes from github (note in github, folder ./Data_and_Models is empty).
- Step 2): Download the file: "data_and_models.zip" from [Google Drive](https://drive.google.com/file/d/1QqhQwuZklHq2sY3fs3F7hgbriHCOJ6m2/view?usp=drive_link).
- Step 3): Move "data_and_models.zip" into ./Data_and_Models, then decompress.
- Step 4): Establish the python environment by './torch_2022-20240519.yml'.
- Step 5): Open './pulmonary_embolism_final/inference/predict_clot_from_raw.py', modify the top_dir_models to your local directory: "./Data_and_Models"
- Step 6): Run 'predict_clot_from_raw.py', to see the example data.
- Step 7): Change the path for .dcm files to predict your own data.
