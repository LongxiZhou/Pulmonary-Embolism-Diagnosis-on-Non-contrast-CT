![image](https://github.com/user-attachments/assets/d5d75134-2c7c-46ca-9c4d-9b9de086de3e)![image](https://github.com/user-attachments/assets/c934ce62-1e5e-43d9-94ab-598af536cfa7)# Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT
Code availability for project of "Accurate Diagnosis of Pulmonary Embolism on Non-contrast CT via Verifiable Learning and Sub-visual Analysis"


## Overview
This repository provides Code availability for:
```
Longxi Zhou, et al. "Beyond Perceptual Limit: Verifiable AI for Pulmonary Embolism Detection on Non-contrast CT".
```

## Description
This project proposes the Sub-visual Pulmonary Emboli Analysis (SPEA) for diagnosing pulmonary embolism (PE) on non-contrast CT scans. 

PE, caused by a blockage of blood flow in the lung arteries, results in millions of deaths annually. Notably, 93% of fatal PE cases occur within 2.5 hours of the initial symptoms, but the current diagnostic workflow for PE remains slow and inefficient. Consequently, the mortality rate of PE in the United States has remained unchanged for over three decades.

The symptoms of PE are often non-specific, and in most cases, PE is not initially suspected. When general tests, such as non-contrast CT, fail to explain the symptoms, further evaluations may reveal evidence suggestive of PE.  Around 70%-90% PE patients received non-contrast CT before being suspected as PE. Directly diagnosing PE on non-contrast CT can considerably reduce PE mortality, but it has long been considered as impossible for human experts due to the faint visibility of emboli.

Trained with 43,841 scans, SPEA can accurately diagnose PE on non-contrast CT scans with area-under-the-curve (AUC) scores of 0.895, 0.877, and 0.887 in internal, external, and prospective observational dataset, respectively. 

SPEA utilized an interpretable training strategy, and proposed a verifiable metric that achieved accurate posterior probability calibration with Brier score of 0.093 and an expected calibration error (ECE) of 0.0258 during the prospective observational study.


## SPEA Development and Evaluations
<div align="center">
  <img src="./github_resources/Figure 1.png" width=1200>
</div>


## Run SPEA Method
We provide a simplified version of SPEA (25% model parameter of the complete version), with AUC of 0.870 on the internal test set.
- Step 1): Dowload the source codes from github (note in github, folder ./Data_and_Models is empty).
- Step 2): Download the file: "data_and_models.zip" from [Google Drive](https://drive.google.com/file/d/17oBAySfVm8WAFWj31fSaU4JrrDtZYz5Z/view?usp=sharing).
- Step 3): Move "data_and_models.zip" into ./Data_and_Models, then decompress.
- Step 4): Establish the Conda environment with './Environment_SPEA.yml'.
- Step 5): Open './pulmonary_embolism_final/inference/predict_clot_from_raw.py', modify the top_dir_models to your local directory of folder "./Data_and_Models"
- Step 6): Read and run 'predict_clot_from_raw.py', to see the example data.
- Step 7): Change the path for .dcm files to predict your own data.


## Recalibrate SPEA on Your Clinical Setting

In real-world practice, **posterior probability calibration** is essential for making AI predictions clinically actionable—especially when doctors cannot directly verify the outputs.

Most existing AI models produce fixed probability scores (e.g., 80%) based only on imaging features, without considering the clinical scenario or disease prevalence. However, the same scan could come from dramatically different settings:

| Clinical Scenario                                | Typical PE Prevalence | Output from Uncalibrated AI | Output from SPEA (Calibrated) |
|--------------------------------------------------|------------------------|------------------------------|-------------------------------|
| General non-contrast CT for routine check        | ~0.1%                  | 80%                          | **5%**                        |
| Emergency non-contrast CT                        | ~1%                    | 80%                          | **30%**                       |
| Non-contrast CT from suspected PE patients       | ~20%                   | 80%                          | **95%**                       |

Without calibration, an “80%” prediction could mean wildly different real-world risks—causing either **overdiagnosis** or **missed critical cases**.

SPEA solves this with a **training-free recalibration** based on our proposed predictive indicator "Relative Emboli Ratio", allowing you to dynamically adapt posterior probability outputs to your clinical setting.

To recalibrate SPEA on your local population, see:
./pulmonary_embolism_final/inference/posterior_pe_adapt_to_new_site.py

---
### How Well-Calibrated is SPEA?

SPEA's predictions are not just interpretable—they are also **exceptionally well-calibrated**. In our prospective study, the recalibrated probability mapping achieved:
- **Expected Calibration Error (ECE)**: 0.0258
  
**ECE** reflects the average deviation between predicted probabilities and actual event frequencies. For example, if SPEA assigns an 80% PE probability across a group of patients, an ECE of 0.0258 suggests that the actual rate of PE among those patients would fall between approximately 77.5% and 82.5%.

By contrast, AI models in medical image classification typically show **ECEs between 0.1 and 0.4**, indicating poor probability calibration. This gap highlights why **posterior calibration is critical**: models with poor ECEs may give confident but unreliable probabilities, making clinical decisions risky.

SPEA closes this gap, delivering reliable probabilistic outputs that clinicians can trust.


## Time and Memory Complexity
SPEA is light and can be run on most gaming laptop.
- SPEA requires GPU RAM >= 8GB and CPU RAM >= 24 GB
- Inference needs about 35 seconds on one V100 GPU + 175 seconds (single thread) on CPU + 600 / num_cpu seconds


## Contact
If you requrest our training data or SPEA with the complete model parameter, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa


## License
[Apache License 2.0](https://github.com/LongxiZhou/Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT/blob/main/LICENSE)
