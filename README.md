# Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT


## Overview
This repository provides Code Availability for the engneering details.

## Background about the disease
Sub-visual Pulmonary Thrombi Analysis (SPTA) can diagnosing pulmonary embolism (PE) on non-contrast CT scans. 

PE, defined by a blockage of blood flow in the lung arteries, results in millions of deaths annually. PE typically presents as an acute event, and timely treatment can reduce its mortality from approximately 30% to less than 10%, but autopsy studies indicated that more than 50% of PE were missed diagnosed or misdiagnosed.

The symptoms of PE are often non-specific, and the confirmatory test is CT pulmonary angiography (CTPA). But CTPA is usually inaccessible in resource-limited regions. In China, 93.82% CPTA conducted in Tier 3 hospitals, and the US performed 14.6-fold higher per capita rate of CTPA than China. This stark contrast underscores the significant disparity in access to CTPA, particularly in rural areas, smaller cities, and township hospitals.

Directly diagnosing PE on non-contrast CT is clinically significant, but it has long been considered as impossible for human experts due to the faint visibility of thrombi.

Trained with 43,841 scans, SPTA can accurately diagnose PE on non-contrast CT scans with area-under-the-curve (AUC) scores of 0.895, 0.877, and 0.890 in internal, external, and prospective observational dataset, respectively. 

SPEA utilized an interpretable training strategy, and proposed a verifiable metric that achieved accurate posterior probability calibration with an expected calibration error (ECE) less than 1% during the prospective observational study.


## Run SPTA Method
We provide a simplified version of SPTA (25% model parameter of the complete version), with AUC of 0.870 on the internal test set.
- Step 1): Dowload the source codes from github (note in github, folder ./Data_and_Models is empty).
- Step 2): Download the file: "data_and_models.zip" from [Google Drive](https://drive.google.com/file/d/17oBAySfVm8WAFWj31fSaU4JrrDtZYz5Z/view?usp=sharing).
- Step 3): Move "data_and_models.zip" into ./Data_and_Models, then decompress.
- Step 4): Establish the Conda environment with './Environment_SPEA.yml'.
- Step 5): Open './pulmonary_embolism_final/inference/predict_clot_from_raw.py', modify the top_dir_models to your local directory of folder "./Data_and_Models"
- Step 6): Read and run 'predict_clot_from_raw.py', to see the example data.
- Step 7): Change the path for .dcm files to predict your own data.


## Recalibrate SPTA on Your Clinical Setting

In real-world practice, **posterior probability calibration** is essential for making AI predictions clinically actionable—especially when doctors cannot directly verify the outputs.

Most existing AI models produce fixed probability scores (e.g., 80%) based only on imaging features, without considering the clinical scenario or disease prevalence. However, the same scan could come from dramatically different settings:

| Clinical Scenario                                | Typical PE Prevalence | Output from Uncalibrated AI | Output from SPEA (Calibrated) |
|--------------------------------------------------|------------------------|------------------------------|-------------------------------|
| General non-contrast CT for routine check        | ~0.1%                  | 80%                          | **5%**                        |
| Emergency non-contrast CT                        | ~1%                    | 80%                          | **30%**                       |
| Non-contrast CT from suspected PE patients       | ~20%                   | 80%                          | **95%**                       |

Without calibration, an “80%” prediction could mean wildly different real-world risks—causing either **overdiagnosis** or **missed critical cases**.

SPEA solves this with a **training-free recalibration** based on our proposed predictive indicator "Relative Emboli Ratio", allowing you to dynamically adapt posterior probability outputs to your clinical setting.

To recalibrate SPTA on your local population, see:
./pulmonary_embolism_final/inference/posterior_pe_adapt_to_new_site.py

---
### How Well-Calibrated is SPEA?

SPEA's predictions are not just interpretable—they are also **exceptionally well-calibrated**. In our prospective study, the recalibrated probability mapping achieved:
- **Expected Calibration Error (ECE)**: ~1%
  
**ECE** reflects the average deviation between predicted probabilities and actual event frequencies. For example, if SPEA assigns an 80% PE probability across a group of patients, an ECE of 0.0258 suggests that the actual rate of PE among those patients would fall between approximately 79% and 81%.

By contrast, AI models in medical image classification typically show **ECEs between 10% and 40%**, indicating poor probability calibration. This gap highlights why **posterior calibration is critical**: models with poor ECEs may give confident but unreliable probabilities, making clinical decisions risky.

SPTA closes this gap, delivering reliable probabilistic outputs that clinicians can trust.


## Time and Memory Complexity
SPTA is light and can be run on most gaming laptop.
- SPTA requires GPU RAM >= 8GB and CPU RAM >= 24 GB
- Inference needs about 35 seconds on one V100 GPU + 175 seconds (single thread) on CPU + 600 / num_cpu seconds


## Contact
If you requrest our training data or SPEA with the complete model parameter, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa


## License
[Apache License 2.0](https://github.com/LongxiZhou/Pulmonary-Embolism-Diagnosis-on-Non-contrast-CT/blob/main/LICENSE)
