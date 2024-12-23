# The aim is to detect pulmonary embolism on non-contrast CT.
The project contains these sections:
1. Up-sample z to enable blood vessel segmentation.
2. Registration of non-contrast CT and CTA to get blood clot annotation on for non-contrast CT.
3. Predict pulmonary embolism (where is the blood clot) on non-contrast CT.
4. Clinical practice to check how to apply our method.

Section 1 and 2 are sub-parts of project "lung_atlas".


# Why use transformer
We use the transformer to predict pulmonary embolism, it has these advantages:
1. Easy for semi-supervised learning.
2. Can look for long-range information, and pulmonary embolism needs very long-range information.
Consider a case: blood clot in a vessel, it cause very slight change in CT value, then the model
need look for other vessels, especially vessels in symmetric positions and vessels with similar
radius. These vessels are far far away from the blood clot.
3. Easy to transfer for other tasks.

The second advantage for transformer may be unsolvable by CNN.


# To apply transformer, we follow these steps:
1. Segment the arteries and veins.
2. Extract center lines for arteries and veins. (help to reduce cube number).
3. Along the center lines, select cubes.
4. Cast cubes into embedding space.
5. Feed into transformer model.


# About cube embedding
The embedding should reflect the similarity of the cubes.
Embedding contains the following parts:
1. Position encoding: locations away from the mass center of blood vessels
2. Feature vector abstracted by CNN.

About 2,
Use 3D convolution to learn the embedding during training.
ViT 14 * 14 = 196 pixels used around 768 kernels
We use 5 * 5 * 5 = 125 voxels also use around 768 kernels
(existing 3D transformer uses one liner mapping to convert cube to vector,  
existing 3D transformer divide the volume into 16 * 16 * 16 cubes,  
we divide into 47 * 47 * 47 cubes)

Use same algorithm for ViT for our model, each scan requires around 2.5 TB GPU ram ~ 100 V100
So our model is for refine, you have to find a ROI before apply the model


# Model pre-training
The model contains the notion of how the tissue looks like
The idea is similar with MAE: mask a lot of cubes, then try to recover it.
1. The idea
