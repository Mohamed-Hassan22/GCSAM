# GCSAM: Gradient Centralized Sharpness Aware Minimization
This repository contains Gradient Centralized Sharpness-Aware Minimization (GCSAM) for Deep Neural Netowrks (DNNs) and vision transformers.
This is an official repository for [GCSAM: Gradient Centralized Sharpness Aware Minimization](https://arxiv.org/abs/2501.11584)

## Abstract 
The generalization performance of deep neural networks (DNNs) is a critical factor in achieving robust model behavior on unseen data. Recent studies have highlighted the importance of sharpness-based measures in promoting generalization by encouraging convergence to flatter minima. Among these approaches, Sharpness-Aware Minimization (SAM) has emerged as an effective optimization technique for reducing the sharpness of the loss landscape, thereby improving generalization. However, SAM's computational overhead and sensitivity to noisy gradients limit its scalability and efficiency. To address these challenges, we propose Gradient-Centralized Sharpness-Aware Minimization (GCSAM), which incorporates Gradient Centralization (GC) to stabilize gradients and accelerate convergence. GCSAM normalizes gradients before the ascent step, reducing noise and variance, and improving stability during training. Our evaluations indicate that GCSAM consistently outperforms SAM and the Adam optimizer in terms of generalization and computational efficiency. These findings demonstrate GCSAM's effectiveness across diverse domains, including general and medical imaging tasks.

## Citation 
```
@article{hassan2025gcsamgradientcentralizedsharpness,
author = {Mohamed Hassan and Aleksandar Vakanski and Boyu Zhang and Min Xian},
title = {GCSAM: Gradient Centralized Sharpness Aware Minimization},
journal = {arXiv preprint arXiv:2501.11584},
year = {2025}
}
```

## References
[1] [Foret et al. 2021](https://arxiv.org/abs/2010.01412)  
[2] [Yong et al. 2020](https://arxiv.org/abs/2004.01461)  
