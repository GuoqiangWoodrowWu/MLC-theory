# Multi-label classification: do Hamming loss and subset accuracy really conflict with each other?
This repository is the official implementation of "Multi-label classification: do Hamming loss and subset accuracy really conflict with each other?" accepted in NeurIPS 2020.
## Programming Language
The source code is written by Matlab
## File description
- ./Datasets -- the datasets downloaded from the websites http://mulan.sourceforge.net/datasets-mlc.html and http://palm.seu.edu.cn/zhangml/
- ./measures -- the measures for MLC including Hamming Loss, Subset Accuracy and Ranking Loss
- ./Results -- store the experimental results
- ./CrossValidation.m -- used to create cross-validation data
- ./train_hinge_hamming_SVRG_BB.m -- utilize SVRG-BB to train the optimizing hamming loss directly with its surrogate loss (i.e. A^h) where the base loss function is hinge loss
- ./train_hinge_subset_SVRG_BB.m -- utilize SVRG-BB to train the optimizing subset loss directly with its surrogate loss (i.e. A^s) where the base loss function is hinge loss
- ./Predict.m -- predict the model
- ./Evaluation_Metrics.m -- evaluate the model on measures including Hamming Loss, Subset Accuracy and Ranking Loss
- run_linear_hamming.m -- run the code to evaluate A^h
- run_linear_subset.m -- run the code to evaluate A^s
## Run
Run the run_linear_hamming.m and run_linear_subset.m in MATLAB, and it will run as its default parameters on sample datasets.
