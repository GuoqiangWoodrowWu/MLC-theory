clear;
close all;
clc;

% dataset name
dataset_name = 'emotions';
path_main = pwd;
path_save = strcat(path_main, filesep, 'Results');
path_data = strcat(path_main, filesep, 'Datasets');
file_save = strcat(path_save, filesep, 'Results_linear_subset.csv');

% setting
rand('seed', 2^40);
result = [];
K_fold = 3; % cross-validation: 3-fold
lambda = 0.1;

% Loading the dataset
file_name = strcat(path_data, filesep, dataset_name, '.mat');
S = load(file_name);

X_all = S.data;
Y_all = S.target;
Y_all(Y_all < 1) = -1;
% append one feature with all equal to 1 to correspond to the bias
num_feature_origin = size(X_all, 2);
X_all(:, num_feature_origin + 1) = 1;

%normalization
[X_all, PS] = mapstd(X_all', 0, 1);
X_all = X_all';

% Shuffle the dataset
[num_samples, num_feature] = size(X_all);
shuffle_index = randperm(num_samples);
X_all = X_all(shuffle_index, :);
Y_all = Y_all(shuffle_index, :);

% Do cross-validation
hl = zeros(1, K_fold);
sa = zeros(1, K_fold);
ranking_loss = zeros(1, K_fold);

for index_cv = 1: K_fold
    [X_train, Y_train, X_vali, Y_vali] = CrossValidation(X_all, Y_all, K_fold, index_cv);      
    % train the train dataset and predict the test dataset 
    alpha = 0.0001;
    [ W, obj ] = train_hinge_subset_SVRG_BB( X_train, Y_train, lambda, alpha );
    [ pre_Label_vali, pre_F_vali ] = Predict( X_vali, W );

    [ HammingLoss,SubsetAccuracy,Ranking_Loss ] = Evaluation_Metrics( pre_Label_vali, pre_F_vali, Y_vali );

    hl(index_cv) = HammingLoss;
    sa(index_cv) = SubsetAccuracy;
    ranking_loss(index_cv) = Ranking_Loss;

end
toc;
time = double(toc);

HL_cv_mean = mean(hl); 
HL_cv_std = std(hl); 
SA_cv_mean = mean(sa);
SA_cv_std = std(sa);
RANKING_LOSS_cv_mean = mean(ranking_loss);
RANKING_LOSS_cv_std = std(ranking_loss);
result = [result; HL_cv_mean HL_cv_std SA_cv_mean SA_cv_std ...
RANKING_LOSS_cv_mean RANKING_LOSS_cv_std lambda time];
csvwrite(file_save, result);
