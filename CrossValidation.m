function [ X_train, Y_train, X_test, Y_test ] = CrossValidation( X, Y, cv_all, i )
% Input: size(X) = [n_instances, n_features] 
%        size(Y) = size(n_instances, n_labels)
%        cv_all: the fold number of cross validation
%        i: the i-th cross validation
%        Y \in {-1, +1}
% Output: the train and test dataset

    [num_samples, num_feature] = size(X);
    % X = mapminmax(X, -1, 1);
    block = floor(num_samples / cv_all);
    test_index = [ block*(i - 1) + 1 : block*i ];
    train_index = [ 1:block*(i-1), block*i + 1 : num_samples];
    X_train = X(train_index, :);
    Y_train = Y(train_index, :);
    X_test = X(test_index, :);
    Y_test = Y(test_index, :);
end

