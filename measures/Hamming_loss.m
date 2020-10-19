function [ HammingLoss ] = Hamming_loss( predict, true )
% Computing Hamming loss, for example:
% dimension(predict) = num_instance * num_class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1] or
% predict = [0, 1, 1; 1, 1, 0];true = [1, 0, 1; 1, 1, 0]
% return 1/3
    [num_samples, num_class] = size(true);
    HammingLoss = sum(sum(double(predict ~= true), 2)) / (num_samples * num_class);
end
