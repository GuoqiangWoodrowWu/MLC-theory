function [ SubsetAccuracy ] = Subset_accuracy( predict, true)
% Computing subset accuracy, for example:
% dimension(predict) = num_instance * num_class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1] or
% predict = [0, 1, 1; 1, 1, 0];true = [1, 0, 1; 1, 1, 0]
% return 0.5
    [num_samples, num_class] = size(true);
    is_equal = sum(double(predict == true), 2);
    equal_pairs = 0;
    for i = 1: num_samples
        if is_equal(i) == num_class
            equal_pairs = equal_pairs + 1;
        end
    end
    SubsetAccuracy = equal_pairs / num_samples;
end

