function [W, obj] = train_hinge_subset_SVRG_BB( X, Y, lambda_1, alpha )
%   Optimize Subset Loss, base loss function: hinge loss
%   alpha: learning_rate
%   lambda_1 for l2 norm
    
    [num_instance, num_feature] = size(X);
    num_label = size(Y, 2);
    W = zeros(num_feature, num_label);
    
    % Do serveral SGD steps first
    for i = 1: 10
        index = randi(num_instance);
        GD_one = calculate_one_gradient(X(index,:), Y(index,:), W, lambda_1);
        W = W - alpha * GD_one;
    end
    
    num_s = 50;
    m = 2 * num_instance;
    epsilon = 0;
    for i = 1: num_s
        W1 = W;
        fG1 = calculate_all_gradient(X, Y, W1, lambda_1);
        if i > 1
            if i > 2 && abs(obj(i-1, 1) - obj(i-2, 1)) / obj(i-2, 1) <= epsilon
                break;
            end
            alpha = norm(W1-W0, 'fro')^2 / trace((W1-W0)'*(fG1-fG0)) / m;
        end
        fG0 = fG1;
        W0 = W1;
        for j = 1: m
            index = randi(num_instance);
            GD_one = calculate_one_gradient(X(index,:), Y(index,:), W, lambda_1);
            GD_ = calculate_one_gradient(X(index,:), Y(index,:), W1, lambda_1);
            W = W - alpha * (GD_one - GD_ + fG1);
            if isnan(W)
               return;
            end
        end
        obj(i,1) = calculate_objective_function(X, Y, W, lambda_1);
        fprintf('Step %d: the objective function value is %.5f\n', i, obj(i,1));
    end
end

function [f_value] = calculate_objective_function(X, Y, W, lambda_1)
    f_value = 0.5 * lambda_1 * norm(W, 'fro')^2;
    [num_instance, num_class] = size(Y);
    
    f_value_subset = calculate_fValue_subset_loss( X, Y, W);

    f_value = f_value + 1 / num_instance * f_value_subset;
end


function [grad] = calculate_all_gradient(X, Y, W, lambda_1)
    [num_instance, num_class] = size(Y);
    num_feature = size(X, 2);

    grad = lambda_1 * W;
    Z_m = zeros(num_feature, num_class);
    
    grad_subset = Z_m;
    
    for i = 1: num_instance
        grad_subset = grad_subset + calculate_one_gradient_subset_loss(...
            X(i,:), Y(i,:), W);
    end

    grad = grad + grad_subset / num_instance;
end

function [grad_one] = calculate_one_gradient(x, y, W, lambda_1)
% input: size(x) = [1, num_feature], size(y) = [1, num_class]
% Calculate hinge loss gradient
    [num_feature, num_class] = size(W);
    Z_m = zeros(num_feature, num_class);
    grad_one = lambda_1 * W;
    
    grad_subset = Z_m;

    grad_subset = calculate_one_gradient_subset_loss(x, y, W);
    
    grad_one = grad_one + grad_subset;

end

function [f_value] = calculate_fValue_subset_loss(X, Y, W)
    [num_instance, num_label] = size(Y);
    f_value = 0;
    for i = 1: num_instance
        tmp_max = 0;
        for j = 1: num_label
            tmp_loss = max(0,1-Y(i,j)*dot(W(:,j), X(i,:)));
            if tmp_loss > tmp_max
                tmp_max = tmp_loss;
            end
        end
        f_value = f_value + tmp_max;
    end
end

function [ W_gradient ] = calculate_one_gradient_subset_loss( x, y, W )
    [n_features, n_labels] = size(W);
    W_gradient = zeros(n_features, n_labels);
    tmp_max = -1;
    index = 1;
    for j = 1: n_labels
        tmp_loss = max(0,1-y(j)*dot(W(:,j), x));
        if tmp_loss > tmp_max
            tmp_max = tmp_loss;
            index = j;
        end
    end
    index = int32(index);
    W_gradient(:,index) = (-y(index)*sign(max(0,1-y(index)*dot(W(:,index),x))))*x';    
end