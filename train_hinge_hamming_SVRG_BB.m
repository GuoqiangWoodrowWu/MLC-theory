function [W, obj] = train_hinge_hamming_SVRG_BB( X, Y, lambda_1, alpha )
%   Optimize hamming loss with its surrogate loss, base loss function: hinge loss
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
    
    num_s = 30;
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
    
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    temp1 = max(Z, I - Y .* (X * W));
    f_value_point = sum(sum(temp1, 2));
    f_value_point = f_value_point / num_class;
    
    f_value = f_value + 1 / num_instance * f_value_point;
end


function [grad] = calculate_all_gradient(X, Y, W, lambda_1)
    [num_instance, num_class] = size(Y);
    num_feature = size(X, 2);

    grad = lambda_1 * W;
    Z_m = zeros(num_feature, num_class);
    
    grad_point = Z_m;
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    grad_point = X' * (-Y .* sign(max(Z, I - Y .* (X * W))));
    grad_point = grad_point / num_class;
    
    grad = grad + grad_point / num_instance;
end

function [grad_one] = calculate_one_gradient(x, y, W, lambda_1)
% input: size(x) = [1, num_feature], size(y) = [1, num_class]
% Calculate hinge loss gradient
    [num_feature, num_class] = size(W);
    Z_m = zeros(num_feature, num_class);
    grad_one = lambda_1 * W;
    
    grad_point = Z_m;

    I = ones(1, num_class);
    Z = zeros(1, num_class);
    grad_point = x' * (-y .* sign(max(Z, I - y .* (x * W))));
    % add 
    grad_point = grad_point / num_class;

    grad_one = grad_one + grad_point;

end