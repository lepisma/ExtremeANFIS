function [test, train, n_test, n_train] = test_train_split(data, ratio)
    % Splits the data in the given ratio
    % Extreme values goes in train data
    %
    % Parameters
    % ----------
    % `data`
    %   The data assuming last column containing class number
    % `ratio`
    %   The ratio of test data to train data
    %
    % Returns
    % -------
    % `test`
    %   The testing data
    % `train`
    %   The training data
    % `n_test`
    %   List containing number of test instances per class
    % `n_train`
    %   List containing number of train instances per class
    
    [extremes, data] = util.remove_extremes(data);
    
    n_classes = size(unique(data(:, end)), 1);
    
    n_instances = zeros(n_classes, 1); % Number of instances per class
    
    for i = 1:n_classes
        n_instances(i) = sum(data(:, end) == i);
    end
    
    n_test = round(ratio * n_instances); % Number of test instances per class
    
    for i = 1:n_classes
        class_data = data(data(:, end) == i, :);
        
        if i == 1
            test = class_data(1:n_test(i), :);
            train = class_data(n_test(i)+1:end, :);
        else
            test = [test; class_data(1:n_test(i), :)];
            train = [train; class_data(n_test(i)+1:end, :)];
        end
    end
    
    train = [train; extremes];
    
    %  Finding number of train instances after adding extremes
    n_train = zeros(n_classes, 1);
    
    for i = 1:n_classes
        n_train(i) = sum(train(:, end) == i);
    end
end