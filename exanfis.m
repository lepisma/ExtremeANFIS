function [fis, errlist] = exanfis(data_train, n_mfs, mf_type, epochs, data_test)
    
    %----------------------------------------------------------------------
    % TRAINING
    
    % Generating FIS
    % Last column is output data, while all others are input data
    
    x_train = data_train(:, 1 : end - 1);
    y_train = data_train(:, end)';
    
    [n_observations, n_variables] = size(x_train);
    var_ranges = range(x_train);
    
    % Fis with uniform mfs and no output
    fis = genfis1(data_train, n_mfs, mf_type);
    
    n_rules = size(fis.rule, 2);
    
    x_for_h = repmat([x_train ones(n_observations, 1)]', n_rules, 1);
    rule_mat = zeros(n_rules, n_observations);
    
    % Generating output for uniform mfs
    uniform_output_params = gen_output_params(fis, x_train, x_for_h, rule_mat, y_train, n_observations, n_variables, n_rules);
    
    % Inserting output parameters in fis
    for i = 1:n_rules
        fis.output.mf(i).params = uniform_output_params(i, :);
    end
    
    % Finding rmse for uniform mfs
    uniform_err = rmse(fis, data_test);
    
    % Flag declaring that current least error is from uniform mfs
    uniform_flag = 1;
    
    % Optimum input output parameters and least error
    opt_mf = fis.input;
    opt_out = uniform_output_params;
    least_err = uniform_err;
    
    % List of errors for each epoch (for debugging)
    errlist = zeros(1, epochs + 1);
    
    % First element is of uniform error
    errlist(1) = uniform_err;
    
    for e = 1:epochs
        % Random guesses
        % Generating random mf parameters
        mf_params = gen_random_mf(x_train, var_ranges, n_variables, n_mfs);

        % Inserting the values of random parameters in generated fis
        for j = 1 : n_variables
            for k = 1 : n_mfs
                fis.input(j).mf(k).params = mf_params((j - 1) * n_mfs + k, :);
            end
        end
        
        % Finding output parameters
        output_params = gen_output_params(fis, x_train, x_for_h, rule_mat, y_train, n_observations, n_variables, n_rules);
        
        % Inserting the values of calculated parameters in generated fis
        for i = 1:n_rules
            fis.output.mf(i).params = output_params(i, :);
        end
        
        % Finding error
        curr_err = rmse(fis, data_test);
        
        % Appending error to list
        errlist(e + 1) = curr_err;
        
        % If error is less than least error, then update optimum values
        if curr_err < least_err
            least_err = curr_err;
            opt_mf = mf_params;
            opt_out = output_params;
            
            % Set the flag to 0
            uniform_flag = 0;
        end
    end
    
    % If the flag is 0, then uniform mfs are not the answer
    % Need to push optimum input parameters in fis
    if uniform_flag == 0
        % Inserting the values of random parameters in generated fis
        for j = 1 : n_variables
            for k = 1 : n_mfs
                fis.input(j).mf(k).params = opt_mf((j - 1) * n_mfs + k, :);
            end
        end
    else
        % Set the input parameters using stored value
        fis.input = opt_mf;
    end

    % Setting the optimum output parameters
    for i = 1:n_rules
        fis.output.mf(i).params = opt_out(i, :);
    end
    
end

function mf_params = gen_random_mf(x_train, var_ranges, n_variables, n_mfs)
    % Returns randomly generated mf parameters
    
    total_mfs = n_mfs * n_variables;
    mf_params = zeros(total_mfs, 3);
    
    % Setting parameter `b` in the range of 1.9 to 2.1
    mf_params(:, 2) = 2.1 - (0.2 * rand(total_mfs, 1));
    
    % Settings `a` and `c` for each attribute in dataset
    for i = 1:n_variables
        
        % Setting `a` in range of (0.5 * tmp_a) to (1.5 * tmp_a))
        tmp_a = var_ranges(i) / (2 * (n_mfs - 1));
        mf_params((i - 1) * n_mfs + 1 : (i * n_mfs), 1) = tmp_a * (1.5 - rand(n_mfs, 1));
        
        % Setting `c`
        tmp_c = linspace(min(x_train(:, i)), max(x_train(:, i)), n_mfs);
        diff_c = var_ranges(i) / (n_mfs - 1);
        
        mf_params((i - 1) * n_mfs + 1, 3) = tmp_c(1) + (diff_c / 2) * rand();
        mf_params((i - 1) * n_mfs + 2 : (i * n_mfs) - 1, 3) = tmp_c(2 : end - 1) + (diff_c / 2) * (1 - 2 * rand());
        mf_params((i * n_mfs), 3) = tmp_c(end) - (diff_c / 2) * rand();
    end
end

function output_params = gen_output_params(fis, x_train, x_for_h, rule_mat, y_train, n_observations, n_variables, n_rules)
    % Returns output parameters using elm method

    % For each input instance
    for i = 1:n_observations
        % Finding firings
        [~, IRR] = evalfismex(x_train(i, :), fis, 101);
        rule_mat(:, i) = prod(IRR, 2);
    end
    
    % getting normalised weights
    sum_col = repmat(sum(rule_mat),n_rules,1);
    norm_rule_mat = rule_mat ./ sum_col;

    % ELM thing
    P = y_train * pinv(x_for_h .* norm_rule_mat(repmat(1 : n_rules, n_variables + 1, 1), :));

    % Regularized inverse
    %P = (eye(n_rules * (n_variables + 1)) / 10000 + H * H') \ H * y_train';

    output_params = reshape(P, [n_variables + 1, n_rules])';
end

function error = rmse(fis, data_test)
    % Calculates the root mean squared error for a test data
    % Faster than method in rmse.m

    output = evalfismex(data_test(:, 1 : end - 1), fis, 101);

    error = rms(output - data_test(:, end));
end
