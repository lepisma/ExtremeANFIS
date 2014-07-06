function [fis, errlist] = exanfis(data_train, n_mfs, epochs, data_test)
    % Extreme ANFIS
    %
    % Parameters
    % ----------
    % `data_train`
    %   data for training, last column represents output
    % `n_mfs`
    %   number of membership functions for each input attribute
    % `epochs`
    %   number of times random input parameters should be generated
    % `data_test`
    %   data for testing, used for finding the best input parameter set
    %
    % Returns
    % -------
    % `fis`
    %   the fuzzy inference system tuned using Extreme ANFIS
    % `errlist`
    %   the list of error for each epoch (for debugging purpose only)
    
    x_train = data_train(:, 1 : end - 1);
    y_train = data_train(:, end)';
    
    [n_observations, n_variables] = size(x_train);
    var_ranges = range(x_train);
    
    % Generating a fuzzy inference system with uniform mfs and no output
    fis = genfis1(data_train, n_mfs, 'gbellmf');
    n_rules = size(fis.rule, 2);
    
    x_for_h = repmat([x_train ones(n_observations, 1)]', n_rules, 1);
    rule_mat = zeros(n_rules, n_observations);
    
    % Optimum input output parameters and least error
    opt_in_params = fis.input;
    opt_out_params = -1;
    least_err = -1;
    
    % List of errors for each epoch (+1 for uniform mfs) for debugging
    errlist = zeros(1, epochs + 1);
    
    for e = 1 : epochs + 1
        if e ~= 1
            % Random guesses
            input_params = gen_random_mf(x_train, var_ranges, n_variables, n_mfs);
            
            % Inserting the values of random parameters
            for j = 1 : n_variables
                for k = 1 : n_mfs
                    fis.input(j).mf(k).params = input_params((j - 1) * n_mfs + k, :);
                end
            end
        end
        
        % Finding output parameters
        for i = 1:n_observations
            % Finding firings
            [~, IRR] = evalfismex(x_train(i, :), fis, 101);
            rule_mat(:, i) = prod(IRR, 2);
        end

        % getting normalised weights
        rule_mat = bsxfun(@rdivide, rule_mat, sum(rule_mat));

        % Inverse using Moore - Penrose psuedo inverse
        H = x_for_h .* rule_mat(repmat(1 : n_rules, n_variables + 1, 1), :);
        %P = y_train * pinv(H);

        % Inverse with regularization
        P = (eye(n_rules * (n_variables + 1)) / 10000 + H * H') \ H * y_train';

        output_params = reshape(P, [n_variables + 1, n_rules])';
        
        % Inserting the values of output parameters
        for i = 1:n_rules
            fis.output.mf(i).params = output_params(i, :);
        end
        
        % Finding error
        current_err = rmse(fis, data_test);
        
        % Appending error to list
        errlist(e) = current_err;
        
        % If error is less than least error, then update optimum values
        if e == 1
            opt_out_params = fis.output;
            least_err = current_err;
        end
        
        if current_err < least_err
            least_err = current_err;
            opt_in_params = fis.input;
            opt_out_params = fis.output;
        end
    end
    
    % Inserting best parameters
    fis.input = opt_in_params;
    fis.output = opt_out_params;
    
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

function error = rmse(fis, data_test)
    % Calculates the root mean squared error for a test data
    % Faster than method in rmse.m

    output = evalfismex(data_test(:, 1 : end - 1), fis, 101);
    error = rms(output - data_test(:, end));
end
