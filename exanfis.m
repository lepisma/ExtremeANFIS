function [fis, errlist] = exanfis(data_train, n_mfs, mf_type, epochs, data_test)
    
    %----------------------------------------------------------------------
    % TRAINING
    
    % Generating FIS
    % Last column is output data, while all others are input data
    
    
    x_train = data_train(:, 1 : end - 1);
    y_train = data_train(:, end)';
    
    [n_observations, n_variables] = size(x_train);
    
    fis = genfis1(data_train, n_mfs, mf_type);
    
    n_rules = size(fis.rule, 2);
    H = zeros(n_rules * (n_variables + 1), n_observations);
    
    err = -1;
    
    opt_mf = -1;
    opt_out = -1;
    
    errs = [];
    
    for e = 1:epochs
        % Generating random mf parameters
        mf_params = gen_random_mf(n_mfs, n_variables, x_train);

        % Inserting the values of random parameters in generated fis
        for j = 1 : n_variables
            for k = 1 : n_mfs
                fis.input(j).mf(k).params = mf_params((j - 1) * n_mfs + k, :);
            end
        end
        
         % For each input instance
        for i = 1:n_observations
            % Finding firings
            [~, IRR] = evalfismex(x_train(i, :), fis, 101);
            rule_firings = prod(IRR, 2);

            h_col = (repmat([x_train(i, :) 1], [n_rules, 1]) .* repmat(rule_firings, [1, n_variables + 1]))';
            H(:, i) = h_col(:);
        end
    
        % ELM thing
        P = y_train * pinv(H);

        % Regularized inverse
        %P = (eye(n_rules * (n_variables + 1)) / 1000 + H * H') \ H * y_train';

        P = reshape(P, [n_variables + 1, n_rules])';
        
        % Inserting the values of calculated parameters in generated fis
        for i = 1:n_rules
            fis.output.mf(i).params = P(i, :);
        end
        
        curr_err = rmse(fis, data_test);
        
        errs = [errs, curr_err];
        
        if (err == -1) || (curr_err < err)
            err = curr_err;
            opt_mf = mf_params;
            opt_out = P;
        end
    end
    
    errlist = errs;
    
    % Inserting the values of random parameters in generated fis
    for j = 1 : n_variables
        for k = 1 : n_mfs
            fis.input(j).mf(k).params = opt_mf((j - 1) * n_mfs + k, :);
        end
    end
    
    for i = 1:n_rules
        fis.output.mf(i).params = opt_out(i, :);
    end
    
end

function  mf_params = gen_random_mf(n_mfs, n_variables, x_train)

    total_mfs = n_mfs * n_variables;
    mf_params = zeros(total_mfs, 3);
    
    % Setting parameter `b` in the range of 1.9 to 2.1
    mf_params(:, 2) = 2.1 - (0.2 * rand(total_mfs, 1));
    
    % Settings `a` and `c` for each attribute in dataset
    for i = 1:n_variables
        rg = range(x_train(:, i));
        
        % Setting `a` in range of (0.5 * tmp_a) to (1.5 * tmp_a))
        tmp_a = rg / (2 * (n_mfs - 1));
        mf_params((i - 1) * n_mfs + 1 : (i * n_mfs), 1) = tmp_a * (1.5 - rand(n_mfs, 1));
        
        % Setting `c`
        tmp_c = linspace(min(x_train(:, i)), max(x_train(:, i)), n_mfs);
        diff_c = rg / (n_mfs - 1);
        
        mf_params((i - 1) * n_mfs + 1, 3) = tmp_c(1) + (diff_c / 2) * rand();
        mf_params((i - 1) * n_mfs + 2 : (i * n_mfs) - 1, 3) = tmp_c(2 : end - 1) + (diff_c / 2) * (1 - 2 * rand());
        mf_params((i * n_mfs), 3) = tmp_c(end) - (diff_c / 2) * rand();
    end
end

function error = rmse(fis, data_test)
    % Calculates the root mean squared error for a test data on a given
    % fis

    output = evalfismex(data_test(:, 1 : end - 1), fis, 101);

    error = rms(output - data_test(:, end));
end