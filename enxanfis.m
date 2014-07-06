function fisses = enxanfis(data_train, n_mfs, epochs, n_sample)
    % Ensemble of extreme anfis
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
    % `fisses`
    %   the fuzzy inference systems tuned using Extreme ANFIS
    
    total_x = data_train(:, 1 : end - 1);
    
    [~, n_variables] = size(total_x);
    var_ranges = range(total_x);
    
    fisses = cell(1, epochs + 1);
    
    % Generating fuzzy inference systems
    for e = 1 : epochs + 1
        fisses{e} = genfis1(data_train, n_mfs, 'gbellmf');
    end
    
    n_rules = size(fisses{1}.rule, 2);
    
    rule_mat = zeros(n_rules, n_sample);
    
    for e = 1 : epochs + 1
        if e ~= 1
            % Random guesses
            input_params = gen_random_mf(total_x, var_ranges, n_variables, n_mfs);
            
            % Inserting the values of random parameters
            for j = 1 : n_variables
                for k = 1 : n_mfs
                    fisses{e}.input(j).mf(k).params = input_params((j - 1) * n_mfs + k, :);
                end
            end
        end
        
        % Sampling data for each fis
        indices = round((size(data_train, 1) - 1) * rand(1, n_sample)) + 1;
        train_ep = data_train(indices, :);
        
        % Finding output parameters
        for i = 1:n_sample
            % Finding firings
            [~, IRR] = evalfismex(train_ep(i, 1 : end - 1), fisses{e}, 101);
            rule_mat(:, i) = prod(IRR, 2);
        end

        % getting normalised weights
        rule_mat = bsxfun(@rdivide, rule_mat, sum(rule_mat));
        
        x_for_h = repmat([train_ep(:, 1 : end - 1) ones(n_sample, 1)]', n_rules, 1);
        
        % Inverse using Moore - Penrose psuedo inverse
        H = x_for_h .* rule_mat(repmat(1 : n_rules, n_variables + 1, 1), :);
        %P = y_train * pinv(H);

        % Inverse with regularization
        P = (eye(n_rules * (n_variables + 1)) / 10000 + H * H') \ H * train_ep(:, end);

        output_params = reshape(P, [n_variables + 1, n_rules])';
        
        % Inserting the values of output parameters
        for i = 1:n_rules
            fisses{e}.output.mf(i).params = output_params(i, :);
        end
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
%         tmp_c = linspace(min(x_train(:, i)), max(x_train(:, i)), n_mfs);
%         diff_c = var_ranges(i) / (n_mfs - 1);
%         
%         mf_params((i - 1) * n_mfs + 1, 3) = tmp_c(1) + (diff_c / 2) * rand();
%         mf_params((i - 1) * n_mfs + 2 : (i * n_mfs) - 1, 3) = tmp_c(2 : end - 1) + (diff_c / 2) * (1 - 2 * rand());
%         mf_params((i * n_mfs), 3) = tmp_c(end) - (diff_c / 2) * rand();
        mf_params((i - 1) * n_mfs + 1 : (i * n_mfs), 3) = max(x_train(:, i)) - var_ranges(i) * rand(n_mfs, 1);
    end
end
