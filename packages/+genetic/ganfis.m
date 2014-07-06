function [fis, global_error_list] = ganfis(data_train, n_mfs, data_test)
    
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
    opt_in_params = -1;
    opt_out_params = -1;
    least_error = -1;
    
    epochs = 20;
    n_gen = 5;
    
    % List of errors for each epoch for debugging
    global_error_list = zeros(1, epochs * n_gen);
    gen_input_params = cell(1, n_gen);
    gen_error_list = zeros(1, n_gen);
    
    for e = 1 : epochs
        
        if e == 1
            
            for p = 1 : n_gen
                
                gen_input_params{p} = gen_random_mf(x_train, var_ranges, n_variables, n_mfs);
            
            end
        
        end
        
        for p = 1 : n_gen
           
            % Inserting the values of random parameters
            for j = 1 : n_variables
                
                for k = 1 : n_mfs
                    
                    fis.input(j).mf(k).params = gen_input_params{p}((j - 1) * n_mfs + k, :);
                
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
            current_error = rmse(fis, data_test);
            global_error_list((e - 1) * n_gen + p) = current_error;
            gen_error_list(p) = current_error;
            
            if (p == e == 1) || (current_error < least_error)
                
                least_error = current_error;
                opt_in_params = fis.input;
                opt_out_params = fis.output;
            
            end
        
        end
        
        gen_input_params = evolve(gen_input_params, gen_error_list, x_train, var_ranges, n_variables, n_mfs);
        
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

function new_list = evolve(in_list, gen_error_list, x_train, var_ranges, n_variables, n_mfs)
    
    [~, order] = sort(gen_error_list);
    new_list = cell(1, 5);
    new_list{1} = (in_list{order(1)} + in_list{order(2)}) / 2;
    
    for p = 2 : 4
        
        new_list{p} = gen_random_mf(x_train, var_ranges, n_variables, n_mfs);
    
    end
    
    new_list{5} = (in_list{order(end)} + in_list{order(end - 1)}) / 2;

end

function error = rmse(fis, data_test)
    % Calculates the root mean squared error for a test data
    % Faster than method in rmse.m

    output = evalfismex(data_test(:, 1 : end - 1), fis, 101);
    error = rms(output - data_test(:, end));

end
