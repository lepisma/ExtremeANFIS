function accu = exanfis(x_train, y_train, x_test, y_test, n_mfs)
    
    %----------------------------------------------------------------------
    % TRAINING
    [n_observations, n_variables] = size(x_train);
    
    % Initializing gbell membership parameters.
    % This matrix holds the whole randomised membership functions
    % parameters `a`, `b` and `c`
    % The choice of limits of range of randomisation is made as per the
    % paper "Comparison of Extrem ANFIS and ANFIS for regression problems"
    mf_params = zeros(n_mfs * n_variables, 3);
    
    % Setting parameter `b` in the range of 1.9 to 2.1
    mf_params(:, 2) = 2.1 - (0.2 * rand(n_mfs * n_variables, 1));
    
    % Settings `a` and `c` for each attribute in dataset
    for i = 1:n_variables
        range = range(x_train(:, i));
        
        % Setting `a` in range of (0.5 * tmp_a) to (1.5 * tmp_a))
        tmp_a = range / (2 * (n_mfs - 1));
        mf_params((i - 1) * n_mfs + 1 : (i * n_mfs), 1) = tmp_a * (1.5 - rand(n_mfs, 1));
        
        % Setting `c`
        tmp_c = linspace(min(x_train(:, i)), max(x_train(:, i)), n_mfs);
        diff_c = (max(x_train(:, i)) - min(x_train(:, i))) / (n_mfs - 1);
        
        mf_params((i - 1) * n_mfs + 1, 3) = tmp_c(1) + (diff_c / 2) * rand();
        mf_params((i - 1) * n_mfs + 2 : (i * n_mfs) - 1, 3) = tmp_c(2 : end - 1) + (diff_c / 2) * (1 - 2 * rand());
        mf_params((i * n_mfs), 3) = tmp_c(end) - (diff_c / 2) * rand();
    end
    
    clear tmp_a;
    clear tmp_c;
    clear diff_c;
    
    % Finding firing of all membership functions
    % `firing_strengths` is a matrix with each row representing single
    % observation and values of columns represent firing strengths of all
    % membership functions for all input variables
    firing_strengths = zeros(n_observations, n_mfs * n_variables);
    
    for i = 1:n_observations
        for j = 1:n_variables
            for k = 1:n_mfs
                tmp_fire = gbellmf(x_train(i, j), mf_params((j - 1) * n_mfs + k, :));
            end
            firing_strengths(i, (j - 1) * n_mfs + 1 : (j * n_mfs)) = tmp_fire;
        end
    end
    
    clear tmp_fire;
    
    n_rules = n_mfs ^ n_variables;
    
    % Making rules considering all possible combinations
    % Finding their firing strengths by taking minimum
    rule = n_mfs * (1 : n_variables) - (n_mfs - 1);
    % rule here is a list of elements to be chosen from firing_strengths to
    % form a rule
    rule_add = [zeros(1, n_variables), 1];
    
    count = 1;
    
    rule_firings = zeros(n_observations, n_rules);
    
    for i = 1:n_observations
        for j = 1:n_rules
            rule_firings(i, j) = min(firing_strengths(i, rule));
            if count < n_mfs
                rule = rule + rule_add;
                count = count + 1;
            else
                count = 1;
                rule_add = circshift(rule_add', 1)';
            end
        end
    end
    
    clear firing_strengths;
    
    % Making H matrix
    
    x_new = [x_train', ones(1, n_observations)];
    
    H = zeros(n_rules * (n_variables + 1), n_observations);
    tmp_h = zeros(n_variables + 1, n_observations);
    
    for i = 1:n_rules
        for j = 1:n_observations
            tmp_h(:, j) = x_new(:, j) * rule_firings(j, i);
        end
        
        H((i - 1) * (n_variables + 1) + 1 : i * (n_variables + 1), :) = tmp_h;
    end
    
    clear x_new;
    clear tmp_h;
    clear rule_firings;
    
    % H generated
    % now parameter matrix can be represented as y * pinv(H)
    
    P = y_train * pinv(H);
    
    % Training complete
    
    %----------------------------------------------------------------------
    % TESTING
    
    n_observations_test = size(x_test, 1);
    
    % Finding firing strengths for membership functions using the earlier
    % generated `mf_params`
    firing_strengths_test = zeros(n_observations_test, n_mfs * n_variables);
    
    for i = 1:n_observations_test
        for j = 1:n_variables
            for k = 1:n_mfs
                tmp_fire = gbellmf(x_test(i, j), mf_params((j - 1) * n_mfs + k, :));
            end
            firing_strengths_test(i, (j - 1) * n_mfs + 1 : (j * n_mfs)) = tmp_fire;
        end
    end
    
    clear tmp_fire;
    
    % Find firings for rules
    % Same rules should be here, implying that rules have to be stored in
    % some form.
    
    rule = n_mfs * (1 : n_variables) - (n_mfs - 1);
    rule_add = [zeros(1, n_variables), 1];
    
    count = 1;
    
    rule_firings_test = zeros(n_observations_test, n_rules);
    
    for i = 1:n_observations_test
        for j = 1:n_rules
            rule_firings_test(i, j) = min(firing_strengths_test(i, rule));
            if count < n_mfs
                rule = rule + rule_add;
                count = count + 1;
            else
                count = 0;
                rule_add = circshift(rule_add', 1)';
            end
        end
    end
    
    clear count;
    clear firing_strengths_test;
    clear rule;
    clear rule_add;
    
    % Rule firings found
    % Finding H
    
    x_new_test = [x_test', ones(1, n_observations_test)];
    
    H = zeros(n_rules * (n_variables_test + 1), n_observations_test);
    tmp_h = zeros(n_variables_test + 1, n_observations_test);
    
    for i = 1:n_rules
        for j = 1:n_observations_test
            tmp_h(:, j) = x_new_test(:, j) * rule_firings_test(j, i);
        end
        
        H((i - 1) * (n_variables_test + 1) + 1 : i * (n_variables_test + 1), :) = tmp_h;
    end
    
    clear rule_firings_test;
    clear tmp_h;
    clear x_new_test;
    
    % H found, finding output
    
    output = P * H;
    
    % Checking for accuracy
    accu = mean(output == y_test);
end