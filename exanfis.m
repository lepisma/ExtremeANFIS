function accu = exanfis(x_train, y_train, x_test, y_test, n_mfs = 5)
    
    [n_observations, n_variables] = size(x_train);
    
    % Initializing gbell membership parameters
    mf_params = zeros(n_mfs * n_variables, 3);
    
    % Setting `b` in the range of 1.9 to 2.1
    mf_params(:, 2) = 2.1 - (0.2 * rand(n_mfs * n_variables, 1));
    
    for i = 1:n_variables
        range = range(x_train(:, i));
        
        % Setting `a` in range of (0.5 * tmp_a) to (1.5 * tmp_a))
        tmp_a = range / (2 * (n_mfs - 1));
        mf_params((i - 1) * n_mfs + 1 : (i * n_mfs), 1) = tmp_a * (1.5 - rand(n_mfs, 1));
        clear tmp_a;
        
        % Setting `c`
        tmp_c = linspace(min(x_train(:, i)), max(x_train(:, i)), n_mfs);
        diff_c = (max(x_train(:, i)) - min(x_train(:, i))) / (n_mfs - 1);
        
        mf_params((i - 1) * n_mfs + 1, 3) = tmp_c(1) + (diff_c / 2) * rand();
        mf_params((i - 1) * n_mfs + 2 : (i * n_mfs) - 1, 3) = tmp_c(2 : end - 1) + (diff_c / 2) * (1 - 2 * rand());
        mf_params((i * n_mfs), 3) = tmp_c(end) - (diff_c / 2) * rand();
    end
    
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
    
    n_rules = n_mfs ^ n_variables;
    
    % Making rules considering all possibilities
    % CODE HERE
    % Assuming firing for rules are calculated and are stored as
    % rule_firings = zeros(n_observations, n_rules);
    
    x_new = [x_train', ones(1, n_observations)];
    
    H = zeros(n_rules * (n_variables + 1), n_observations);
    tmp_h = zeros(n_variables + 1, n_observations);
    
    for i = 1:n_rules
        for j = 1:n_observations
            tmp_h(:, j) = x_new(:, j) * rule_firings(j, i);
        end
        
        H((i - 1) * (n_variables + 1) + 1 : i * (n_variables + 1), :) = tmp_h;
    end
    
    % H generated
    % now parameter matrix can be represented as y * pinv(H)
    
    P = y_train * pinv(H);
    
    % Training complete
    % Now preparing to test
    
    return;
end