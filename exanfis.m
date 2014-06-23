function accu = exanfis(x, y, n_mfs = 5)
    
    [n_observations, n_variables] = size(x);
    
    % Initializing gbell membership parameters
    mf_params = zeros(n_mfs * n_variables, 3);
    
    % Setting `b` in the range of 1.9 to 2.1
    mf_params(:, 2) = 2.1 - (0.2 * rand(n_mfs * n_variables, 1));
    
    for i = 1:n_variables
        range = range(x(:, i))
        
        % Setting `a` in range of (0.5 * tmp_a) to (1.5 * tmp_a))
        tmp_a = range / (2 * (n_mfs - 1));
        mf_params((i - 1) * n_mfs + 1 :(i * n_mfs), 1) = tmp_a * (1.5 - rand(n_mfs, 1));
        clear tmp_a;
        
        % Setting `c`
        tmp_c = linspace(min(x(:, i)), max(x(:, i)), n_mfs);
        diff_c = (max(x(:, i)) - min(x(:, i))) / (n_mfs - 1);
        
        mf_params((i - 1) * n_mfs + 1, 3) = tmp_c(1) + (diff_c / 2) * rand();
        mf_params((i - 1) * n_mfs + 2 :(i * n_mfs) - 1, 3) = tmp_c(2: end - 1) + (diff_c / 2) * (1 - 2 * rand());
        mf_params((i * n_mfs), 3) = tmp_c(end) - (diff_c / 2) * rand();
    end
    
    return(0);
end