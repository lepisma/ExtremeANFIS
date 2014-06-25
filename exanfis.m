function fis = exanfis(x_train, y_train, n_mfs)
    
    %----------------------------------------------------------------------
    % TRAINING
    
    % Generating FIS
    % `genfis1` needs data so that the last column in matrix is the output
    % while all columns before are inputs.
    % So we need to merge x and y
    [n_observations, n_variables] = size(x_train);
    
    data = [x_train, y_train'];
    mf_type = 'gbellmf';
    
    fis = genfis1(data, n_mfs, mf_type);
    
    % Now we have all rules made, and membership functions uniformly
    % distributed.
    % We have to think about randomising membership functions, but they are
    % not a problem as of now.
    % Currently, all the consequence parameters are set to 0. So `evalfis`
    % at any input will give 0
    % But we are not interested in that, `evalfis` also returns a value of
    % firing strengths of all the rules. We are going to find them for each
    % input instance and then calculate the `H` matrix as in `exanfis`
    % function.
    
    n_rules = size(fis.rule, 2);
    H = zeros(n_rules * (n_variables + 1), n_observations);
    
    % For each input instance
    for i = 1:n_observations
        % Finding firings
        [~, IRR] = evalfismex(x_train(i, :), fis, 101);
        rule_firings = min(IRR, [], 2);
        
        tmp_in = repmat([x_train(i, :) 1], [n_rules, 1]);
        tmp_wt = repmat(rule_firings, [1, n_variables + 1]);
        
        h_col = tmp_in .* tmp_wt;
        h_col = h_col';
        
        H(:, i) = h_col(:);
    end
    
    % ELM thing
    P = y_train * pinv(H);
    P = reshape(P, [n_variables + 1, n_rules])';
    
    % Inserting the values of calculated parameters in generated fis
    for i = 1:n_rules
        fis.output.mf(i).params = P(i, :);
    end
    
end