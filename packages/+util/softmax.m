function int_out = softmax(out)
    % Returns int (and resolved) output for multiclass classifier
    % with number of classifiers equal to that of class
    %
    % Parameters
    % ----------
    % `out`
    %   the output from the system of fisses
    %   (n * c)
    %   n = number of observations
    %   c = number of classifiers
    %
    % Returns
    % -------
    % `int_out`
    %   the final (and resolved) output in 0, 1 form.
    %   (n * c)
    %   n = number of observations
    %   c = number of classes
    
    [n, c] = size(out);
    exp_out = exp(out);
    
    % prob_out is the confidence indicator
    prob_out = diag(1 ./ sum(exp_out, 2)) * exp_out;
    
    int_out = zeros(n, c);
    
    for obs = 1:n
        [max_val, max_idx] = max(prob_out(obs, :));
        if max_val > 0.5
            int_out(obs, max_idx) = 1;
        end
    end
    
end