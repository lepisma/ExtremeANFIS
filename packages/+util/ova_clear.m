function int_out = ova_clear(out)
    % This function gives the output for a one-vs-all system
    % by resolving classification doubts. Assumes number of classifiers
    % equal to the number of classes.
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
    %   (n * cl)
    %   n = number of observations
    %   cl = number of classes
    
    [n, c] = size(out);
    
    int_out = zeros(n, c);
    dist_from_thresh = out - (0.5 * ones(n, c));
    
    for obs = 1:n
        [max_val, max_idx] = max(dist_from_thresh(obs, :));
        if max_val > 0
            int_out(obs, max_idx) = 1;
        end
    end
end