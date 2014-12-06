function int_out = ova_clear(out)
    % This function gives the output for a one-vs-all system
    % by resolving classification doubts. Assumes number of classifiers
    % equal to (cl - 1), where cl is the number of classes.
    % 
    % Parameters
    % ----------
    % `out`
    %   the output from the system of fisses
    %   (n * c)
    %   n = number of observations
    %   c = number of classifiers = (cl - 1)
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
    dist_from_one = abs(ones(n, c) - out); % Confidence indicator
    
    for obs = 1:n
        [min_val, min_idx] = min(dist_from_one(obs, :));
        if min_val < 0.5
            int_out(obs, min_idx) = 1;
        end
    end
    
    int_out = [int_out, ~sum(int_out, 2)];
end