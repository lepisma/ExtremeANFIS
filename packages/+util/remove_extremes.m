function [extremes, others] = remove_extremes(data)
    % This function removes the rows containing extremes of each columns.
    % This is done to provide complete range of inputs to genfis1
    % Also an assumption is made that data(:, end) is the output
    %
    % Parameters
    % ----------
    % `data`
    %   The data matrix
    % 
    % Returns
    % -------
    % `extremes`
    %   The extreme rows
    % `others`
    %   Data except extremes
    
    [~, min_idx] = min(data(:, 1:end-1));
    [~, max_idx] = max(data(:, 1:end-1));
    
    extremes_indices = unique([min_idx, max_idx]);
    extremes = data(extremes_indices, :);
    
    others_indices = setdiff(1:size(data, 1), extremes_indices);
    others = data(others_indices, :);
end