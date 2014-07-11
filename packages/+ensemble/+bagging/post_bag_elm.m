function weights = post_bag_elm(fisses, data_train)

% Finds the weights to be used for finding final output

n_fis = length(fisses);
weights = zeros(1, n_fis);

n_observations = size(data_train, 1);

output_matrix = zeros(n_observations, n_fis);

for i = 1 : n_fis
    
    output_matrix(:, i) = evalfismex(data_train(:, 1 : end - 1), fisses{i}, 101);
    
end

weights = pinv(output_matrix) * data_train(:, end);

end