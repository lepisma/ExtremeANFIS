function error = rmse2(fisses, weights, data_test)
% Calculates the rms error for bagged Extreme ANFIS by evaulating
% them using weights provided

n_fis = size(fisses);
n_observations = size(data_test, 1);

output_bags = zeros(n_observations, n_fis);

for i = 1 : n_fis

    output_bags(:, i) = evalfis(data_test(:, 1 : end - 1), fisses{i});

end

output = output_bags * weights;

error = rms(output - data_test(:, end));

end