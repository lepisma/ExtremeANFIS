function error = rmse(fis, data_test)
% Calculates the root mean squared error for a test data on a given
% fis

output = evalfis(data_test(:, 1 : end - 1), fis);

error = rms(output - data_test(:, end));
end