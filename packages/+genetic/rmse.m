function err = rmse(fis, data_test)
% Returns the root mean square error for genetic anfis

out = evalfis(data_test(:, 1 : end - 1), fis);

err = rms(out - data_test(:, end));

end