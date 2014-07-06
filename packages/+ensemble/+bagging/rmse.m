function [error, err_list] = rmse(fisses, data_test)
% Calculates the rms error for bagged Extreme ANFIS using
% weighted average

epochs = length(fisses);

err_list = zeros(1, epochs);

for i = 1 : epochs
    
    err_list(i) = fis_error(fisses{i}, data_test);

end

% Finding error for
en_out = zeros(size(data_test, 1), 1);
err_max = max(err_list);
norm_err = err_list / err_max;
norm_err = 1 - norm_err;


for i = 1 : epochs
    
    en_out = en_out + norm_err(i) * evalfis(data_test(:, 1 : end - 1), fisses{i});

end

en_out = en_out / sum(norm_err);

error = rms(en_out - data_test(:, end));

end

function error = fis_error(fis, data_test)
   
    output = evalfis(data_test(:, 1 : end - 1), fis); 
    error = rms(output - data_test(:, end));

end