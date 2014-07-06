function [error, err_list] = addfis_err(fisses, data_test)
% Calculates the percent error

epochs = length(fisses);

err_list = zeros(1, epochs);

for i = 1 : epochs
    err_list(i) = fis_error(fisses{i}, data_test);
end

add_out = zeros(size(data_test, 1), 1);

for i = 1 : epochs
    add_out = add_out + evalfis(data_test(:, 1 : end - 1), ...
                                fisses{i});
end

error = rms(add_out - data_test(:, end));
end

function error = fis_error(fis, data_test)
    output = evalfis(data_test(:, 1 : end - 1), fis);
    
    error = rms(output - data_test(:, end));
end