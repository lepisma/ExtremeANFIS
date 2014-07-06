function [error, err_list] = cent_err(fisses, data_test)
% Calculates the percent error

epochs = length(fisses) - 1;

err_list = zeros(1, epochs + 1);

for i = 1 : epochs + 1
    err_list(i) = fis_error(fisses{i}, data_test);
end

% Finding error for
en_output = zeros(size(data_test, 1), 1);
err_max = max(err_list);
norm_err = err_list / err_max;
norm_err = 1 - norm_err;

for i = 1 : epochs + 1
    en_output = en_output + evalfis(data_test(:, 1 : end - 1), fisses{i});
end

en_output = en_output / (epochs + 1);

for i = 1 : size(data_test, 1)
    if en_output(i) >= 0.5
        en_output(i) = 1;
    else
        en_output(i) = 0;
    end
end

incorrect = sum(en_output ~= data_test(:, end));
error = incorrect / size(data_test, 1);
end

function error = fis_error(fis, data_test)
    output = evalfis(data_test(:, 1 : end - 1), fis);
    
    for i = 1 : size(data_test, 1)
        if output(i) >= 0.5
            output(i) = 1;
        else
            output(i) = 0;
        end
    end
    
    incorrect = sum(output ~= data_test(:, end));
    error = incorrect / size(data_test, 1);
end