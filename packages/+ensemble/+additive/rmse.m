function err = rmse(fisses, data_test)
% Returns the root mean square error on the additive ensemble

add_out = zeros(size(data_test, 1), 1);

for i = 1 : 2
    
    add_out = add_out + evalfis(data_test(:, 1 : end - 1), fisses{i});
    
end

err = rms(add_out - data_test(:, end));

end