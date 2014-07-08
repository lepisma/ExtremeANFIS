function error = error(fisses, data_test)
% Calculates the percent error for addfis

epochs = length(fisses);

% Finding error for
en_output = zeros(size(data_test, 1), 1);

for i = 1 : epochs + 1
    
    en_output = en_output + evalfis(data_test(:, 1 : end - 1), fisses{i});

end

en_output = en_output;

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