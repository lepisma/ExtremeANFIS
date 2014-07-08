function error = error(fis, data_test)
% Calculates the percent error for exanfis

en_output = evalfis(data_test(:, 1 : end - 1), fis);

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