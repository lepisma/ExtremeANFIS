function accuracy = fis_accuracy(fis, x_test, y_test)

    % Calculates the accuracy for any matlab fis, given the fis and the
    % testing data

    output = evalfis(x_test, fis);
    
    accuracy = mean(output == y_test');
    
end