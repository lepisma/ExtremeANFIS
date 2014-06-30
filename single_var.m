% Function with single variable

n_pts = 501;
x = linspace(-1, 1, n_pts)';
y = 0.6 * sin(pi * x) + 0.3 * sin(3 * pi * x) + 0.1 * sin(5 * pi * x);
data = [x y];
data_train = data(1 : 2 : n_pts, :);
data_test = data(2 : 2 : n_pts, :);

figure(1)
plot(data_train(:, 1), data_train(:, 2), 'o', data_test(:, 1), data_test(:, 2), 'x')
title('Main data');

n_mfs = 10;

a_fis = anfis(data_train, n_mfs, 10);
a_err = rmse(a_fis, data_test);

e_fis = exanfis(data_train, n_mfs, 10, data_test);
e_err = rmse(e_fis, data_test);

% Plots
% Test error as title of plots
figure(2)
plot(data_test(:, 1), evalfis(data_test(:, 1), a_fis));
title(a_err);

figure(3)
plot(data_test(:, 1), evalfis(data_test(:, 1), e_fis));
title(e_err);