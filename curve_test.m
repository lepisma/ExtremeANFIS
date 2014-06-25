% This is a test code for testing simple curve generation
% Here, anfis is faster :(
% Also, most of the time in exanfis function is taken while running evalfis

n_pts = 51;
x = linspace(-1, 1, n_pts)';
y = 0.6 * sin(pi * x) + 0.3 * sin(3 * pi * x) + 0.1 * sin(5 * pi * x);
data = [x y];
data_train = data(1 : 2 : n_pts, :);
data_test = data(2 : 3 : n_pts, :);

figure(1)
plot(data_train(:, 1), data_train(:, 2), 'o', data_test(:, 1), data_test(:, 2), 'x')

n_mfs = 5000;

a_fis = anfis(data_train, n_mfs);

e_fis = exanfis(data_train(:, 1), data_train(:, 2)', n_mfs);

figure(2)
plot(data_test(:, 1), evalfis(data_test(:, 1), a_fis));

figure(3)
plot(data_test(:, 1), evalfis(data_test(:, 1), e_fis));