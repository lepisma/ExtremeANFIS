% Function with single variable

n_pts = 501;
x = linspace(-1, 1, n_pts)';
y = 0.6 * sin(pi * x) + 0.3 * sin(3 * pi * x) + 0.1 * sin(5 * pi * x);
data = [x y];
data_train = data(1 : 2 : n_pts, :);
data_test = data(2 : 2 : n_pts, :);

n_mfs = 10;

fisses = enxanfis(data_train, n_mfs, 5, 300);
[enx_err, err_list] = en_err(fisses, data_test);

e_fis = exanfis(data_train, n_mfs, 5, data_test);
e_err = rmse(e_fis, data_test);

e_err

enx_err