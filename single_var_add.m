% Function with single variable

n_pts = 501;
x = linspace(-1, 1, n_pts)';
y = 0.6 * sin(pi * x) + 0.3 * sin(3 * pi * x) + 0.1 * sin(5 * pi * x);
data = [x y];
data_train = data(1 : 2 : n_pts, :);
data_test = data(2 : 2 : n_pts, :);

n_mfs = 10;

fisses = addfis(data_train, n_mfs, 2, data_test);
[e_fis, e_errs] = exanfis(data_train, n_mfs, 2, data_test);

add_err = addfis_err(fisses, data_test);


ex_err = min(e_errs)
add_err