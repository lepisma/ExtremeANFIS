% A function with three variables

x = 1 : 6;
x = repmat(x, 36, 1);
x = x(:);

y = 1 : 6;
y = repmat(y, 6, 1);
y = y(:);
y = repmat(y, 6, 1);

z = (1 : 6)';
z = repmat(z, 36, 1);

o = (1 + x .^ 0.5 + y .^ (-1) + z .^ (-1.5)).^ 2;
data_train = [x y z o];

%----

x = 1.5 : 5.5;
x = repmat(x, 25, 1);
x = x(:);

y = 1.5 : 5.5;
y = repmat(y, 5, 1);
y = y(:);
y = repmat(y, 5, 1);

z = (1.5 : 5.5)';
z = repmat(z, 25, 1);

o = (1 + x .^ 0.5 + y .^ (-1) + z .^ (-1.5)).^ 2;
data_test = [x y z o];

% Data generation done

n_mfs = 4;

[g_fis, g_errs] = ganfis(data_train, n_mfs, data_test);
[e_fis, e_errs] = exanfis(data_train, n_mfs, 100, data_test);
% Errors for epochs
g_mean = mean(g_errs)
e_mean = mean(e_errs)
g_std = std(g_errs)
e_std = std(e_errs)

% Train errors
g_err_train = rmse(g_fis, data_train)
e_err_train = rmse(e_fis, data_train)

% Test errors
g_err_test = rmse(g_fis, data_test)
e_err_test = rmse(e_fis, data_test)