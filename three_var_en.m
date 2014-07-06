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

fisses = enxanfis(data_train, n_mfs, 100, 80);
[e_fis, e_errs] = exanfis(data_train, n_mfs, 100, data_test);

[enx_err, err_list] = en_err(fisses, data_test);


ex_err = min(e_errs)
enx_err