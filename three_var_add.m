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

fisses = addfis(data_train, n_mfs, 2, data_test);
[e_fis, e_errs] = exanfis(data_train, n_mfs, 10, data_test);

[add_err, a_errs] = addfis_err(fisses, data_test);

e_errs
ex_err = min(e_errs)
a_errs
add_err