function my_elm = my_elm(data_file, n_hid_neurons, train2test_ratio, epoch)

% t = target
% last column of data file is assumed to be the target

data = load(data_file);


n_var = size(data,2)-1;
n_obs = size(data,1);
% train2test = .5;
train_indices = randsample(1:n_obs, floor(train2test_ratio * n_obs));

data_train = data(train_indices,:);
data_test = setdiff(data,data_train,'rows');

x_train = data_train(:,1:end-1);
t_train = data_train(:,end);

x_test = data_test(:,1:end-1);
t_test = data_test(:,end);

% Training
err_train = zeros(1,epoch);
w_train = [];
t = cputime;
for e = 1 : epoch
    
    x = [x_train ones(size(x_train,1),1)];
    w = rand(n_var, n_hid_neurons)*2 - 1;
    bias = rand(1,n_hid_neurons);
    w = [w ; bias];
    tempH = x * w;
    % Sigmoid fn.
    H = 1 ./ 1 + exp(-tempH);

% Used this to remove NaN and inf, but no good

%    for e = 1:size(H,1)
%        i = 1;
%       for f = 1:size(H,2)
%           if H(i,f) == NaN || H(i,f) == inf
%                H(i,:) = [];
%               continue
%           else
%               i = i+1;
%           end
%       end
%   end
    B = pinv(H) * t_train;
    y_train = H * B;
    err = rms(t_train - y_train);
    err_train(1,e) = err;
    w_train = [w_train w];
end
[min_err_train, i] = min(err_train,[],2);

best_w = w_train(:,((i-1)*n_hid_neurons)+1:(i*n_hid_neurons));

time_train = cputime - t;

% Testing
t = cputime;
    x = [x_test ones(size(x_test,1),1)];
    tempH = x * w;
    H = 1 ./ 1 + exp(-tempH);
    y_test =  H * B;
    err_test = rms(t_test - y_test);
time_test = cputime - t;

min_err_train
err_test
time_train
time_test
size(best_w)
size(w_train)
i
