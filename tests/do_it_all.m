raw = importdata('F:\,.Advanced Learning\Pillai Sir\ExANFIS\ExAnfis_Classification\dat\balance-scale.txt');
raw_data = [raw.data zeros(size(raw.data,1),1)];
raw_class = cell2mat(raw.textdata);
data_1 = [];
data_2 = [];
data_3 = [];
class_1 = 'L';
class_2 = 'R';
class_3 = 'B';

% sorting data according to classes
for i = 1:size(raw_data,1)
    if raw_class(i) == class_1
        data_1 = [data_1; raw_data(i,:)];
        data_1(end,end) = -1;
    end
end
size_class_1 = size(data_1,1);

for i = 1:size(raw_data,1)
    if raw_class(i) == class_2
        data_2 = [data_2; raw_data(i,:)];
        data_2(end,end) = 0;
    end
end
size_class_2 = size(data_2,1);

for i = 1:size(raw_data,1)
    if raw_class(i) == class_3
        data_3 = [data_3; raw_data(i,:)];
        data_3(end,end) = 1;
    end
end
size_class_3 = size(data_3,1);

% setting train-to-test ratio
train_ratio = 0.8;
train_count_1 = floor(size_class_1 * train_ratio);
train_count_2 = floor(size_class_2 * train_ratio);
train_count_3 = floor(size_class_3 * train_ratio);

% setting training and test indices
data_train_1 = [data_1(1:train_count_1, 1:end-1) ones(train_count_1,1) ; data_2(1:train_count_2, 1:end-1) zeros(train_count_2,1) ; data_3(1:train_count_3, 1:end-1) zeros(train_count_3,1)];
data_train_2 = [data_1(1:train_count_1, 1:end-1) zeros(train_count_1,1) ; data_2(1:train_count_2, 1:end-1) ones(train_count_2,1) ; data_3(1:train_count_3, 1:end-1) zeros(train_count_3,1)];
data_test = [data_1(train_count_1 +1 : end, :); data_2(train_count_2 +1 : end, :); data_3(train_count_3 +1 : end, :)];


% obtained train and test data
% now starting the real thing

n_mfs = 3;

ex_fis_1 = extreme.exanfis(data_train_1, n_mfs, 50, data_test);
ex_fis_2 = extreme.exanfis(data_train_2, n_mfs, 50, data_test);

infis_1 = genfis1(data_train_1, n_mfs, 'gbellmf');
afis_1 = anfis(data_train_1, infis_1, 20);

infis_2 = genfis1(data_train_2, n_mfs, 'gbellmf');
afis_2 = anfis(data_train_2, infis_2, 20);

efis_1 = sir.elanfis(data_train_1(:, 1:end-1), data_train_1(:, end), n_mfs, 50);
efis_2 = sir.elanfis(data_train_2(:, 1:end-1), data_train_2(:, end), n_mfs, 50);

a_1_out = evalfis(data_test(:, 1:end-1), afis_1);
e_1_out = evalfis(data_test(:, 1:end-1), efis_1);
ex_1_out = evalfis(data_test(:,1:end-1), ex_fis_1);

a_2_out = evalfis(data_test(:, 1:end-1), afis_2);
e_2_out = evalfis(data_test(:, 1:end-1), efis_2);
ex_2_out = evalfis(data_test(:,1:end-1), ex_fis_2);

for i = 1:size(data_test,1)
   if a_1_out(i) < 0.50
       a_1_out(i) = 0;
   else
       a_1_out(i) = 1;
   end 
 
   if e_1_out(i) < 0.50
       e_1_out(i) = 0;
   else
       e_1_out(i) = 1;
   end 
 
   if ex_1_out(i) < 0.50
       ex_1_out(i) = 0;
   else
       ex_1_out(i) = 1;
   end 
   
   % now for the second class
   if a_2_out(i) < 0.50
       a_2_out(i) = 0;
   else
       a_2_out(i) = 1;
   end 
 
   if e_2_out(i) < 0.50
       e_2_out(i) = 0;
   else
       e_2_out(i) = 1;
   end 
 
   if ex_2_out(i) < 0.50
       ex_2_out(i) = 0;
   else
       ex_2_out(i) = 1;
   end 
end 

out_1 = [( ones((size_class_1-train_count_1),1)); zeros((size_class_2-train_count_2),1); zeros((size_class_3-train_count_3),1)];
out_2 = [( zeros((size_class_1-train_count_1),1)); ones((size_class_2-train_count_2),1); zeros((size_class_3-train_count_3),1)];
   
a_error_count_1 = sum(a_1_out ~= out_1);
e_error_count_1 = sum(e_1_out ~= out_1);
ex_error_count_1 = sum(ex_1_out ~= out_1);

% a_error_1 = 100 * (a_error_count / size(data_test,1))
% e_error_1 = 100 * (e_error_count / size(data_test,1))
% ex_error_1 = 100 * (ex_error_count / size(data_test,1))

a_error_count_2 = sum(a_2_out ~= out_2);
e_error_count_2 = sum(e_2_out ~= out_2);
ex_error_count_2 = sum(ex_2_out ~= out_2);

% a_error_2 = 100 * (a_error_count_2 / size(data_test,1))
% e_error_2 = 100 * (e_error_count_2 / size(data_test,1))
% ex_error_2 = 100 * (ex_error_count_2 / size(data_test,1))

a_error = 100 * (a_error_count_1 + a_error_count_1 / size(data_test,1))
e_error = 100 * (e_error_count_1 + e_error_count_2 / size(data_test,1))
ex_error = 100 * (ex_error_count_1 + ex_error_count_2 / size(data_test,1))
