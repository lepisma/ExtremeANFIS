function ndata = normalize(data)
    ndata = zeros(size(data));
    for i = 1:size(data, 2)
        ndata(:, i) = mat2gray(data(:, i));
    end
end