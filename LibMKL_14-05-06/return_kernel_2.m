function kernel = return_kernel_2(coefficients, basekernels)
% [PURPOSE]
%
%   - Return a kernel that is constructed as K = \sum^M_{m=1} d_m \tilde{K_m} 

[n, row, col] = size(basekernels);
kernel = zeros(n, row);
for i = 1:col
    basekernel(:, :) = basekernels(:, :, i);
    kernel = kernel + coefficients(i)*basekernel;
    clear basekernel
end
