function [coefficients, model, obj] = train_SM1MKL(labels, basekernels, C, theta)
%% Hinge Loss Soft Margin MKL;
% contact: Xu Xinxing, xuxi0006@e.ntu.edu.sg;
% July-7-2012
%% Reference:
% Xinxing Xu, Ivor W. Tsang and Dong Xu: Soft Margin Multiple Kernel Learning. 
% IEEE Trans. Neural Netw. Learning Syst., vol. 24, no. 5, pp. 749–761, 2013.
%%
% input: labels n*n ; label vector
%        basekernels n*n*m; a total number of m base kernels
%        C ; C parameter for SVM;
%        theta;  the regularization parameter in the range of (1/M =<theta <=1);
% output: coefficients, the linear kernel combination coefficients;
%%
% addpath('.\libsvm-3.12\matlab');
%%
% if nargin <= 
%     C = 1; % default
% end
p = 1;
n_basekernels = size(basekernels, 3);
tau         = 1e-3;
MAX_ITER    = 30;   %   the maximum iteration for the WHILE loop in Algorithm 1
history.obj = [];

%%% Initilization
coefficients    = 1/n_basekernels * ones(n_basekernels, 1);

%%% Main code
%fprintf('\tIter #1 :\t');
[model, obj, q] = return_alpha(n_basekernels, coefficients, labels, basekernels, C);
history.obj     = obj;
%fprintf('obj = %.15f\n', obj);

for i = 2:MAX_ITER
    %fprintf('\tIter #%-2d:\t', i);
    
    %coefficients_new = analytic_solution(coefficients,q,p);
    coefficients_new = analytic_solution_sm_lpnorm(coefficients,q,p,theta);

    
    [model, obj, q] = return_alpha(n_basekernels, coefficients_new, labels, basekernels, C);
    
    coefficients = coefficients_new;
    history.obj = [history.obj;obj];
%    fprintf('obj = %.15f, abs(obj(%d) - obj(%d)) = %.15f\n', history.obj(i), i, i-1, abs(history.obj(i) - history.obj(i-1)));

    if abs(history.obj(i) - history.obj(i-1))/history.obj(i-1) <= tau
        break;
    end
end

end

%%% Subfunction: return_alpha
function [model, obj, q] = return_alpha(n_basekernels, coefficients, labels, basekernels, C)

training_kernel = return_kernel_2(coefficients, basekernels);

model = svmtrain(labels, [(1:size(training_kernel, 1))', double(training_kernel)], ['-q -s 0 -t 4 -c ', num2str(C)]);
alpha = zeros(length(labels), 1);

alpha(full(model.SVs)) = abs(model.sv_coef);
y_alpha                = labels .* alpha;

q = zeros(n_basekernels, 1);
for m = 1:n_basekernels
    training_basekernel(:, :) = basekernels(:, :, m);
    q(m) = 0.5 * y_alpha' * training_basekernel * y_alpha;
    clear training_basekernel
end

obj = (sum(alpha) - q'*coefficients);

end
