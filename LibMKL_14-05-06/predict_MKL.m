function ypred = predict_MKL(model,coefficients,bKt)

% LibMKL by Xu Xinxing@CeMNet, NTU, 2013-12-13; contact:xxxing1987@gmail.com
%% test code for MKL;
% bKt: nt*n*M tensor for M base kernels from the nt testing points and n training points;
% coefficients: the learned kernel combination coefficients from train_LpMKL.m;
% model: the trained model from train_LpMKL.m
%% Reference:
% Xinxing Xu, Ivor W. Tsang and Dong Xu: Soft Margin Multiple Kernel Learning. 
% IEEE Trans. Neural Netw. Learning Syst., vol. 24, no. 5, pp. 749–761, 2013.

    bKt_new = bKt(:,model.SVs,:);
    Kt = return_kernel_2(coefficients, bKt_new);
    w = model.sv_coef*model.Label(1,1);
    bsvm = -model.rho*model.Label(1,1);
    ypred=Kt*w+bsvm;
    
end