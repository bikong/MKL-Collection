function toy_MKL()

%% a sample code for 
% LibMKL by Xu Xinxing@CeMNet, NTU, 2014-05-06; contact:xxxing1987@gmail.com
%% Reference:
% Xinxing Xu, Ivor W. Tsang and Dong Xu: Soft Margin Multiple Kernel Learning. 
% IEEE Trans. Neural Netw. Learning Syst., vol. 24, no. 5, pp. 749–761, 2013.
p = 2;
C = 10000000;
n = 100;
dim = 10;
X = rand(n,dim);
labels = sign(sum(X,2)/dim-0.6);
m = 1000;
Xt = rand(m,dim);
for ith = 1:dim
    basekernels(:,:,ith) = X(:,ith)*(X(:,ith)');
    bKt(:,:,ith) = Xt(:,ith)*(X(:,ith)');
end

%% smaple code for using the SMMKL;
%% training phase;
% labels: n*1 label vectors, with +1, or -1;
% basekernels: n*n*M tensor for M base kernels from the n training points;
% C: the SVM hyperparameter;

theta = 1./(size(basekernels,3)-3);% theta is in the range of (1/M =<theta <=1);
[coefficients1, model1, obj1] = train_SM1MKL(labels, basekernels, C, theta);
[coefficients2, model2, obj2] = train_LpMKL(labels, basekernels, p, C);

%% testing phase;
% bKt: nt*n*M tensor for M base kernels from the nt testing points and n training points;
% coefficients: the learned kernel combination;
% model: the trained model;
ypred1 = predict_MKL(model1,coefficients1,bKt);
ypred2 = predict_MKL(model2,coefficients2,bKt);


end

