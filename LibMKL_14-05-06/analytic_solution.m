function Sigma_new = analytic_solution(Sigma,h,p)

%%
% p = 2;
% h = rand(10,1);
% Sigma = ones(10,1);
%%
    nbkernel = length(Sigma);
    w_square = h.*(Sigma.^2);
    w_square = max(w_square,0);
    
    wp_norm = sum(w_square.^(p/(p+1)))^(1/p);

    for i = 1: nbkernel
        Sigma_new(i,1) = w_square(i,1).^(1/(p+1))/wp_norm;
    end
    
end