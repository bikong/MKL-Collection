function Sigma_new = analytic_solution_sm_lpnorm(Sigma,h,p,theta)
    w_square = h.*(Sigma.^2);
    w_square = max(w_square,0);
    Sigma_new = box_lpnorm_inverse_proj_dense(theta,w_square,p);
end