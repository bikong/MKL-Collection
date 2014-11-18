function muopt = box_lpnorm_inverse_proj_dense(theta,a,p)
%%
% box constraint simplex projection:
% min sum(a_{m}/mu_{m})
% s.t. theta0 < = mu <= theta;
%      sum(mu_{m}^(p)) = 1.
%      theta0 = max(1-(n-1)*theta,0) >=0
% xxxing1987@gmail.com
% Modified Oct. 25, 2012@CeMNet,NTU
%%
n = length(a);
theta0 = max(1-(n-1)*theta,0);
muopt = [];
mu_rho = theta0*ones(n,1);
sign = 1;
rho = 0;
[a_sort,sort_ind] = sort(a,'descend');
rhon0 = find(a_sort>0, 1, 'last' );

while rho<rhon0 && sign
%       lambda_sqrt = sum(sqrt(a_sort(rho+1:rhon0,1)));
      lambda_lp = sum(a_sort(rho+1:rhon0,1).^(p/(p+1)));
      mu_rho(1:rhon0,1) = (a_sort(1:rhon0,1)).^(1/(p+1))*((1-rho*theta.^(p)))^(1/p)./((lambda_lp).^(1/p));
      mu_rhop1 = mu_rho(rho+1);
      if (mu_rhop1 <= theta)% = in case of 1 non-zero element for a;
         if rhon0>rho
            sign = 0;
            muopt(sort_ind(1:rho,:),1) = theta;
            muopt(sort_ind(1+rho:end,:),1) = mu_rho(1+rho:end,:);
         end
      end
      rho = rho + 1;
end

if isempty(muopt) % in case of zero values in the a;
   muopt(sort_ind(1:rhon0,:),1) = theta;
   muopt(sort_ind(1+rhon0:end,:),1) = (1-theta*rhon0)/(n-rhon0);
end
    
end