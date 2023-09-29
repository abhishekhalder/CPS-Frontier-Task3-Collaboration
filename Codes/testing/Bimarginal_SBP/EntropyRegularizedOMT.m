function [w_gam_squared, OptimalCouplingMatrix_gam] = EntropyRegularizedOMT(x,y,rho_x,rho_y,gam)

% [X,Y] = meshgrid(x,y);
% 
% C = (X-Y).^2;
% M = exp(-((X-Y).^2)/gam);

% alternatively (faster runtime)
C = pdist2(x,y,'squaredeuclidean');
M = exp(-C/gam);

OptimalCouplingMatrix_gam = SinkhornOMT(M, rho_x, rho_y);


Elementwise_log = log(OptimalCouplingMatrix_gam);
Elementwise_log(Elementwise_log == -Inf) = 0; % force log(0) = -Inf terms to zero

% w_gam_squared = trace(C'*OptimalCouplingMatrix_gam) - gam*sum(sum(OptimalCouplingMatrix_gam.*Elementwise_log));
w_gam_squared = trace(C*OptimalCouplingMatrix_gam') - gam*sum(sum(OptimalCouplingMatrix_gam.*Elementwise_log));