close all; clear; clc;

set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% MSBP data
% =========================================================================
% time horizon
% numMarginal = 6;
% 
% t_initial = 0; t_final = 5; dt = 0.1; t_vec = t_initial: dt :t_final;

% state space
xmin = -2; xmax = 2; dx = 0.1; x = xmin:dx:xmax; 
% ymin = -4; ymax = 4; dy = dx; y = ymin:dy:ymax;
y = x;

[X,Y] = meshgrid(x,y); XX = [X(:) Y(:)]; nSample = size(XX,1);

%% Problem parameters
% ========================
epsilon = 0.1; % entropic regularization parameter

C = pdist2(XX,XX,'squaredeuclidean');

Gamma = exp(-C/epsilon); 

% endpoint measures
mu0 = (1 + ((X.^2 - 16).^2).*exp(-X/2)).*(1 + ((Y.^2 - 16).^2).*exp(-Y/2));
mu0 = mu0/sum(mu0,"all");
mu0 = mu0(:);

mu1 = (1.2 - cos(pi*(X+4)/2)).*(1.2 - cos(pi*(Y+4)/2));
mu1 = mu1/sum(mu1,"all");
mu1 = mu1(:);

%% Algorithm parameters
% ========================
maxIter = 1000; tol = 1e-3;

u0 = [rand(nSample,1), zeros(nSample,maxIter)]; 
u1 = [rand(nSample,1), zeros(nSample,maxIter)];

er0 = zeros(maxIter,1); er1 = zeros(maxIter,1);

iter_idx = 1;
while iter_idx <= maxIter
    disp(['Iteration ',num2str(iter_idx)])

    u0(:,iter_idx+1) = mu0 ./ (Gamma * u1(:,iter_idx));

    u1(:,iter_idx+1) = mu1 ./ (Gamma' * u0(:,iter_idx+1));

    er0(iter_idx) = HilbertProjectiveMetric(u0(:,iter_idx+1),u0(:,iter_idx));
    er1(iter_idx) = HilbertProjectiveMetric(u1(:,iter_idx+1),u1(:,iter_idx));
    
    % check convergence in Hilbert metric
    if (er0(iter_idx)<tol && er1(iter_idx)<tol) 
        break;       
    else         
      iter_idx = iter_idx+1;   
    end
end

%% Plot stuff
%=======================================
% plot the endpoint measures
figure(1)
subplot(1,2,1)
contourf(X,Y,reshape(mu0,size(X)))
set(gca,'FontSize',30)
xlabel('$x_{1}$','FontSize',30)
ylabel('$x_{2}$','FontSize',30,'rotation',0)
title('$\mu_0$','FontSize',30)
subplot(1,2,2)
contourf(X,Y,reshape(mu1,size(X)))
set(gca,'FontSize',30)
xlabel('$x_{1}$','FontSize',30)
ylabel('$x_{2}$','FontSize',30,'rotation',0)
title('$\mu_1$','FontSize',30)

% plot the covergence in Hilbert metric
figure(2)
semilogy(er0,'-ro','LineWidth',2)
hold on
semilogy(er1,'-bo','LineWidth',2)
set(gca,'FontSize',30)
xlabel('iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')



