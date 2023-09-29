close all; clear; clc;

set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Problem parameters
% ========================
epsilon = 0.1;

%% MSBP data
% =========================================================================
M0 = readmatrix("../halder_outfiles_0824/kbm_sim_32767_1080_0_t4.txt");
M1 = readmatrix("../halder_outfiles_0824/kbm_sim_32767_1080_0_t5.txt");

% Rescale
instr_scale = 1e-8;
llcloads_scale = 1e-6;

M0(:,1) = M0(:,1) * instr_scale;
M1(:,1) = M1(:,1) * instr_scale;

M0(:,2) = M0(:,2) * llcloads_scale;
M1(:,2) = M1(:,2) * llcloads_scale;

X = [M0(:,1) M0(:,2)];
Y = [M1(:,1) M1(:,2)];

% C = pdist2(X, Y, 'squaredeuclidean');
C = pdist2(X, Y, 'squaredeuclidean');
K = exp(-C/epsilon);
    
nSample = size(X,1); % We assume that all marginals have the same number of samples.
n = nSample;         % Simplify for later
    
mu0 = 1/n * ones(n, 1);
mu1 = 1/n * ones(n, 1);

%% Algorithm parameters
% ========================
maxIter = 1000; tol = 1e-12;

% u0 = log( [rand(nSample,1), zeros(nSample,maxIter)] ); 
% u1 = log( [rand(nSample,1), zeros(nSample,maxIter)] );
u0 = [rand(nSample,1), zeros(nSample,maxIter)];
u1 = [rand(nSample,1), zeros(nSample,maxIter)];

er0 = zeros(maxIter,1); er1 = zeros(maxIter,1);

iter_idx = 1;
while iter_idx <= maxIter
    disp(['Iteration ',num2str(iter_idx)])

    u0(:,iter_idx+1) = mu0 ./ (K * u1(:,iter_idx));

    u1(:,iter_idx+1) = mu1 ./ (K' * u0(:,iter_idx+1));

    er0(iter_idx) = HilbertProjectiveMetric(u0(:,iter_idx+1),u0(:,iter_idx));
    er1(iter_idx) = HilbertProjectiveMetric(u1(:,iter_idx+1),u1(:,iter_idx));
    
    disp(['Err0 ',num2str(er0(iter_idx))])
    disp(['Err1 ',num2str(er1(iter_idx))])
    
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
% % figure(1)
% % subplot(1,2,1)
% % contourf(X,Y,reshape(mu0,size(X)))
% % set(gca,'FontSize',30)
% % xlabel('$x_{1}$','FontSize',30)
% % ylabel('$x_{2}$','FontSize',30,'rotation',0)
% % title('$\mu_0$','FontSize',30)
% % subplot(1,2,2)
% % contourf(X,Y,reshape(mu1,size(X)))
% % set(gca,'FontSize',30)
% % xlabel('$x_{1}$','FontSize',30)
% % ylabel('$x_{2}$','FontSize',30,'rotation',0)
% % title('$\mu_1$','FontSize',30)

% plot the covergence in Hilbert metric
figure(2)
semilogy(abs(er0),'-ro','LineWidth',2)
hold on
semilogy(abs(er1),'-bo','LineWidth',2)
set(gca,'FontSize',30)
xlabel('iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')


