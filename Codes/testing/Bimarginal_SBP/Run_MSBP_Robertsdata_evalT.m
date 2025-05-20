close all; clear; clc;

set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Problem parameters
% ========================
epsilon = 0.001;

%% MSBP data
% =========================================================================
M0 = readmatrix("../halder_outfiles_0824/kbm_sim_32767_1080_0_t3.txt");
M1 = readmatrix("../halder_outfiles_0824/kbm_sim_32767_1080_0_t4.txt");

% Rescale
instr_scale = 1e-8;
% llcloads_scale = 1e-6;

M0(:,1) = M0(:,1) * instr_scale;
M1(:,1) = M1(:,1) * instr_scale;

% M0(:,2) = M0(:,2) * llcloads_scale;
% M1(:,2) = M1(:,2) * llcloads_scale;

% X = [M0(:,1) M0(:,2)];
% Y = [M1(:,1) M1(:,2)];

X = [M0(:,1)];
Y = [M1(:,1)];

% C = pdist2(X, Y, 'squaredeuclidean');
C = pdist2(X, Y, 'squaredeuclidean');
K = exp(-C/epsilon);
    
nSample = size(X,1); % We assume that all marginals have the same number of samples.
n = nSample;         % Simplify for later
    
mu0 = 1/n * ones(n, 1);
mu1 = 1/n * ones(n, 1);

%% Algorithm parameters
% ========================
maxIter = 10000; tol = 1e-12;

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

%% Calculate M to find intermediate distributions
%====================================================
M = diag(u0(:,iter_idx+1))*K*diag(u1(:,iter_idx+1));

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
% figure(2)
% semilogy(abs(er0),'-ro','LineWidth',2)
% hold on
% semilogy(abs(er1),'-bo','LineWidth',2)
% set(gca,'FontSize',30)
% xlabel('iteration index $j$','FontSize',30)
% ylabel('Error','FontSize',30)
% legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')

% % Plot u_0 and u_1
% figure(3)
% subplot(1,2,1)
% plot(u0)
% hold on;
% set(gca,'FontSize',30)
% xlabel('$i$','FontSize',30)
% ylabel("$(u_{0})_i$",'FontSize',30,'rotation',0)
% hold off;
% subplot(1,2,2)
% plot(u1)
% hold on;
% set(gca,'FontSize',30)
% xlabel('$i$','FontSize',30)
% ylabel("$(u_{1})_i$",'FontSize',30,'rotation',0)
% hold off;


taus = [0.3 0.5 0.75];
f4 = figure(4);
% Plot marginal distributions
% scatter3(zeros(size(X)), X, mu0);
[f,xi] = ksdensity(X, 'Weights', mu0, 'Bandwidth', 0.0003);
plot3(3*ones(size(xi)), xi, f, 'color', 'k');
hold on;
% scatter3(ones(size(Y)), Y, mu1);
[f,xi] = ksdensity(Y, 'Weights', mu1, 'Bandwidth', 0.0003);
plot3(4*ones(size(xi)), xi, f, 'color', 'k');

% Calculate intermediate distributions
for k=1:numel(taus)
    tau = taus(k);
    locs = zeros(numel(X)*numel(Y),1);
    mags = zeros(numel(X)*numel(Y),1);
    for i=1:n
        % disp(i);
        for j=1:n
            locs((i-1)*n+j) = (1-tau)*X(i)+tau*Y(j);
            mags((i-1)*n+j) = M(i,j);
        end
    end
    % scatter(tau*ones(numel(X)*numel(Y),1), locs, mags)
    [f,xi] = ksdensity(locs, 'Weights', mags, 'Bandwidth', 0.001^2);
    plot3((3+tau)*ones(size(xi)), xi, f, 'color', 'b');
    disp(sum(mags))
end
xlabel("t");
hold off;

% pathX = 0:numMarginal-1;
% for k=1:n
%     pathY = zeros(numMarginal);
%     pathY(1) = M{1}(k);
%     
%     for j=1:numMarginal-1
%         pathY(j+1) = T{j}(pathY(j));
%     end
%     
% 	plot(pathX, pathY);
% end
