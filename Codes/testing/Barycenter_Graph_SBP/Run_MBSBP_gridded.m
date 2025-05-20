%% Description
%
% We solve a Barycenter tracking MSBP with synthetic data
%
% =========================================================================
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Problem parameters
num_Timesteps = 5;  % The number of time steps to solve over
num_CPUs = 4;        % The number of 'nodes' (sensors, CPUs, etc.) at each time step

numMarginal_Time = num_Timesteps; 
numMarginal_CPU  = num_CPUs + 1; % (+1) to include the spectral barycenter

% state space
grid_range = 0.1;
dx = grid_range / 10;
xmin = -grid_range; xmax = grid_range; x = xmin:dx:xmax;
y = x;
[X,Y] = ndgrid(x,y); XX = [X(:) Y(:)]; nSample = size(XX,1);
% ========================
epsilon = 0.01; % entropic regularization parameter
C = pdist2(XX,XX,'squaredeuclidean');   % inter-barycenter cost matrix
C_tilde = C;                            % barycenter-to-marginal spectra cost matrix
K = exp(-C/epsilon);
K_tilde = exp(-C_tilde/epsilon);
mu = cell(numMarginal_Time,numMarginal_CPU);

% mean0 = [0.1; 0.1];
mean0 = [0 ; 0];
 
Cov0 = [1.25 0.05;
    0.05 1.01];
A = [0.4 -0.1;
    2 0.6];
% endpoint measures
mu{1,1} = mvnpdf(XX,mean0',Cov0);
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        % M = A^(numMarginal_Time*(k-1)+j);
        mean0 = (rand(2,1) - 1/2) ./ (1/grid_range);
        M = rand(2,1);
        M = M*M';
        M = eye(2) + M;
        mu{k,j} = mvnpdf(XX,(M*mean0)',M*Cov0*M');
        mu{k,j} = mu{k,j}/sum(mu{k,j});
    end
end

%% Visualize Marginal Data
% =======================================
figure(1)
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        subplot(numMarginal_Time,numMarginal_CPU,numMarginal_Time*(k-1)+j);
        contourf(X,Y,reshape(mu{k,j},size(X)))
    % set(gca,'FontSize',30)
    % xlabel('$x_{1}$','FontSize',30)
    % ylabel('$x_{2}$','FontSize',30,'rotation',0)
    end
end

%% Algorithm parameters
% ========================
maxIter = 1000; tol = 1e-3; maxtol = 1e5;
u = cell(numMarginal_Time,numMarginal_CPU);
err = cell(numMarginal_Time,numMarginal_CPU);

for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        u{k,j} = rand(nSample,1);
        % u{k,j} = u{k,j}/norm(u{k,j});
        err{k,j} = { []; [] };
    end
end
     
% tic;
iter_idx = 1;
t = 1;
j = 2;
while iter_idx <= maxIter
    fprintf('(Iter,t,j) = (%d, %d, %d)\n', iter_idx, t, j);
    
    u_old = u{t,j};
    
    % Calculate projection
    Proj = Proj1_gridded(t, j, numMarginal_Time, numMarginal_CPU, K, K_tilde, u);

    % Update iteration
    u{t,j} = u{t,j} .* mu{t,j} ./ Proj;
    
    % Calculate error
    err{t,j}{1}(end+1) = iter_idx;
    err{t,j}{2}(end+1) = HilbertProjectiveMetric(u{t,j},u_old);
    
    disp(['Err ',num2str(err{t,j}{2}(end))])
    max_err = err{t,j}{2}(end);
    if (iter_idx >= numMarginal_Time*(numMarginal_CPU-1))
        for k=1:numMarginal_Time
            for l=2:numMarginal_CPU
                max_err = max(max_err, err{k,l}{2}(end));
            end
        end
    else
        max_err = 228;
    end
    disp(['Max_Err ',num2str(max_err)])
    
    % check convergence in Hilbert metric
    if (max_err < tol)
        break;
    elseif( isinf(err{t,j}{2}(end)) || isnan(err{t,j}{2}(end)))
        fprintf('Error: NaN or Inf detected in Hilbert metric on iteration (t,j)=(%d,%d). Stopping...\n', t, j);
        break;
    else        
        iter_idx = iter_idx+1;  
        if j == numMarginal_CPU
            j = 2;
        else
            j = j + 1;
        end
        % j = mod(j, numMarginal_CPU) + 1;
        if j == 2
            t = mod(t, numMarginal_Time) + 1;
        end
    end
end
% comptime_recursion = toc

%%
% plot the covergence in Hilbert metric
figure(2)
% semilogy(err{2,2}{1}, err{2,2}{2}, 'LineWidth',2, 'color', 'k')
% hold on;
for k=1:numMarginal_Time
    for l=2:numMarginal_CPU
        semilogy(err{k,l}{1}, err{k,l}{2}, 'LineWidth',2);
        hold on;
    end
end
% set(gca,'FontSize',30)
xlabel('iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
% legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')
yline(tol);
hold off;

