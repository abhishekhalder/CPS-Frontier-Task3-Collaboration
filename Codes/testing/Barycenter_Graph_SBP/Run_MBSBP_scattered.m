%% Description
%
% We solve a Barycenter tracking MSBP with synthetic data, 
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
nSample = 500;                                  % The number of scattered points at each spectral marginal.
n = nSample;
num_interpedges = (num_CPUs*(num_CPUs-1)) / 2;  % The number of spectral marginal pairs to interpolate
                                                % between for the computation of the barycenter (equal to
                                                % to the number of edges in the complete graph K_{num_CPUs})

n_cluster = 100;                                % The number of clusters in each interpolation.
n_tilde = num_interpedges * n_cluster;          % Number of points in the barycenter

grid_range = 0.1;
dx = grid_range / 10;
xmin = -grid_range; xmax = grid_range;

% ========================

epsilon = 0.1; % entropic regularization parameter

mu = cell(numMarginal_Time,numMarginal_CPU);
locs = cell(numMarginal_Time,numMarginal_CPU);
C = cell(numMarginal_Time,numMarginal_CPU);
K = cell(numMarginal_Time,numMarginal_CPU);

pmin = 3;
prange = 3;

% endpoint measures
for k=1:numMarginal_Time
    % Generate scattered data points for all spectral marginals
    for j=2:numMarginal_CPU
        
        p = ceil( pmin + prange * rand(1) );
        X = xmin + (xmax - xmin)*sum(rand(n,p),2)/p;
        Y = xmin + (xmax - xmin)*sum(rand(n,p),2)/p;
        
        locs{k,j} = [X(:), Y(:)];
        
        mu{k,j} = (1/n) * ones(n,1);
    end
    
    % Generate scattered data points for Barycenters via interpolation between
    % all pairs of spectral marginals, followed by clustering
    iter = 0;
    locs{k,1} = zeros(n_tilde, 2);
	for j=2:numMarginal_CPU
        for l=j+1:numMarginal_CPU
            tlocs = zeros(n^2,2);
            for a=1:n
                for b=1:n
                    tlocs((a-1)*n+b,:) = ( locs{k,j}(a,:)+locs{k,l}(b,:) ) / 2;
                end
            end
            tlocs = downsample(tlocs, n^2/n_cluster);   % This is very crude
            locs{k,1}(1+(iter*n_cluster):((iter+1)*n_cluster),:) = tlocs;
            iter = iter + 1;
        end
        %tlocs = pcdownsample(tlocs, 'nonuniformGridSample', n_cluster);
    end
    
    % Generate cost matrices
    for j=2:numMarginal_CPU
        % C{k,j} = pdist2(locs{k,j}, locs{k,1});
        C{k,j} = pdist2(locs{k,1}, locs{k,j}); % C_tilde
        K{k,j} = exp(-C{k,j}/epsilon);
    end
    if k > 1
        C{k-1,1} = pdist2(locs{k-1,1}, locs{k,1});
        K{k-1,1} = exp(-C{k-1,1}/epsilon);
    end
    mu{k,1} = (1/n_tilde) * ones(n_tilde,1);
end

%% Visualize Marginal Data
% =======================================
figure(1)
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        subplot(numMarginal_Time,numMarginal_CPU,numMarginal_Time*(k-1)+j);
        scatter(locs{k,j}(:,1), locs{k,j}(:,2));
        title("locs\{" + num2str(k) + "," + num2str(j) + "\}");
    end
end

%% Algorithm parameters
% ========================
maxIter = 1000; tol = 1e-3; maxtol = 1e5;
u = cell(numMarginal_Time,numMarginal_CPU);
err = cell(numMarginal_Time,numMarginal_CPU);

for k=1:numMarginal_Time
    u{k,1} = rand(n_tilde,1);
    err{k,1} = { []; [] };
    
    for j=2:numMarginal_CPU
        u{k,j} = rand(n,1);
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
    % Proj = Proj1_gridded(t, j, numMarginal_Time, numMarginal_CPU, K, K_tilde, u);
    Proj = Proj1_scattered(t, j, numMarginal_Time, numMarginal_CPU, K, u);

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

