%% Description
%
% We solve a Barycenter tracking MSBP with Robert's parsec data, 
%
%==========================================================================
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
rng(0);

%% Problem parameters
%==========================================================================
dfile_pfix = "../halder_outfiles_0409/canneal_1__MARG";
dfile_sfix = "_3dim.txt";
out_dir    = "./data_out/";

marg_times    = [0.0, 0.5, 1.5, 2.5, 5.0, 9.5, 10.5];     % Times at which marginals are placed
taus          = [0.8, 2.2, 7.0, 9.0, 10.0];               % Times at which to interpolate
num_Timesteps = 7;          % The number of time steps to solve over
num_CPUs      = 4;          % The number of 'nodes' (sensors, CPUs, etc.) at each time step

numMarginal_Time = num_Timesteps; 
numMarginal_CPU  = num_CPUs + 1; % (+1) to include the spectral barycenter

% state space
nSample = 400;                                  % The number of scattered points at each spectral marginal.
n = nSample;
num_interpedges = (num_CPUs*(num_CPUs-1)) / 2;  % The number of spectral marginal pairs to interpolate
                                                % between for the computation of the barycenter (equal to
                                                % to the number of edges in the complete graph K_{num_CPUs})

n_cluster = 100;                                % The number of clusters in each interpolation.
n_tilde = num_interpedges * n_cluster;          % Number of points in the barycenter

% Default data scaling parameters
instr_scale     = 1e-9;
llcreq_scale    = 1e-7;
llcmiss_scale   = 1e-5;

autoscale = false;                               % If false, use defaults above
scaled_maxval = 0.1;                            % If autoscale==true, all components of data scaled to [0,scaled_maxval]

epsilon = 0.05;                                  % entropic regularization parameter
%==========================================================================

rawD = cell(numMarginal_Time,num_CPUs);              % raw data
mu = cell(numMarginal_Time,numMarginal_CPU);
locs = cell(numMarginal_Time,numMarginal_CPU);
C = cell(numMarginal_Time,numMarginal_CPU);
K = cell(numMarginal_Time,numMarginal_CPU);

% endpoint measures
for k=1:numMarginal_Time
    % Load and scale data into spectral marginals
    for j=2:numMarginal_CPU
        rawD{k,j} = readmatrix(dfile_pfix + num2str(k-1) + "." + num2str(j-1) + dfile_sfix);
        
        if( size(rawD{k,j},1) < n )                     % Pad the data if necessary
            rawD{k,j} = padarray(rawD{k,j},n-size(rawD{k,j},1),0,'post');
        end
        
        if( autoscale )
            if( max(rawD{k,j}(:,1)) ~= 0 ) instr_scale   = scaled_maxval / max(rawD{k,j}(:,1)); else instr_scale   = 1; end
            if( max(rawD{k,j}(:,2)) ~= 0 ) llcreq_scale  = scaled_maxval / max(rawD{k,j}(:,2)); else llcreq_scale  = 1; end
            if( max(rawD{k,j}(:,3)) ~= 0 ) llcmiss_scale = scaled_maxval / max(rawD{k,j}(:,3)); else llcmiss_scale = 1; end
            fprintf("====Autoscaling Parameters====\n");
            fprintf('instr_scale   = %f\n', instr_scale);
            fprintf('llcreq_scale  = %f\n', llcreq_scale);
            fprintf('llcmiss_scale = %f\n', llcmiss_scale);
            fprintf("==============================\n\n");
        end
        
        rawD{k,j}(:,1) = rawD{k,j}(:,1) * instr_scale;
        rawD{k,j}(:,2) = rawD{k,j}(:,2) * llcreq_scale;
        rawD{k,j}(:,3) = rawD{k,j}(:,3) * llcmiss_scale;
        
        locs{k,j} = [rawD{k,j}(:,1), rawD{k,j}(:,2), rawD{k,j}(:,3)];
        
        mu{k,j} = (1/n) * ones(n,1);
    end
    
    % Generate scattered data points for Barycenters via interpolation between
    % all pairs of spectral marginals, followed by clustering
    iter = 0;
    locs{k,1} = zeros(n_tilde,3);
	for j=2:numMarginal_CPU
        for l=j+1:numMarginal_CPU
            tlocs = zeros(n^2,3);
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
% ==========================================================================
% figure(1)
% for k=1:numMarginal_Time
%     for j=1:numMarginal_CPU
%         subplot(numMarginal_Time,numMarginal_CPU,numMarginal_Time*(k-1)+j);
%         scatter(locs{k,j}(:,1), locs{k,j}(:,2));
%         title("locs\{" + num2str(k) + "," + num2str(j) + "\}");
%     end
% end

%% Algorithm parameters
%==========================================================================
maxIter = 10000; tol = 1e-13; maxtol = 1e5;
u = cell(numMarginal_Time,numMarginal_CPU);
err = cell(numMarginal_Time,numMarginal_CPU);
ptimes = [];

% for k=1:numMarginal_Time
%     u{k,1} = rand(n_tilde,1);
%     err{k,1} = { []; [] };
%     
%     for j=2:numMarginal_CPU
%         u{k,j} = rand(n,1);
%         err{k,j} = { []; [] };
%     end
% end
u_saved = load('u_nonrand.mat').u;
for k=1:numMarginal_Time
    u{k,1} = u_saved{k,1}(1:n_tilde);
    err{k,1} = { []; [] };
    
    for j=2:numMarginal_CPU
        u{k,j} = u_saved{k,j}(1:n);
        err{k,j} = { []; [] };
    end
end
     
tic;
iter_idx = 1;
t = 1;
j = 2;
while iter_idx <= maxIter
    fprintf('(Iter,t,j) = (%d, %d, %d)\n', iter_idx, t, j);
    
    u_old = u{t,j};
    
    % Calculate projection
    % tic;
    Proj = Proj1_scattered(t, j, numMarginal_Time, numMarginal_CPU, K, u);
    % ptimes(end+1) = toc;
    
    % Update iteration
    u{t,j} = u{t,j} .* mu{t,j} ./ Proj;
    
    % Calculate error
    err{t,j}{1}(end+1) = iter_idx;
    err{t,j}{2}(end+1) = max(1e-16, HilbertProjectiveMetric(u{t,j},u_old));
    
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
comptime_recursion = toc

%% Find the transport maps between adjacent marginals
%==========================================================================
M = cell(numMarginal_Time,numMarginal_CPU);
Msums = zeros(numMarginal_Time,numMarginal_CPU);
% for i=1:numMarginal-1
%     M{i} = Proj2_mm(i,i+1, K, u);
%     disp(sum(M{i},"all"));
% end
for k=1:numMarginal_Time
    for j=2:numMarginal_CPU
        % C{k,j} = pdist2(locs{k,1}, locs{k,j}); % C_tilde
        % K{k,j} = exp(-C{k,j}/epsilon);
        
        M{k,j} = Proj2_scattered([k,1], [k,j], numMarginal_Time, numMarginal_CPU, K, u);
        Msums(k,j) = sum(M{k,j},"all");
        disp(sum(M{k,j},"all"));
    end
    if k > 1
        % C{k-1,1} = pdist2(locs{k-1,1}, locs{k,1});
        % K{k-1,1} = exp(-C{k-1,1}/epsilon);
        
        M{k-1,1} = Proj2_scattered([k-1,1], [k,1], numMarginal_Time, numMarginal_CPU, K, u);
        Msums(k-1,1) = sum(M{k-1,1},"all");
        disp("t" + num2str(k) + " " + sum(M{k-1,1},"all"));
    end
end

%% Figure 1 : Plot convergence in Hilbert metric
%==========================================================================
% plot the covergence in Hilbert metric
figure(1)
% semilogy(err{2,2}{1}, err{2,2}{2}, 'LineWidth',2, 'color', 'k')
% hold on;
for k=1:numMarginal_Time
    for l=2:numMarginal_CPU
        semilogy(err{k,l}{1}, err{k,l}{2}, 'LineWidth',2);
        writecell(err{k,l}, out_dir + "f1_err" + num2str((k-1)*num_CPUs + (l-1)) + ".txt");
        hold on;
    end
end
% set(gca,'FontSize',30)
xlabel('Iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
% legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')
yline(tol);
hold off;

%% Figure 2 : Plot interpolated marginals at specified times
%==========================================================================
num_interp = numel(taus);
r = 1e-3;

figure(2);

% Figure 2: Plot interpolated distributions
% M_int = Proj2_mm(si, ei, K, u);

% ts   = (0:(numIntermediate+1))/(numIntermediate+1);
ilocs    = cell(num_interp, num_CPUs);
imags    = cell(num_interp, num_CPUs);
weights  = cell(num_interp, num_CPUs);

nS       = 500;                     % Number of downsamples

dlocs    = cell(num_interp, num_CPUs);      % Downsampled locs
dmags    = cell(num_interp, num_CPUs);      % Downsampled mags
dweights = cell(num_interp, num_CPUs);      % Downsampled weights (for graphing)

for k=1:numel(taus)
    fprintf('k=%d\n', k);
        
    tau = taus(k);
    % Find correct M to use
    M_ind = find(marg_times>tau,1)-1;
    % Find bounds in interpolation interval
    tau_int_b = marg_times(M_ind);
    tau_int_e = marg_times(M_ind+1);
    tau_scaled = (tau-tau_int_b) / (tau_int_e-tau_int_b);
    
    for p=1:num_CPUs
        ilocs{k,p}    = zeros(n^2,3);
        imags{k,p}    = zeros(n^2,1);
        weights{k,p}  = zeros(n^2,1);
        dlocs{k,p}    = zeros(nS,3);
        dmags{k,p}    = zeros(nS,1);
        dweights{k,p} = zeros(nS,1);
        M_int = M{M_ind,p+1}' * M{M_ind,1} * M{M_ind+1,p+1};
        disp(sum(M_int,"all"));
        for i=1:n
            for j=1:n
                ilocs{k,p}((i-1)*n+j,:) = (1-tau_scaled)*rawD{M_ind,p+1}(i,:)+tau_scaled*rawD{M_ind+1,p+1}(j,:);
                imags{k,p}((i-1)*n+j)   = M_int(i,j);
            end
        end
        % Perform downsamping
        t_locs = ilocs{k,p};
        t_mags = imags{k,p};
        for i=1:nS
            % disp(i);
            dlocs{k,p}(i,:) = t_locs(1,:);
            dmags{k,p}(i) = t_mags(1);
            t_locs(1,:) = [];
            t_mags(1) = [];
            [minValues,cIs] = mink(vecnorm(dlocs{k,p}(i,:)-t_locs,2,2), n^2/nS-1);
            for j=1:(n^2/nS-1)
                dlocs{k,p}(i,:) = (dmags{k,p}(i)*dlocs{k,p}(i,:) + t_mags(cIs(j)).*t_locs(cIs(j),:)) ...
                    / (dmags{k,p}(i)+t_mags(cIs(j)));
                dmags{k,p}(i) = dmags{k,p}(i) + t_mags(cIs(j));
            end
            t_locs(cIs,:) = [];
            t_mags(cIs)   = [];
        end
        for j=1:nS
            dweights{k,p}(j) = sum( (vecnorm(dlocs{k,p}(j,:)-dlocs{k,p},2,2) < r) .* dmags{k,p} );
        end
        dweights{k,p} = dweights{k,p} / nS;
        
        subplot(num_CPUs, num_interp, (p-1)*num_interp+k);
        % scatter3(ilocs{k}(:,1), ilocs{k}(:,2), ilocs{k}(:,3), 1, weights{k}, 'filled');
        scatter3(dlocs{k,p}(:,1), dlocs{k,p}(:,2), dlocs{k,p}(:,3), 20, dweights{k,p}, 'filled');
        if( k == 1 )
            zlabel("CPU" + num2str(p),'FontSize',30)
        end
        if( p == 1 )
            title("$t=" + num2str(tau) + "$",'FontSize',20);
        end
        % set(gca,'XLim',x_lim,'YLim',y_lim,'ZLim',z_lim);
        % h = scatter3(locs{k}(:,1), locs{k}(:,2), locs{k}(:,3), 40, 'filled');
        % set(h, 'MarkerEdgeAlpha', alphaI, 'MarkerFaceAlpha', alphaI);
        % disp(sum(imags{k,p}))
        % title("$\hat{\mu}_{" + num2str(si+(ei-si)*(k/(num_interp+1)),3) + "}$",'FontSize',30);
    end
end

%% Load additional data for comparison
addD = cell(num_interp,num_CPUs);

figure(4);
% endpoint measures
for k=1:num_interp
    % Load and scale data into spectral marginals
    for j=1:num_CPUs
        % addD{k,j} = readmatrix("../halder_outfiles_0103/canneal_7_72_MARG" + num2str(k-1) + "." + num2str(j) + dfile_sfix);
        addD{k,j} = readmatrix(dfile_pfix + "INT" + num2str(k-1) + "." + num2str(j) + dfile_sfix);
        
        if( size(addD{k,j},1) < n )                     % Pad the data if necessary
            addD{k,j} = padarray(addD{k,j},n-size(addD{k,j},1),0,'post')
        end
        
        addD{k,j}(:,1) = addD{k,j}(:,1) * instr_scale;
        addD{k,j}(:,2) = addD{k,j}(:,2) * llcreq_scale;
        addD{k,j}(:,3) = addD{k,j}(:,3) * llcmiss_scale;
        
        subplot(num_CPUs, num_interp, (j-1)*num_interp+k);
        % scatter3(addD{k,j}(:,1), addD{k,j}(:,2), addD{k,j}(:,3), 40, 1/n*ones(n,1), 'filled');
        scatter3(addD{k,j}(:,1), addD{k,j}(:,2), addD{k,j}(:,3), 40, 'filled');
        % locs{k,j} = [addD{k,j}(:,1), addD{k,j}(:,2), addD{k,j}(:,3)];
    end
end


% %% Table 1: Compute Wasserstein distances.
% % Get Entropy-Regularized Wasserstein distances between interpolations
% % and measured.
% epsilon_wass = 0.01;
% wass  = zeros(num_interp,num_CPUs);
% for k=1:num_interp
%     for p=1:num_CPUs
%         wass(k,p) = Wasserstein(addD{k,p}, dlocs{k,p}, 1/n*ones(n,1), dmags{k,p});
%         fprintf("(k,p)=(%d,%d) | %f\n", k, p, wass(k,p));
%     end
% end
% save('wass_bc.mat', 'wass');
% 
% %% Print the values
% disp("--------Wasserstein Distances for Interpolations--------");
% for p=1:num_CPUs
%     fprintf('CPU%d | ', p);
%     for k=1:num_interp
%         fprintf('%5.5f ', wass(k,p));
%     end
%     fprintf('\n');
% end
% disp("--------------------------------------------------------");

%% Figure 5: Plot 1D marginalized interpolated marginals vs. measured at specified times
%==========================================================================
f5 = figure(5);
set(gcf,'color','w');
set(0, 'DefaultLineLineWidth', 2.5);
nBins = 200;
for p=1:num_CPUs
    for j=1:3
        subplot(6,2,(1+6*floor(p/3)+(1-mod(p,2)))+2*(j-1));
        
        % Plot the measured intermediate distributions
        for k=1:num_interp
            [X1, marg] = getMarginal1D(addD{k,p}(:,j), nBins);
            marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
            plot3(((k)-0.001)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "r");
            hold on;
            fill3(((k)-0.001)*ones(numel(X1),1), X1, marg, "r", 'FaceAlpha', 0.5);
            if( ~any(addD{k,p}(:,j)) )
                line([(k)-0.001,(k)-0.001], [0,0], [0,1000], 'Color', 'r');
            end
        end
        
        % Plot interpolated distributions
        for k=1:num_interp
            %         [X1, marg] = getWeightedMarginal1D(locs{k}(:,j), mags{k}, nBins*1);
            %         marg = smoothdata(marg);
            %         plot3((si+(ei-si)*(k/(num_interp+1)))*ones(numel(X1),1), X1, marg, "g");
            [X1, marg] = getWeightedMarginal1D(dlocs{k,p}(:,j), dmags{k,p}, nBins*1);
            marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
            plot3((k)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "b");
            fill3((k)*ones(numel(X1),1), X1, marg, "b", 'FaceAlpha', 0.5);
            if( ~any(dlocs{k,p}(:,j)) )
                line([(k),(k)], [0,0], [0,1000], 'Color', 'b');
            end
        end
        
        % Formatting
        % set(gca,'FontSize',30)
        set(gca,'YTick',[], 'ZTick', [])
        zlim([0 400]);
        xlim([1-0.002 num_interp+0.002]);
        view(-12,35);
        ylabel("$\xi_{" + num2str(j) + "}$")
        % xlabel("$\tau$")
        if ( j == 1 )
            title("CPU"+num2str(p),'FontSize',30);
            % title("$\hat{\mu}_{\hat{\tau}_j}$, $\mu_{\hat{\tau}_j}$ for $j\in[5]$");
            %     elseif ( j == 2 )
            %     	zticklabels(1/n*zticks);
            %     elseif ( j == 3 )
        end
        if ( j == 3 )
            set(gca,'XTick', 1:num_interp);
            xticklabels(1:num_interp);
        else
            set(gca,'XTick',[])
        end
        % zticklabels(1/n*zticks);
        hold off;
    end
end
% annotation('line', [0.5 0.5], [0.1 0.9], 'Color', 'k', 'LineWidth', 2);
% annotation('line', [0.1 0.9], [0.5 0.5], 'Color', 'k', 'LineWidth', 2);


%% Figure 6: 1D marginalized vs. interpolated, sep. graph for each one
%==========================================================================
nBins = 100;
for p=1:num_CPUs
    f5 = figure(5+p);
    set(gcf,'color','w');
    set(0, 'DefaultLineLineWidth', 1.5);
    for j=1:3
        %subplot(3,num_interp,); %(1+6*floor(p/3)+(1-mod(p,2)))+2*(j-1));
        
        for k=1:num_interp
            % Plot the measured intermediate distributions
            subplot(3,num_interp,(j-1)*(num_interp)+k);
            [X1, marg] = getMarginal1D(addD{k,p}(:,j), nBins);
            % marg = smoothdata(marg); 
            marg = smoothdata(normalize(marg,"norm")); 
            marg(1) = 0; marg(end) = 0;
            writematrix([X1' marg], out_dir + "measured_t" + num2str(k) + "_xi" + num2str(j) + "_CPU" + num2str(p) + ".txt");
            plot(X1(2:end-1), marg(2:end-1), "r");
            hold on;
            %fill(X1, marg, "r", 'FaceAlpha', 0.5);
            if( ~any(addD{k,p}(:,j)) )
                line([0,0], [0,1], 'Color', 'r');
            end
            
            % Plot interpolated distributions
            [X1, marg] = getWeightedMarginal1D(dlocs{k,p}(:,j), dmags{k,p}, nBins);
            % marg = smoothdata(marg); 
            marg = smoothdata(normalize(marg,"norm")); 
            marg(1) = 0; marg(end) = 0;
            writematrix([X1' marg], out_dir + "interpolated_t" + num2str(k) + "_xi" + num2str(j) + "_CPU" + num2str(p) + ".txt");
            plot(X1(2:end-1), marg(2:end-1), "b");
            % fill(X1, marg, "b", 'FaceAlpha', 0.5);
            if( ~any(dlocs{k,p}(:,j)) )
                line([0,0], [0,1], 'Color', 'b');
            end
            if( k == 1 )
                ylabel("$\xi_{" + num2str(j) + "}$", 'FontSize', 16)
            end
            if ( j == 1 )
                % title("CPU"+num2str(p),'FontSize',30);
                title("$\hat{\mu}_{\hat{\tau}_j}$, $\mu_{\hat{\tau}_j}$ for $j=" + num2str(k) + "$", 'FontSize', 16);
                %     elseif ( j == 2 )
                %     	zticklabels(1/n*zticks);
                %     elseif ( j == 3 )
            end
        end
        
        hold off;
    end
    sgtitle("CPU"+num2str(p),'FontSize',30)
end
