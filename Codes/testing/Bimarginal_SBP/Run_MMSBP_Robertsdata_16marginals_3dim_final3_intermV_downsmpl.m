%
% This code runs the Sinkhorn iterations on Robert's data, converging to
% the solution vectors u_{i\in[numMarginal]}. We then use these vectors to
% determine T_\epsilon^{i\in[numMarginal-1]}(\xi), which performs
% single-sample prediction from \mu_i to \mu_{i+1}.
%
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
%% Problem parameters
% =========================================================================
numIntermediate = 4;
dfile_pfix = "../halder_outfiles_0927_intermarginals" ...
             + num2str(numIntermediate) + "/kbm_sim_32767_1080_0_t";
dfile_sfix = "_3dim.txt";
out_dir    = "./data_out/";
numMarginal     = 6 + 5*numIntermediate;
instr_scale     = 1e-8;
llcreq_scale    = 1e-6;
llcmiss_scale   = 1e-5;
epsilon = 0.1;                  % entropic regularization parameter
rawD = cell(numMarginal,1);     % raw data
C    = cell(numMarginal-1,1);
K    = cell(numMarginal-1,1);
mu   = cell(numMarginal,1);

%% Load MSBP data (Robert's data)
%
% We load state (sub)space data (instuctions retired, LLC loads) at each
% snapshot (i.e. at t=0 and at end of each control cycle) and fit to sum of
% Dirac distribution - these are our marginals (mu). We determine C^{(i)}
% for i\in\{1,\dots,numMarginal}, where C^{(i)}_{j,k} is the Euclidean
% distance between the j'th point in the i-1th marginal and the k'th point
% in the ith marginal. K's are then computed from C's.
%
% =========================================================================

for i=0:numMarginal-1
    rawD{i+1} = readmatrix(dfile_pfix + num2str(i) + dfile_sfix);
    rawD{i+1}(:,1) = rawD{i+1}(:,1) * instr_scale;
    rawD{i+1}(:,2) = rawD{i+1}(:,2) * llcreq_scale;
    rawD{i+1}(:,3) = rawD{i+1}(:,3) * llcmiss_scale;
    
    if( i~=0 )
        C{i} = pdist2(rawD{i}, rawD{i+1}, 'squaredeuclidean');
        K{i} = exp(-C{i}/epsilon); 
    end
    
    nSample = size(rawD{i+1},1);    % We assume that all marginals have the same number of samples.
    n = nSample;                    % Simplify for later
    
    mu{i+1} = 1/n * ones(n, 1);
end

%% Algorithm parameters
% ========================
maxIter = 1000; tol = 1e-10; maxtol = 1e5;
u = cell(numMarginal,1);
err = cell(numMarginal,1);
 for k=1:numMarginal
     u{k} = rand(n,1);
     err{k} = { []; [] };
 end
     
t = 1;
iter_idx = 1;
while iter_idx <= maxIter
    disp(['Iteration ',num2str(iter_idx)])
    
    u_old = u{t};
    
    % Calculate projection
    P_b = 1; P_e = 1;
    if( t == 1 )
        P_e = P_e * K{numMarginal-1} * u{numMarginal};
        for k=numMarginal-2:-1:1
            P_e = K{k} * diag(u{k+1}) * P_e;
        end
    elseif( t == numMarginal ) 
        P_b = u{1}' * K{1}';
        for k=2:numMarginal-1
            P_b = P_b * diag(u{k}) * K{k}';
        end
    else
        P_b = u{1}' * K{1}';
        for k=2:t-1
            P_b = P_b * diag(u{k}) * K{k}';
        end
        P_e = P_e * K{numMarginal-1} * u{numMarginal};
        for k=numMarginal-2:-1:t+1
            P_e = K{k} * diag(u{k+1}) * P_e;
        end
    end
    
    u{t} = mu{t} ./ (P_b' .* P_e);

    err{t}{1}(end+1) = iter_idx;
    err{t}{2}(end+1) = HilbertProjectiveMetric(u{t},u_old);
    
    disp(['Err ',num2str(err{t}{2}(end))])
    max_err = err{t}{2}(end);
    if (iter_idx >= numMarginal)
        for k=1:numMarginal
            max_err = max(max_err, err{k}{2}(end));
        end
    end
    disp(['Max_Err ',num2str(max_err)])
    
    % Check convergence in Hilbert metric
    if (max_err < tol)
         break;
    else        
        iter_idx = iter_idx+1;  
        t = mod(t, numMarginal) + 1;
    end
end


%% Find the transport maps between adjacent marginals
%========================================================
M = cell(numMarginal-1,1);

for i=1:numMarginal-1
    M{i} = Proj2_mm(i,i+1, K, u);
    disp(sum(M{i},"all"));
end

%% Figure 1 : Plot covergence in Hilbert metric
%==========================================================================
figure(1);
semilogy(err{1}{1}, err{1}{2}, 'LineWidth',2, 'color', 'r')
hold on;
for i=1:numMarginal
    semilogy(err{i}{1}, err{i}{2}, 'LineWidth',2)
end
set(gca,'FontSize',30)
xlabel('iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
hold off;

% Write errors to file
for i=1:numMarginal
    writecell(err{i}, out_dir + "f1_err" + num2str(i) + ".txt");
end


%% Figure 2: Plot 3D interpolated marginals between t_4 and t_5
%==========================================================================
figure(2);
alphaM = 0.2;
alphaI = 0.05;
x_lim  = [0.26 0.35];
y_lim  = [0.05 0.12];
z_lim  = [0.18 0.32];

% Set weights for all marginals
r = 1e-2;
weights = cell(numMarginal,1);
for i=1:numMarginal
    weights{i} = zeros(n,1);
    for j=1:n
        weights{i}(j) = sum( vecnorm(rawD{i}(j,:)-rawD{i},2,2) < r );
    end
    weights{i} = weights{i} / n;
end

l = 3;                              % Interpolate between t_l and t_{l+1}
num_interp = numIntermediate + 1;	% Number of locations to interpolate at.
si = 1+(l-1)*(numIntermediate+1);
ei = 1+( l )*(numIntermediate+1);

% Plot marginals at controller boundaries
subplot(2,2 + num_interp,[1 3 + num_interp]);
scatter3(rawD{si}(:,1), rawD{si}(:,2), rawD{si}(:,3), 40, weights{si}, 'filled');
% set(gca,'XLim',x_lim,'YLim',y_lim,'ZLim',z_lim);
% h = scatter3(rawD{si}(:,1), rawD{si}(:,2), rawD{si}(:,3), 40, 'filled');
% set(h, 'MarkerEdgeAlpha', alphaM, 'MarkerFaceAlpha', alphaM);
title("$\mu_{" + num2str(si) + "}$ ($t=\tau_{" + num2str(l-1) + "}$)",'FontSize',30);
subplot(2,2 + num_interp,[2 + num_interp, 2*(2 + num_interp)]);
h = scatter3(rawD{ei}(:,1), rawD{ei}(:,2), rawD{ei}(:,3), 40, weights{ei}, 'filled');
% set(gca,'XLim',x_lim,'YLim',y_lim,'ZLim',z_lim);
% h = scatter3(rawD{ei}(:,1), rawD{ei}(:,2), rawD{ei}(:,3), 40, 'filled');
% set(h, 'MarkerEdgeAlpha', alphaM, 'MarkerFaceAlpha', alphaM);
title("$\mu_{" + num2str(ei) + "}$ ($t=\tau_{" + num2str( l ) + "}$)",'FontSize',30);

%% Figure 2: Plot the measured intermediate distributions
addD = cell(num_interp,1);
for k=1:num_interp
    % addD{k} = readmatrix("../halder_outfiles_0927_intermarginals/kbm_sim_32767_1080_0_t" + num2str(4*(l-1)+k) + "_3dim.txt");
    % addD{k} = readmatrix("../halder_outfiles_0927_intermarginals4/kbm_sim_32767_1080_0_t" + num2str((num_interp+1)*(l-1)+k) + "_3dim.txt");
    addD{k} = readmatrix("../halder_outfiles_0927_intermarginals" + ...
                          num2str(num_interp) + "/kbm_sim_32767_1080_0_t" ...
                          + num2str((num_interp+1)*(l-1)+k) + "_3dim.txt");
    addD{k}(:,1) = addD{k}(:,1) * instr_scale;
    addD{k}(:,2) = addD{k}(:,2) * llcreq_scale;
    addD{k}(:,3) = addD{k}(:,3) * llcmiss_scale;
    subplot(2, 2+num_interp, 1+k);
    weightsD = zeros(n,1);
    for j=1:n
        weightsD(j) = sum( vecnorm(addD{k}(j,:)-addD{k},2,2) < r );
    end
    weightsD = weightsD / n;
    scatter3(addD{k}(:,1), addD{k}(:,2), addD{k}(:,3), 40, weightsD, 'filled');
    % set(gca,'XLim',x_lim,'YLim',y_lim,'ZLim',z_lim);
    title("$\mu_{" + num2str(si+(ei-si)*(k/(num_interp+1)),3) + "}$",'FontSize',30);
end

%% Figure 2: Plot interpolated distributions
M_int = Proj2_mm(si, ei, K, u);
disp(sum(M_int,"all"));

taus = (1:num_interp)/(num_interp+1);
ts   = (0:(numIntermediate+1))/(numIntermediate+1);
locs     = cell(num_interp, 1);      % Full-scale locs
mags     = cell(num_interp, 1);      % Full-scale mags
weights  = cell(num_interp, 1);      % Full-scale weights (for graphing)

nS       = 500;                     % Number of downsamples

dlocs    = cell(num_interp, 1);      % Downsampled locs
dmags    = cell(num_interp, 1);      % Downsampled mags
dweights = cell(num_interp, 1);      % Downsampled weights (for graphing)

for k=1:numel(taus)
    tau = taus(k);
    locs{k}     = zeros(n^2,3);
    mags{k}     = zeros(n^2,1);
    weights{k}  = zeros(n^2,1);
    dlocs{k}    = zeros(nS,3);
    dmags{k}    = zeros(nS,1);
    dweights{k} = zeros(nS,1);
    % Find correct M to use
    M_ind = si + (find(ts>tau,1)-2);
    % Find bounds in interpolation interval
    tau_int_b = ts(find(ts>tau,1)-1);
    tau_int_e = ts(find(ts>tau,1));
    tau_scaled = (tau-tau_int_b) / (tau_int_e-tau_int_b);
    for i=1:n
        for j=1:n
            locs{k}((i-1)*n+j,:) = (1-tau_scaled)*rawD{M_ind}(i,:)+tau_scaled*rawD{M_ind+1}(j,:);
            mags{k}((i-1)*n+j) = M{M_ind}(i,j);
        end
    end
    
    % Perform downsamping
    t_locs = locs{k};
    t_mags = mags{k};
    for i=1:nS
        disp(i);
        dlocs{k}(i,:) = t_locs(1,:);
        dmags{k}(i) = t_mags(1);
        t_locs(1,:) = [];
        t_mags(1) = [];
        [minValues,cIs] = mink(vecnorm(dlocs{k}(i,:)-t_locs,2,2), n^2/nS-1);
        for j=1:(n^2/nS-1)
            dlocs{k}(i,:) = (dmags{k}(i)*dlocs{k}(i,:) + t_mags(cIs(j)).*t_locs(cIs(j),:)) ...
                        / (dmags{k}(i)+t_mags(cIs(j)));
            dmags{k}(i) = dmags{k}(i) + t_mags(cIs(j));
        end
        t_locs(cIs,:) = [];
        t_mags(cIs)   = [];
    end
    for j=1:nS
        dweights{k}(j) = sum( (vecnorm(dlocs{k}(j,:)-dlocs{k},2,2) < r) .* dmags{k} );
    end
    weights{k} = weights{k} / nS;
    
    % This loop takes ~17 min to run.
%     for j=1:n^2
%         disp(j);
%         weights{k}(j) = sum( (vecnorm(locs{k}(j,:)-locs{k},2,2) < r) .* mags{k} );
%     end
    subplot(2, 2+num_interp, 3+num_interp+k);
	% scatter3(locs{k}(:,1), locs{k}(:,2), locs{k}(:,3), 1, weights{k}, 'filled');
	scatter3(dlocs{k}(:,1), dlocs{k}(:,2), dlocs{k}(:,3), 10, dweights{k}, 'filled');
    % set(gca,'XLim',x_lim,'YLim',y_lim,'ZLim',z_lim);
	% h = scatter3(locs{k}(:,1), locs{k}(:,2), locs{k}(:,3), 40, 'filled');
    % set(h, 'MarkerEdgeAlpha', alphaI, 'MarkerFaceAlpha', alphaI);
    disp(sum(mags{k}))
    title("$\hat{\mu}_{" + num2str(si+(ei-si)*(k/(num_interp+1)),3) + "}$",'FontSize',30);
end

%% Figure 3: Plot 1D interpolated marginals between t_4 and t_5
%==========================================================================
f3 = figure(3);
set(gcf,'color','w');
set(0, 'DefaultLineLineWidth', 2.5);
nBins = 100;
for j=1:3
    subplot(3,1,j);
    % Plot marginals at starting controller boundary
    [X1, marg] = getMarginal1D(rawD{si}(:,j), nBins);
    marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
    plot3(si*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "k");
    hold on;
    fill3(si*ones(numel(X1),1), X1, marg, "k", 'FaceAlpha', 0.5);
    
    % Plot the measured intermediate distributions
    for k=1:num_interp
        [X1, marg] = getMarginal1D(addD{k}(:,j), nBins);
        marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
        plot3((si+(ei-si)*(k/(num_interp+1))-0.001)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "r");
        fill3((si+(ei-si)*(k/(num_interp+1))-0.001)*ones(numel(X1),1), X1, marg, "r", 'FaceAlpha', 0.5);
    end
    
    % Plot interpolated distributions
    for k=1:num_interp
%         [X1, marg] = getWeightedMarginal1D(locs{k}(:,j), mags{k}, nBins*1);
%         marg = smoothdata(marg);
%         plot3((si+(ei-si)*(k/(num_interp+1)))*ones(numel(X1),1), X1, marg, "g");
        [X1, marg] = getWeightedMarginal1D(dlocs{k}(:,j), dmags{k}, nBins*1);
        marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
        plot3((si+(ei-si)*(k/(num_interp+1)))*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "b");
        fill3((si+(ei-si)*(k/(num_interp+1)))*ones(numel(X1),1), X1, marg, "b", 'FaceAlpha', 0.5);
    end
    
    % Plot marginals at end controller boundary
    [X1, marg] = getMarginal1D(rawD{ei}(:,j), nBins);
    marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
    plot3(ei*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "k");
    fill3(ei*ones(numel(X1),1), X1, marg, "k", 'FaceAlpha', 0.5);
    
    % Plot intermediate marginals (DEBUG)
%     for k=1:numIntermediate
%         [X1, marg] = getMarginal1D(rawD{si+k}(:,j), nBins);
%         marg = smoothdata(marg);
%         plot3((si+k)*ones(numel(X1),1), X1, marg, "g");
%     end

    set(gca,'FontSize',30)
    ylabel("$\xi_{" + num2str(j) + "}$")
    xlabel("$\sigma$")
    if ( j == 1 )
        title("$\hat{\mu}_j$, $\mu_{\hat{\tau}_j}$ for $j\in[5]$");
%     elseif ( j == 2 )
%     	zticklabels(1/n*zticks);
%     elseif ( j == 3 )
    end
	zticklabels(1/n*zticks);
    hold off;
end

%% Table 1: Compute Wasserstein distances.
% Get Entropy-Regularized Wasserstein distances between interpolations
% and measured.
epsilon_wass = 0.01;
wass  = zeros(num_interp,1);
disp("------");
for k=1:num_interp
    % wass(k) = sqrt( EntropyRegularizedOMT(addD{k},locs{k},1/n*ones(n,1),mags{k},epsilon_wass) );
    % wass(k) = ws_distance(addD{k},locs{k},2);
    wass(k) = Wasserstein(addD{k}, dlocs{k}, 1/n*ones(n,1), dmags{k});
    disp(wass(k));
end
% disp("------------");
% sqrt( EntropyRegularizedOMT(addD{k},addD{k},1/n*ones(n,1),1/n*ones(n,1),epsilon_wass) )

% ws_distance(addD{k},locs{k},1)
