%
% This code runs the Sinkhorn iterations on Robert's data, converging to
% the solution vectors u_{i\in[numMarginal]}. We then use these vectors to
% determine T_\epsilon^{i\in[numMarginal-1]}(\xi), which performs
% single-sample prediction from \mu_i to \mu_{i+1}.
%
% This code uses \xi=[inst. ret., LLC requests, LLC misses] \in\R^3
%
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
%% Problem parameters
% =========================================================================
dfile_pfix = "../halder_outfiles_0922/kbm_sim_32767_1080_0_t";
dfile_sfix = "_3dim.txt";
numMarginal = 6;
instr_scale = 1e-8;
llcreq_scale = 1e-6;
llcmiss_scale = 1e-5;
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
    % rawD{i+1} = rawD{i+1}(:,1);
    rawD{i+1}(:,1) = rawD{i+1}(:,1) * instr_scale;
    rawD{i+1}(:,2) = rawD{i+1}(:,2) * llcreq_scale;
    rawD{i+1}(:,3) = rawD{i+1}(:,3) * llcmiss_scale;
    
    if( i~=0 )
        C{i} = pdist2(rawD{i}, rawD{i+1}, 'squaredeuclidean');
        K{i} = exp(-C{i}/epsilon); 
    end
    
    nSample = size(rawD{i+1},1);	% We assume that all marginals have the same number of samples.
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

%% Determine the mappings T_\epsilon^{i\in[numMarginals]}
% ============================================================
T = cell(numMarginal-1,1);

for j=1:numMarginal-1
    % xi is assumed to be a row vector of size 4x1
    T{j} = @(xi) sum( rawD{j+1}.*u{j+1}.*exp(-1/(2*epsilon)*vecnorm(xi-rawD{j+1},2,2).^2),1 ) ...
           / sum( u{j+1}.*exp(-1/(2*epsilon)*vecnorm(xi-rawD{j+1},2,2).^2) );
end

%% Find the transport maps between adjacent marginals
%========================================================
M = cell(numMarginal-1,1);

for i=1:numMarginal-1
    M{i} = Proj2_mm(i,i+1, K, u);
    disp(sum(M{i},"all"));
end

%% Plots
%========
close all;
nBins = 100;
r = 1e-2;

% Plot all marginals
figure(1);
weights = cell(numMarginal,1);
for i=1:numMarginal
    weights{i} = zeros(n,1);
    for j=1:n
        weights{i}(j) = sum( vecnorm(rawD{i}(j,:)-rawD{i},2,2) < r );
    end
    weights{i} = weights{i} / n;

    subplot(1,numMarginal,i);
    scatter3(rawD{i}(:,1), rawD{i}(:,2), rawD{i}(:,3), 40, weights{i}, 'filled');
    title("$\mu_" + num2str(i) + "$",'FontSize',30);
end

%% Plot two marginals, interpolated distributions between them, and compare
%  against measured distributions at interpolated times.

l = 2;              % Interpolate between t_l and t_{l+1}
num_interp = 3;     % Number of locations to interpolate at.

figure(2);

% Plot the 2 marginals
subplot(2,2 + num_interp,1);
scatter3(rawD{l}(:,1), rawD{l}(:,2), rawD{l}(:,3), 40, weights{l}, 'filled');
title("$\mu_" + num2str(l) + "$",'FontSize',30);
subplot(2,2 + num_interp,2 + num_interp);
scatter3(rawD{l+1}(:,1), rawD{l+1}(:,2), rawD{l+1}(:,3), 40, weights{l+1}, 'filled');
title("$\mu_" + num2str(l+1) + "$",'FontSize',30);

% Plot the measured intermediate distributions

% Plot the interpolated distributions
taus = (1:num_interp)/(num_interp+1);
locs    = cell(num_interp, 1);
mags    = cell(num_interp, 1);
weights = cell(num_interp, 1);
for k=1:numel(taus)
    tau = taus(k);
    locs{k}    = zeros(n^2,3);
    mags{k}    = zeros(n^2,1);
    weights{k} = zeros(n^2,1);
    for i=1:n
        for j=1:n
            locs{k}((i-1)*n+j,:) = (1-tau)*rawD{l}(i,:)+tau*rawD{l+1}(j,:);
            mags{k}((i-1)*n+j) = M{l}(i,j);
        end
    end
    % This loop takes ~17 min to run.
%     for j=1:n^2
%         disp(j);
%         weights(j) = sum( (vecnorm(locs(j,:)-locs,2,2) < r) .* mags );
%     end
%     subplot(2, 2+num_interp, 3+num_interp+k);
%     scatter3(locs(:,1), locs(:,2), locs(:,3), 1, weights, 'filled');
%     disp(sum(mags))
end

%%
addD = cell(3,1);
for l=1:3
    addD{l} = readmatrix("../halder_outfiles_0922_intermarginals/kbm_sim_32767_1080_0_t" + num2str(4+l) + "_3dim.txt");
    addD{l}(:,1) = addD{l}(:,1) * instr_scale;
    addD{l}(:,2) = addD{l}(:,2) * llcreq_scale;
    addD{l}(:,3) = addD{l}(:,3) * llcmiss_scale;
    subplot(2, 2+num_interp, 1+l);
    weights = zeros(n,1);
    for j=1:n
        weights(j) = sum( vecnorm(addD{l}(j,:)-addD{l},2,2) < r );
    end
    weights = weights / n;
    scatter3(addD{l}(:,1), addD{l}(:,2), addD{l}(:,3), 40, weights, 'filled');
    title("$\mu_{2." + num2str(25*l) + "}$",'FontSize',30);
end

%%
% interp1 = axObjs(4).Children;
% interp1 = 

%% Downsample locs{1}
nS = 1000;
dlocs = cell(num_interp,1);
dmags = cell(num_interp,1);
for l=1:num_interp
    dlocs{l} = zeros(nS,3);
    dmags{l} = zeros(nS,1);
    t_locs = locs{l};
    t_mags = mags{l};
    for i=1:nS
        disp(i);
        dlocs{l}(i,:) = t_locs(1,:);
        dmags{l}(i) = t_mags(1);
        t_locs(1,:) = [];
        t_mags(1) = [];
        [minValues,cIs] = mink(vecnorm(dlocs{l}(i,:)-t_locs,2,2), n^2/nS-1);
        for j=1:(n^2/nS-1)
            dlocs{l}(i,:) = (dmags{l}(i)*dlocs{l}(i,:) + t_mags(cIs(j)).*t_locs(cIs(j),:)) ...
                        / (dmags{l}(i)+t_mags(cIs(j)));
            dmags{l}(i) = dmags{l}(i) + t_mags(cIs(j));
        end
        t_locs(cIs,:) = [];
        t_mags(cIs)   = [];
    end
    weights = zeros(nS,1);
    subplot(2,2 + num_interp,3+num_interp + l);
    for j=1:nS
        weights(j) = sum( (vecnorm(dlocs{l}(j,:)-dlocs{l},2,2) < r) .* dmags{l} );
    end
    weights = weights / nS;
	scatter3(dlocs{l}(:,1), dlocs{l}(:,2), dlocs{l}(:,3), 20, weights, 'filled');
end

for l=1:3
    disp( sqrt( Wasserstein(dlocs{l}, addD{l}, dmags{l}, 1/n*ones(n,1)) ) );
end
    
%%
l = 2; 
nBins = 100;
figure(6);
for i=1:num_interp
    subplot(2,num_interp,num_interp+i);
    [X1,marg] = getWeightedMarginal1D(dlocs{l}(:,i),dmags{l},nBins); plot(X1, marg, 'color', 'g', 'linewidth', 2);
    hold on;
    [X1,marg] = getMarginal1D(dlocs{l}(:,i),nBins); plot(X1, marg, 'color', 'k');
    hold off;
    title("$\hat{\mu}_{2." + num2str(25*l) + "}$",'FontSize',30);
    subplot(2,num_interp,i);
    [X1,marg] = getMarginal1D(addD{l}(:,i),nBins); plot(X1, marg);
    title("$\mu_{2." + num2str(25*l) + "}$",'FontSize',30);
end

%%
% figure(5);
% for l=1:num_interp
%     subplot(1,num_interp,l);
%     scatter3(dlocs{l}(:,1), dlocs{l}(:,2), dlocs{l}(:,3), 40, weights, 'filled');
% end


% f4 = figure(4);
% % Plot marginal distributions
% for i=2:numMarginal
%     [f,xi] = ksdensity(rawD{i}, 'Weights', mu{i}, 'Bandwidth', 0.0003);
%     plot3((i-1)*ones(size(xi)), xi, f, 'color', 'k');
%     hold on;
% end
% 

% % Calculate intermediate distributions
% for l=2:numMarginal-1
%     taus = [1/3 2/3];
%     for k=1:numel(taus)
%         tau = taus(k);
%         locs = zeros(n^2,1);
%         mags = zeros(n^2,1);
%         for i=1:n
%             for j=1:n
%                 locs((i-1)*n+j) = (1-tau)*rawD{l}(i)+tau*rawD{l+1}(j);
%                 mags((i-1)*n+j) = M{l}(i,j);
%             end
%         end
%         % scatter(tau*ones(numel(X)*numel(Y),1), locs, mags)
%         [f,xi] = ksdensity(locs, 'Weights', mags, 'Bandwidth', 0.001^2);
%         plot3(((l-1)+tau)*ones(size(xi)), xi, f, 'color', 'b');
%         disp(sum(mags))
%     end
% end
% xlabel("t");
% zlim([0 150]);
% hold off;

% Plot stuff
% %% =======================================
% % plot the endpoint measures
% % figure(1)
% % for k=1:numMarginal
% %     subplot(1,numMarginal,k)
% %     contourf(X,Y,reshape(mu_snapshots{k},size(X)))
% %     set(gca,'FontSize',30)
% %     xlabel('$x_{1}$','FontSize',30)
% %     ylabel('$x_{2}$','FontSize',30,'rotation',0)
% %     %title(['$\mu_$' num2str(k)],'FontSize',30,'Interpreter','latex')
% % end
% 
% % plot the covergence in Hilbert metric
% figure(2)
% % error = 0;
% % for k = 1:numMarginal
% %     error = error + err{k}
% % end
% % semilogy(error,'LineWidth',2)
% semilogy(err{1}{1}, err{1}{2}, 'LineWidth',2, 'color', 'r')
% hold on;
% semilogy(err{2}{1}, err{2}{2}, 'LineWidth',2, 'color', 'g')
% semilogy(err{3}{1}, err{3}{2}, 'LineWidth',2, 'color', 'b')
% semilogy(err{4}{1}, err{4}{2}, 'LineWidth',2, 'color', 'k')
% semilogy(err{5}{1}, err{5}{2}, 'LineWidth',2, 'color', 'y')
% semilogy(err{6}{1}, err{6}{2}, 'LineWidth',2, 'color', 'c')
% % for k=2:numMarginal
% %     % semilogy(err{k},'LineWidth',2)
% %     semilogy(err{k}{1}, err{k}{2}, 'LineWidth',2)
% % end
% set(gca,'FontSize',30)
% xlabel('iteration index $j$','FontSize',30)
% ylabel('Error','FontSize',30)
% % legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')
% hold off;
% 

% % Plot u_{i\in[numMarginal]}
% figure(3)
% for k=1:numMarginal
%     subplot(1,numMarginal,k)
% 	plot(u{k})
%     set(gca,'FontSize',30)
%     xlabel('$i$','FontSize',30)
%     ylabel("$(u_{" + num2str(k) + "})_i$",'FontSize',30,'rotation',0)
%     %title(['$\mu_$' num2str(k)],'FontSize',30,'Interpreter','latex')
% end

% f4 = figure(4);
% for k=1:numMarginal
% 	scatter((k-1)*ones(size(rawD{k})), rawD{k});
%     hold on;
% end



% pathX = 0:numMarginal-1;
% for k=1:n
%     pathY = zeros(numMarginal);
%     pathY(1) = rawD{1}(k);
%     
%     for j=1:numMarginal-1
%         pathY(j+1) = T{j}(pathY(j));
%     end
%     
% 	plot(pathX, pathY);
% end

% pathX = 1:numMarginal-1;
% for k=1:n
%     pathY = zeros(numMarginal-1);
%     pathY(1) = M{2}(k);
%     
%     for j=2:numMarginal-1
%         pathY(j) = T{j-1}(pathY(j-1));
%     end
%     
% 	plot(pathX, pathY);
%     % set(gca,'FontSize',30)
%     % xlabel('$i$','FontSize',30)
%     % ylabel("$(u_{" + num2str(k) + "})_i$",'FontSize',30,'rotation',0)
%     % title(['$\mu_$' num2str(k)],'FontSize',30,'Interpreter','latex')
% end
% xlabel('$i-1$','FontSize',10)
% ylabel("$\xi_1$",'FontSize',10,'rotation',0)
% title('Known values of $\xi_1$ at $t_i$ (scatter) and those predicted by $T_\epsilon^i$','FontSize',10,'Interpreter','latex')
% hold off;
