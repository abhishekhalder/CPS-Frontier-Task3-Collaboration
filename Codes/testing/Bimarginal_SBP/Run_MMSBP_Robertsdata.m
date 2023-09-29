close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
%% Problem parameters
% =========================================================================
dfile_pfix = "../halder_outfiles_0919/kbm_sim_32767_1080_0_t";
dfile_sfix = "_allstates.txt";
numMarginal = 6;
instr_scale = 1e-8;
llcloads_scale = 1e-6;
llcstores_scale = 1e-5;
llcmisses_scale = 1e-5;
epsilon = 0.1; % entropic regularization parameter
C  = cell(numMarginal-1,1);
K  = cell(numMarginal-1,1);
mu = cell(numMarginal,1);   % = mu_snapshots

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
    M = readmatrix(dfile_pfix + num2str(i) + dfile_sfix);
    M(:,1) = M(:,1) * instr_scale;
    M(:,2) = M(:,2) * llcloads_scale;
    M(:,3) = M(:,3) * llcstores_scale;
    M(:,4) = M(:,4) * llcmisses_scale;
    
    if( i~=0 )
        % C{i} = pdist2(X, [M(:,1) M(:,2) M(:,3) M(:,4)], 'squaredeuclidean');
        C{i} = pdist2(X, M, 'squaredeuclidean');
        K{i} = exp(-C{i}/epsilon); 
        % Elements of C are on the order of 1e14 -> e^(-1e14) \approx 0.
        % Thus K evaluates to the zero matrix.
        % We conclude that everything from here must be done in log scale.
    end
    
    % X = [M(:,1) M(:,2) M(:,3) M(:,4)];
    X = M;
    
    nSample = size(X,1); % We assume that all marginals have the same number of samples.
    n = nSample;         % Simplify for later
    
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
        
    % P_t = P_b' .* u{t} .* P_e;
    
    % u{t} = u{t} .* mu_snapshots{t} ./ P_t; 
    u{t} = mu{t} ./ (P_b' .* P_e);
    % u{t} = exp( log(mu_snapshots{t}) - log(P_b') - log(P_e) );

    err{t}{1}(end+1) = iter_idx;
    err{t}{2}(end+1) = HilbertProjectiveMetric(u{t},u_old);
    % err{t}{2}(end+1) = norm(u{t}-u_old);
    
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

for i=1:numMarginal-1
    disp(sum(Proj2_mm(i,i+1, K, u),"all"))
end

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
% figure(3)
% for k=1:numMarginal
%     subplot(1,numMarginal,k)
% 	plot(u{k})
%     set(gca,'FontSize',30)
%     xlabel('$i$','FontSize',30)
%     ylabel("$(u_{" + num2str(k) + "})_i$",'FontSize',30,'rotation',0)
%     %title(['$\mu_$' num2str(k)],'FontSize',30,'Interpreter','latex')
% end
