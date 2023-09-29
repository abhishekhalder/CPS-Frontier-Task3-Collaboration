close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
%% Load MSBP data (Robert's data)
%
% We load state (sub)space data (instuctions retired, LLC loads) at each
% snapshot (i.e. at t=0 and at end of each control cycle) and fit to sum 
%
% =========================================================================
numMarginal = 6;

% state space
xmin = -1; xmax = 1; dx = 0.1; x = xmin:dx:xmax;
y = x;
[X,Y] = ndgrid(x,y); XX = [X(:) Y(:)]; nSample = size(XX,1);
%% Problem parameters
% ========================
epsilon = 0.1; % entropic regularization parameter
C = pdist2(XX,XX,'squaredeuclidean');
% Gamma = exp(-C/epsilon); % Gamma := K
K = exp(-C/epsilon);
mu_snapshots = cell(numMarginal,1);

mean0 = [0.1; 0.1];
 
Cov0 = [1.25 0.05;
    0.05 1.01];
A = [0.4 -0.1;
    2 0.6];
% endpoint measures
mu_snapshots{1} = mvnpdf(XX,mean0',Cov0);
for k=1:numMarginal
    M = A^k;
    mu_snapshots{k} = mvnpdf(XX,(M*mean0)',M*Cov0*M');
    mu_snapshots{k} = mu_snapshots{k}/sum(mu_snapshots{k});
end

%% Algorithm parameters
% ========================
maxIter = 1000; tol = 1e-3; maxtol = 1e5;
u = cell(numMarginal,1);
err = cell(numMarginal,1);
% u0 = [rand(nSample,1), zeros(nSample,maxIter)];
% u1 = [rand(nSample,1), zeros(nSample,maxIter)];
 for k=1:numMarginal
     u{k} = rand(nSample,1);
     u{k} = u{k}/norm(u{k});
     % err{k} = zeros(maxIter,1);
     err{k} = { []; [] };
 end
     
% er0 = zeros(maxIter,1); er1 = zeros(maxIter,1);
% 
%
% tic;
iter_idx = 1;
t = 1;
while iter_idx <= maxIter
    disp(['Iteration ',num2str(iter_idx)])
    
    u_old = u{t};
    
    % Calculate projection
    P_b = 1; P_e = 1;
    if( t == 1 )
        for k=1:numMarginal-1
            P_e = P_e * K * diag(u{k});
        end
        P_e = P_e * K * u{numMarginal};
    elseif( t == numMarginal ) 
        P_b = u{1}' * K;
        for k=2:numMarginal-1
            P_b = P_b * diag(u{k}) * K;
        end
    else
        P_b = u{1}' * K;
        for k=2:t-1
            P_b = P_b * diag(u{k}) * K;
        end
        for k=t+1:numMarginal-1
            P_e = P_e * K * diag(u{k});
        end
        P_e = P_e * K * u{numMarginal};
    end
        
    % P_t = P_b' .* u{t} .* P_e;
    
    % u{t} = u{t} .* mu_snapshots{t} ./ P_t; 
    % u{t} = mu_snapshots{t} ./ (P_b' .* P_e);
    u{t} = exp( log(mu_snapshots{t}) - log(P_b') - log(P_e) );

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
    
    % check convergence in Hilbert metric
    if (max_err < tol)
    % if (err{t}{2}(end) < tol | err{t}{2}(end) > maxtol | isnan(err{t}{2}(end)) )
         break;
    else        
        % disp(['Err ',num2str(err{t}(iter_idx))])
        iter_idx = iter_idx+1;  
        t = mod(t, numMarginal) + 1;
    end
end
% comptime_recursion = toc
% Plot stuff
%% =======================================
% plot the endpoint measures
figure(1)
for k=1:numMarginal
    subplot(1,numMarginal,k)
    contourf(X,Y,reshape(mu_snapshots{k},size(X)))
    set(gca,'FontSize',30)
    xlabel('$x_{1}$','FontSize',30)
    ylabel('$x_{2}$','FontSize',30,'rotation',0)
    %title(['$\mu_$' num2str(k)],'FontSize',30,'Interpreter','latex')
end

% plot the covergence in Hilbert metric
figure(2)
% error = 0;
% for k = 1:numMarginal
%     error = error + err{k}
% end
% semilogy(error,'LineWidth',2)
semilogy(err{1}{1}, err{1}{2}, 'LineWidth',2, 'color', 'r')
hold on;
semilogy(err{2}{1}, err{2}{2}, 'LineWidth',2, 'color', 'g')
semilogy(err{3}{1}, err{3}{2}, 'LineWidth',2, 'color', 'b')
semilogy(err{4}{1}, err{4}{2}, 'LineWidth',2, 'color', 'k')
semilogy(err{5}{1}, err{5}{2}, 'LineWidth',2, 'color', 'y')
% semilogy(err{6}{1}, err{6}{2}, 'LineWidth',2, 'color', 'c')
% for k=2:numMarginal
%     % semilogy(err{k},'LineWidth',2)
%     semilogy(err{k}{1}, err{k}{2}, 'LineWidth',2)
% end
set(gca,'FontSize',30)
xlabel('iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
% legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')
hold off;

