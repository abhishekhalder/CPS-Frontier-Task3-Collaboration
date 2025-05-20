function u = solve_path_sbp(K, dist, num_dist, n)
% Define K,
numMarginal_CPU = 1;

% Initialize U
u = cell(num_dist,1);
err = cell(num_dist,1);
for k=1:num_dist
	u{k,1} = rand(n,1);
    err{k,1} = { []; [] };
end

% Perform Sinkhorn iterations
maxIter = 10000; 
mintol = 1e-13; 
maxtol = 1e5;

iter_idx = 1;
t        = 1;
j        = 1;
tic;
while iter_idx <= maxIter
    fprintf('(Iter,t,j) = (%d, %d, %d)\n', iter_idx, t, j);
    
    u_old = u{t,j};
    
    % Calculate projection
    Proj = Proj1_scattered(t, num_dist, K, u);
    
    % Update iteration
    u{t,j} = u{t,j} .* dist{t,j} ./ Proj;
    
    % Calculate error
    err{t,j}{1}(end+1) = iter_idx;
    err{t,j}{2}(end+1) = max(1e-16, HilbertProjectiveMetric(u{t,j},u_old));
    
    disp(['Err ',num2str(err{t,j}{2}(end))])
    max_err = err{t,j}{2}(end);
    if (iter_idx >= num_dist*numMarginal_CPU)
        for k=1:num_dist
            if k==1 || k==num_dist
                max_err = max(max_err, err{k,1}{2}(end));
            else
                for l=1:numMarginal_CPU
                    max_err = max(max_err, err{k,l}{2}(end));
                end
            end
        end
    else
        max_err = 228;
    end
    disp(['Max_Err ',num2str(max_err)])
    
    % check convergence in Hilbert metric
    if (max_err < mintol)
        break;
    elseif( isinf(err{t,j}{2}(end)) || isnan(err{t,j}{2}(end)))
        fprintf('Error: NaN or Inf detected in Hilbert metric on iteration (t,j)=(%d,%d). Stopping...\n', t, j);
        break;
    else
        iter_idx = iter_idx+1;
        if j == numMarginal_CPU || (t==1 || t==num_dist)
            j = 1;
        else
            j = j + 1;
        end
        if j == 1
            t = mod(t, num_dist) + 1;
        end
    end
end

% Return M
% U = tensorprod(u{1},u{2},2,2); for i=3:num_dist; U = tensorprod(U,u{i}); end
% K = zeros(n*ones(num_dist,1));
% 
% 
% M = K .* U;
return


function Proj = Proj1_scattered(t,s,K,u)
	% Calculate projection
    P_b = 1; P_e = 1;
    if( t == 1 )
        P_e = P_e * K{s-1} * u{s};
        for k=s-2:-1:1
            P_e = K{k} * diag(u{k+1}) * P_e;
        end
    elseif( t == s ) 
        P_b = u{1}' * K{1}';
        for k=2:s-1
            P_b = P_b * diag(u{k}) * K{k}';
        end
    else
        P_b = u{1}' * K{1}';
        for k=2:t-1
            P_b = P_b * diag(u{k}) * K{k}';
        end
        P_e = P_e * K{s-1} * u{s};
        for k=s-2:-1:t+1
            P_e = K{k} * diag(u{k+1}) * P_e;
        end
    end
    Proj = P_b' .* u{t} .* P_e
return