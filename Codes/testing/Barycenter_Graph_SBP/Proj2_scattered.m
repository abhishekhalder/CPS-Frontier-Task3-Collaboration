%
% Computes the projection of tensor (K .* U) onto the two marginals.
%
% @param T the number of barycenters
% @param J the number of spectral marginals
% @param (tj1, tj2) tuples of the form [t,j]; the marginals to project between
% @param K the cell array of K, K_tilde matrices.
% @param u needful terms
%
function Proj = Proj2_scattered(tj1, tj2, T, J, K, u)

p = cell(T,1);
t1 = tj1(1); t2 = tj2(1); j1 = tj1(2); j2 = tj2(2);

% Calculate p_t for all t\in[T]
for k=1:T
    p{k} = u{k,1};
    for l=2:J
        p{k} = p{k} .* ( K{k,l}*u{k,l} );
    end
end

% Handle the various cases
if( (t1 == t2) && (j1 == 1) ) % Projecting between barycenter and spectral marginal at same time
    t = t1; j = j2;
    
    % Calculate rho_b and rho_e
    rho_b = 1;
    rho_e = 1;
    if( t == 1 )
        rho_e = rho_e * K{T-1,1} * p{T};
        for k=T-2:-1:1
            rho_e = K{k,1} * diag(p{k+1}) * rho_e;
        end
    elseif( t == T )
        rho_b = p{1}' * K{1,1};
        for k=2:T-1
            rho_b = rho_b * diag(p{k}) * K{k,1};
        end
    else
        rho_b = p{1}' * K{1,1};
        for k=2:t-1
            rho_b = rho_b * diag(p{k}) * K{k,1};
        end
        rho_e = rho_e * K{T-1} * p{T};
        for k=T-2:-1:t+1
            rho_e = K{k,1} * diag(p{k+1}) * rho_e;
        end
    end
    
    rho = rho_b' .* ( p{t}./(u{t,1}.*(K{t,j}*u{t,j})) ) .* rho_e;
    
    % Proj = diag(u{t,1}) * diag(K{t,j}'*rho) * K{t,j} * diag(u{t,j}); % ORIG
    Proj = diag(u{t,1}) * diag(rho) * K{t,j} * diag(u{t,j});
elseif( (j1 == 1) && (j2 == 1) ) % Projecting between barycenters
    Proj = Proj2_sequential(t1, t2, K(1:end-1,1), p);
else
    fprintf('Error [Proj2]: Cannot project between non-connected marginals (%d,%d) and (%d,%d).\n', t1,j1, t2,j2);
    Proj = NaN;
end

return;
