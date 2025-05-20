%
% Computes the projection of tensor (K .* U) onto two _adjacent_ marginals,
% where K is Series-Parallel structured.
% 
% Fixed on 02/12/25 to correct for mistake in derivations
%
% @param T the number of barycenters
% @param J the number of CPUs
% @param (tj1, tj2) tuples of the form [t,j]; the marginals to project between
% @param K the cell array of K.
% @param u needful terms
%
function Proj = Proj2_scattered_fixed(tj1, tj2, T, J, K, u)

t1 = tj1(1); t2 = tj2(1); j1 = tj1(2); j2 = tj2(2);

Proj_b = 1;     % w_1
Proj_m = 1;     % W_2
Proj_e = 1;     % w_3

% Handle the various cases
if( (t1 == 1 && j1 == 1) && (t2 == 2) ) % Projecting between marginals at 'fork'
    % Calculate W_2
    B = getB(j2, T, J, K, u);
    Y = getY(2,j2, T, J, K, u);
    Proj_m = K{1,j2} .* ( B*diag(u{T,1})*Y' );
    
elseif( (t1 == T-1 ) && (t2 == T && j2 == 1) ) % Projecting between marginals at 'join'
    % Calculate W_2
    B = getB(j1, T, J, K, u);
    X = getX(t1,j1, T, J, K, u);
    Proj_m = K{t1,j1} .* ( X'*diag(u{1,1})*B );
    
elseif( (j1 == j2) && (t2 == t1+1) ) % Projecting between intermediate marginals along the same path
    % Calculate W_2
    B = getB(j1, T, J, K, u);
    X = getX(t1,j1, T, J, K, u);
    Y = getY(t2,j1, T, J, K, u);
    Proj_m = K{t1,j1} .* ( X'*diag(u{1,1})*B*diag(u{T,1})*Y' );
    
else
    fprintf('Error [Proj2]: Cannot project between non-adjacent marginals (%d,%d) and (%d,%d).\n', t1,j1, t2,j2);
    Proj = NaN;
    return;
end

Proj = diag(Proj_b' .* u{t1,j1}) * Proj_m' * diag(u{t2,j2} .* Proj_e);

return;


% -------------------------------------------------------------------------

function X = getX(t, j, T, J, K, u)
    X = K{1,j};
    for m=2:t-1
        X = X * ( diag(u{m,j})*K{m,j} );
    end
return


function Y = getY(t, j, T, J, K, u)
    Y = K{t,j};
    for m=t+1:T-1
        Y = Y * ( diag(u{m,j})*K{m,j} );
    end
return


function B = getB(j, T, J, K, u)
    kvals = setdiff(1:J,j);
    
    % Calculate relevant A_k values
    A = cell(J-1,1);
    % for k=1:J
    for k=1:numel(kvals) % (= J-1)
        kval = kvals(k);
        A{k} = K{1,kval};
        for m=2:T-1
            A{k} = A{k} * ( diag(u{m,kval})*K{m,kval} );
        end
    end
    
    B = 1; for k=1:numel(kvals) B = B .* A{k}; end
return


