%
% Computes the projection of tensor (K .* U) onto the (t,j)'th marginal.
% marginals, where K is Series-Parallel structured.
%
% Fixed on 02/12/25 to correct for mistake in derivations
%
% @param T the number of barycenters
% @param J the number of CPUs
% @param (t,j) the marginal to project onto
% @param K the cell array of K matrices.
% @param u needful terms
%
function Proj = Proj1_scattered_fixed(t, j, T, J, K, u)

Proj_b = 1;     % w_1
Proj_e = 1;     % w_2

% Calculate A_k
A = cell(J,1);
for k=1:J
    A{k} = K{1,k};
    for m=2:T-1
        A{k} = A{k} * ( diag(u{m,k})*K{m,k} );
    end
end

% Calculate Proj_b and Proj_e
if( t == 1 )        % In this case, we care only about j=1
    for k=1:J
        Proj_e = Proj_e .* A{k};
    end
    Proj_e = Proj_e * u{T,1};
elseif( t == T )    % In this case, we care only about j=1
    for k=1:J
        Proj_b = Proj_b .* A{k};
    end
    Proj_b = u{1,1}' * Proj_b;
else
    X = K{1,j}; for m=2:t-1 X = X * (diag(u{m,j})*K{m,j}); end
    Y = K{t,j}; for m=t+1:T-1 X = X * (diag(u{m,j})*K{m,j}); end
    B = 1; for k=setdiff(1:J,j) B = B .* A{k}; end
    Proj_e = diag( diag(u{1,1})*X*Y*B'*diag(u{T,1}) );
end

Proj = Proj_b' .* u{t,j} .* Proj_e;
