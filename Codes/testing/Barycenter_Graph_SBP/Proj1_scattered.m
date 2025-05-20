%
% Computes the projection of tensor (K .* U) onto the (t,j)'th marginal.
% marginals.
%
% @param T the number of barycenters
% @param J the number of spectral marginals
% @param (t,j) the marginal to project onto
% @param K the cell array of K, K_tilde matrices.
% @param u needful terms
%
function Proj = Proj1_scattered(t, j, T, J, K, u)

p      = cell(T,1);
Proj_b = 1;
Proj_e = 1;

% Calculate p_t for all t\in[T]
for k=1:T
    p{k} = u{k,1};
    for l=2:J
        p{k} = p{k} .* ( K{k,l}*u{k,l} );
    end
end

% Calculate Proj_b and Proj_e
if( t == 1 )
    Proj_e = (Proj_e * K{T-1,1}) * p{T};
    for k=T-2:-1:1
        Proj_e = K{k,1} * (diag(p{k+1}) * Proj_e);
    end
elseif( t == T )
    Proj_b = p{1}' * K{1,1};
    for k=2:T-1
        Proj_b = (Proj_b * diag(p{k})) * K{k,1};
    end
else
    Proj_b = p{1}' * K{1,1};
    for k=2:t-1
        Proj_b = (Proj_b * diag(p{k})) * K{k,1};
    end
    Proj_e = (Proj_e * K{T-1}) * p{T};
    for k=T-2:-1:t+1
        Proj_e = K{k,1} * (diag(p{k+1}) * Proj_e);
    end
end

if( j==1 )
    Proj = Proj_b' .* p{t} .* Proj_e;
else
    Proj = u{t,j} .* (K{t,j}' * ( Proj_b' .* (p{t}./(K{t,j}*u{t,j})) .* Proj_e ) );
end
