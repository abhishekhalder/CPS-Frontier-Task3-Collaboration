%
% Computes the projection of tensor (K .* U) onto the (t,j)'th marginal.
% marginals.
%
% @param T the number of barycenters
% @param J the number of spectral marginals
% @param (t,j) the marginal to project onto
% @param K, Kt correspond to K, \tilde{K} in Elvander et al, Proposition 4
% @param u needful terms
%
function Proj = Proj1_gridded(t, j, T, J, K, Kt, u)

p      = cell(T,1);
Proj_b = 1;
Proj_e = 1;

% Calculate p_t for all t\in[T]
for k=1:T
    p{k} = u{k,1};
    for l=2:J
        p{k} = p{k} .* ( Kt*u{k,l} );
    end
end

% Calculate Proj_b and Proj_e
if( t == 1 )
    Proj_e = Proj_e * K * p{T};
    for k=T-2:-1:1
        Proj_e = K * diag(p{k+1}) * Proj_e;
    end
elseif( t == T )
    Proj_b = p{1}' * K;
    for k=2:T-1
        Proj_b = Proj_b * diag(p{k}) * K;
    end
else
    Proj_b = p{1}' * K;
    for k=2:t-1
        Proj_b = Proj_b * diag(p{k}) * K;
    end
    Proj_e = Proj_e * K * p{T};
    for k=T-2:-1:t+1
        Proj_e = K * diag(p{k+1}) * Proj_e;
    end
end

if( j==1 )
    Proj = Proj_b' .* p{t} .* Proj_e;
else
    Proj = u{t,j} .* (Kt' * ( Proj_b' .* (p{t}./(Kt*u{t,j})) .* Proj_e ) );
end
