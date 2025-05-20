%
% Computes the projection of tensor (K .* U) onto the (t,j)'th marginal.
% marginals, where K is Series-Parallel structured.
%
% @param T the number of barycenters
% @param J the number of CPUs
% @param (t,j) the marginal to project onto
% @param K the cell array of K matrices.
% @param u needful terms
%
function Proj = Proj1_scattered(t, j, T, J, K, u)

Proj_b = 1;     % w_1
Proj_e = 1;     % w_2

% Calculate Proj_b and Proj_e
if( t == 1 )        % In this case, we care only about j=1
    Proj_e = Proj_e * u{T,1};
    for k=J:-1:1
        Proj_e = diag(u{T-1,k}) * (K{T-1,k} * Proj_e);
        for m=T-1:-1:3
            Proj_e = diag(u{m-1,k}) * (K{m-1,k} * Proj_e);
        end
        Proj_e = K{1,k} * Proj_e;
    end
elseif( t == T )    % In this case, we care only about j=1
    Proj_b = Proj_b * u{1,1}';
    for k=1:J
        Proj_b = Proj_b * K{1,k};
        for m=3:T-1
            Proj_b = (Proj_b * diag(u{m-1,k})) * K{m-1,k};
        end
        Proj_b = (Proj_b * diag(u{T-1,k})) * K{T-1,k};
    end
else
    Proj_b = Proj_b * u{1,1}';
    for k=1:j-1
        Proj_b = Proj_b * K{1,k};
        for m=3:T-1
            Proj_b = (Proj_b * diag(u{m-1,k})) * K{m-1,k};
        end
        Proj_b = (Proj_b * diag(u{T-1,k})) * K{T-1,k};
    end
	Proj_b = Proj_b * K{1,j};
    for m=3:t
        Proj_b = (Proj_b * diag(u{m-1,j})) * K{m-1,j};
    end
    
    Proj_e = Proj_e * u{T,1};
    for k=J:-1:j+1
        Proj_e = diag(u{T-1,k}) * (K{T-1,k} * Proj_e);
        for m=T-1:-1:3
            Proj_e = diag(u{m-1,k}) * (K{m-1,k} * Proj_e);
        end
        Proj_e = K{1,k} * Proj_e;
    end
    Proj_e = K{j,1} * Proj_e;
    for m=T-1:-1:t+1
        Proj_e =  K{m-1,j} * (diag(u{m-1,j}) * Proj_e);
    end
end

Proj = Proj_b' .* u{t,j} .* Proj_e;
