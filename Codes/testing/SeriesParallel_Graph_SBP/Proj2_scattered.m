%
% Computes the projection of tensor (K .* U) onto two _adjacent_ marginals,
% where K is Series-Parallel structured.
%
% @param T the number of barycenters
% @param J the number of CPUs
% @param (tj1, tj2) tuples of the form [t,j]; the marginals to project between
% @param K the cell array of K.
% @param u needful terms
%
function Proj = Proj2_scattered(tj1, tj2, T, J, K, u)

t1 = tj1(1); t2 = tj2(1); j1 = tj1(2); j2 = tj2(2);

Proj_b = 1;     % w_1
Proj_m = 1;     % W_2
Proj_e = 1;     % w_3

% Handle the various cases
if( (t1 == 1 && j1 == 1) && (t2 == 2) ) % Projecting between marginals at 'fork'
    % Calculate W_2
    for k=1:j2-1
        Proj_m = Proj_m * K{1,k};
        for m=3:T-1
            Proj_m = Proj_m * diag(u{m-1,k}) * K{m-1,k};
        end
        Proj_m = Proj_m * diag(u{T-1,k}) * K{T-1,k};
    end
    Proj_m = Proj_m * K{1,j2};
    
    % Calculate w_3
    Proj_e = Proj_e * u{T,1};
    for k=J:-1:j2+1
        Proj_e = diag(u{T-1,k}) * K{T-1,k} * Proj_e;
        for m=T-1:-1:3
            Proj_e = diag(u{m-1,k}) * K{m-1,k} * Proj_e;
        end
        Proj_e = K{1,k} * Proj_e;
    end
    Proj_e = diag(u{T-1,j2}) * K{T-1,j2} * Proj_e;
    for m=T-1:-1:4
        Proj_e = diag(u{m-1,j2}) * K{m-1,j2} * Proj_e;
    end
    Proj_e = K{2,j2} * Proj_e;
    
elseif( (t1 == T-1 ) && (t2 == T && j2 == 1) ) % Projecting between marginals at 'join'
    % Calculate w_1
    Proj_b = Proj_b * u{1,1}';
    for k=1:j1-1
        Proj_b = Proj_b * K{1,k};
        for m=3:T-1
            Proj_b = Proj_b * diag(u{m-1,k}) * K{m-1,k};
        end
        Proj_b = Proj_b * diag(u{T-1,k}) * K{T-1,k};
    end
    Proj_b = Proj_b * K{1,j1};
    for m=3:T-1
        Proj_b = Proj_b * diag(u{m-1,j1}) * K{m-1,j1};
    end
    
	% Calculate W_2
    Proj_m = K{T-1,j1};
    for k=j1+1:J
        Proj_m = Proj_m * K{1,k};
        for m=3:T-1
            Proj_m = Proj_m * diag(u{m-1,k}) * K{m-1,k};
        end
        Proj_m = Proj_m * diag(u{T-1,k}) * K{T-1,k};
    end
    
elseif( (j1 == j2) && (t2 == t1+1) ) % Projecting between intermediate marginals along the same path
    % Calculate w_1
    Proj_b = Proj_b * u{1,1}';
    for k=1:j1-1
        Proj_b = Proj_b * K{1,k};
        for m=3:T-1
            Proj_b = Proj_b * diag(u{m-1,k}) * K{m-1,k};
        end
        Proj_b = Proj_b * diag(u{T-1,k}) * K{T-1,k};
    end
    Proj_b = Proj_b * K{1,j1};
    for m=3:t1
        Proj_b = Proj_b * diag(u{m-1,j1}) * K{m-1,j1};
    end
    
    % Calculate W_2
    Proj_m = K{t1,j1};
    
    % Calculate w_3
    Proj_e = Proj_e * u{T,1};
    for k=J:-1:j2+1
        Proj_e = diag(u{T-1,k}) * K{T-1,k} * Proj_e;
        for m=T-1:-1:3
            Proj_e = diag(u{m-1,k}) * K{m-1,k} * Proj_e;
        end
        Proj_e = K{1,k} * Proj_e;
    end
    Proj_e = diag(u{T-1,j2}) * K{T-1,j2} * Proj_e;
    for m=T-1:-1:t2+2 % = t_1 + 3
        Proj_e = diag(u{m-1,j2}) * K{m-1,j2} * Proj_e;
    end
    Proj_e = K{t2,j2} * Proj_e;
    
else
    fprintf('Error [Proj2]: Cannot project between non-adjacent marginals (%d,%d) and (%d,%d).\n', t1,j1, t2,j2);
    Proj = NaN;
    return;
end

Proj = diag(Proj_b' .* u{t1,j1}) * Proj_m' * diag(u{t2,j2} .* Proj_e);

return;
