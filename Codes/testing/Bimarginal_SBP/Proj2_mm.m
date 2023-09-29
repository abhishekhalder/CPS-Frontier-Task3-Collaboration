%
% Computes the projection of tensor (K .* U) onto the t1'th and t2'th
% marginals. It is assumed that t1 < t2.
%
function P = Proj2_mm(t1, t2, K, u)

nM = numel(u);
n  = numel(u{1});

% Calculate P_b
if (t1 == 1)
    P_b = ones(n,1);
else
    P_b = u{1}' * K{1};
    for i=2:t1-1
        P_b = (P_b * diag(u{i})) * K{i};
    end
end

% Calculate P_m
P_m = diag(u{t1});
for i=t1+1:t2
    P_m = P_m * K{i-1} * diag(u{i});
end

% Calculate P_e
if (t2 == nM)
    P_e = ones(n,1);
else
    P_e = K{nM-1} * u{nM};
    for i=nM-1:-1:t2+1
        P_e = K{i-1} * (diag(u{i}) * P_e);
    end
end

% P = diag(P_b) .* P_m .* diag(P_e);
P = diag(P_b) * P_m * diag(P_e);