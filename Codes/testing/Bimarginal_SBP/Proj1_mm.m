%
% Computes the projection of tensor (K .* U) onto the t'th marginal.
% marginals. It is assumed that t \in [nN].
%
function P = Proj1_mm(t, K, u)

P = [];