% p and q are assumed to be log scaled, i.e. 
% HilbertProjectiveMetric_logscale(p,q) = HilbertProjectiveMetric(exp(p),exp(q))
% HilbertProjectiveMetric_logscale(log(p),log(q)) = HilbertProjectiveMetric(p,q)
%
function d_Hilbert = HilbertProjectiveMetric_logscale(p,q)

x = p-q;
a = min(x);
b = max(x);

d_Hilbert = b - a;