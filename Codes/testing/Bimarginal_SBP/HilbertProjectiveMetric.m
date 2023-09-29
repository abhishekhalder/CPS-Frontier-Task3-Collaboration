function d_Hilbert = HilbertProjectiveMetric(p,q)

x = p./q;
a = min(x);
b = max(x);

d_Hilbert = log(b/a);