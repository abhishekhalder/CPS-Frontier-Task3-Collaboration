function margPF = get_context_marg_iCPUs(X1,locs,PDF,nBins)

dx1 = mean(diff(X1));
A = dx1;
[x1] = ndgrid(X1);
margPF = zeros(nBins,1);

for i=1:nBins
    ll = find( locs(:,4) == x1(i) );
    margPF(i) = sum(PDF(ll));
end

margPF = margPF/sum(margPF(:))/A;