function margPF = get_context_marg_singlecore(X1,X2,locs,PDF,nBins)

dx1 = mean(diff(X1));
dx2 = mean(diff(X2));
A = dx1*dx2;
[x1, x2] = ndgrid(X1,X2);
margPF = zeros(nBins,nBins);

for i=1:nBins
    for j=1:nBins
        ll = find( locs(:,4) == x1(i) & locs(:,5) == x2(j) );
        margPF(i,j) = sum(PDF(ll));
    end
end

margPF = margPF/sum(margPF(:))/A;