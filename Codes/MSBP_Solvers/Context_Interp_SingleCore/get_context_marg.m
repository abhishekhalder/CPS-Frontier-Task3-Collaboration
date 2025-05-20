function margPF = get_context_marg(X1,X2,X3,X4,locs,PDF,nBins)

dx1 = mean(diff(X1));
dx2 = mean(diff(X2));
dx3 = mean(diff(X3));
dx4 = mean(diff(X4));

A = dx1*dx2*dx3*dx4;

[x1,x2,x3,x4] = ndgrid(X1, X2, X3, X4);

% margMC = zeros(nBins,nBins,nBins,nBins);
margPF = zeros(nBins,nBins,nBins,nBins);

for i=1:nBins
    for j=1:nBins
        for k=1:nBins
            for l=1:nBins
                ll = find(locs(:,4) == x1(i,j,k,l) & ...
                          locs(:,5) == x2(i,j,k,l) & ...
                          locs(:,6) == x3(i,j,k,l) & ...
                          locs(:,7) == x4(i,j,k,l) );
                
                % margMC(i,j,k,l) = length(ll);
                margPF(i,j,k,l) = sum(PDF(ll));
            end
        end
    end
end

% margMC = margMC/sum(margMC(:))/A;
margPF = margPF/sum(margPF(:))/A;