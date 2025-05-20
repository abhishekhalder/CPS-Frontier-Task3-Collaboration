function margPF = get_xi_marg_singlecore(X1,X2,X3,c,locs,PDF,nBins)

dx1 = mean(diff(X1));
dx2 = mean(diff(X2));
dx3 = mean(diff(X3));

A = dx1*dx2*dx3;
[x1,x2,x3] = ndgrid(X1, X2, X3);
margPF = zeros(nBins,nBins,nBins);

% Find locations of \xi for a given c
% ll = find( locs(:,4) == c(1) & locs(:,5) == c(2) ); % works for 2-dim c
ll = find( locs(:,4) == c(1) & locs(:,5) == c(2) & locs(:,6) == c(3) ); % works for 3-dim c
% ll = find( locs(:,4:end) == c ); % doesnt work
locs_trunc = locs(ll,1:3);
PDF_trunc  = PDF(ll);

for i=1:nBins
    for j=1:nBins
        for k=1:nBins
            ind = find(locs_trunc(:,1) == x1(i,j,k) & ...
                       locs_trunc(:,2) == x2(i,j,k) & ...
                       locs_trunc(:,3) == x3(i,j,k) );
            margPF(i,j,k) = PDF_trunc(ind);
        end
    end
end

margPF = margPF/sum(margPF(:))/A;