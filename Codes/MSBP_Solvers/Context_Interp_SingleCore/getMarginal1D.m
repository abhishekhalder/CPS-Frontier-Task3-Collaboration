function [X1,marg] = getMarginal1D(X,nBins)

X1   = zeros(1,nBins);
marg = zeros(nBins,1);

Grid = linspace(min(X),max(X),nBins+1);
dx = mean(diff(Grid));

for i=1:nBins

    ii = find(X>=Grid(i) & X < Grid(i+1));

    marg(i,1) = length(ii); % MC marginal via frequentist counting 

    X1(i) = (Grid(i)+Grid(i+1))/2.0;

end

% marg.MC = marg.MC/sum(marg.MC)/dx;
marg = marg/sum(marg)/dx;