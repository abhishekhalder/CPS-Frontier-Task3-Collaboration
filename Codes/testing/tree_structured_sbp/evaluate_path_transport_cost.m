function cst = evaluate_path_transport_cost(C,K,u,s,n,epsilon)
    % Form the U tensor
    U = tensorprod(u{1},u{2},2,2); for i=3:s; U = tensorprod(U,u{i}); end
    cst = recursive_eval(C,K,U,s,n,epsilon,[],0);
return

% ! WARNING: this is dumb
function cst = recursive_eval(C,K,U,s,n,epsilon,inds,cst)
    if s==0
        Cel = 0; for i=1:numel(inds)-1; Cel = Cel + C{i}(inds(i),inds(i+1)); end
        % Kel = 1; for i=1:numel(inds)-1; Kel = Kel * K{i}(inds(i),inds(i+1)); end
        Kel = exp(-Cel/epsilon);
        indss = num2cell(inds);
        Uel = U(sub2ind(size(U),indss{:}));
        Mel = Kel * Uel;
        cst = ( Cel + epsilon*log(Mel) ) * Mel;
        if isnan(cst)
            cst = 1;
        end
        return
    else
        for i=1:n
            cst = cst + recursive_eval(C,K,U,s-1,n,epsilon,[inds i],cst);
        end
    end
return