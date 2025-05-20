% function output_matrix = SinkhornOMT(input_matrix, rho_x, rho_y)
% 
% v = ones(length(rho_x),1);
% 
% max_iter = 100; tol = 1e-3;
% 
% iter = 1;
% 
% while iter < max_iter
%     
%     u = rho_y ./ (input_matrix*v);
%     
%     v = rho_x ./ (input_matrix'*u);
%     
%     output_matrix = diag(v)*input_matrix*diag(u);
%     
%     if (norm(output_matrix'*ones(length(rho_x),1) - rho_y,inf) < tol) && (norm(output_matrix*ones(length(rho_y),1) - rho_x,inf) < tol)
%         break;
%     else
%         iter = iter + 1;
%     end
%     
% end


function output_matrix = SinkhornOMT(input_matrix, rho_x, rho_y)

v = ones(length(rho_x),1);
% v = ones(length(rho_y),1);

max_iter = 100; tol = 1e-3;

iter = 1;

while iter < max_iter
    
    u = rho_y ./ (input_matrix'*v);
    v = rho_x ./ (input_matrix*u);
    
%     u = rho_x ./ (input_matrix*v);
%     v = rho_y ./ (input_matrix'*u);
    
    output_matrix = input_matrix;
    
    for i=1:size(input_matrix,2)
        output_matrix(:,i) = output_matrix(:,i) .* u(i);
    end
    for i=1:size(input_matrix,1)
        output_matrix(i,:) = output_matrix(i) .* v(i);
    end
    
    if (norm(output_matrix'*ones(length(rho_x),1) - rho_y,inf) < tol) && (norm(output_matrix*ones(length(rho_y),1) - rho_x,inf) < tol)
        break;
    else
        iter = iter + 1;
    end
    
end
