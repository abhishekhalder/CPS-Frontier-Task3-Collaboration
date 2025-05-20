%% Description
%
% We generate some plots using the solution of the SPSBP with Robert's
% Dedup data.
%
%==========================================================================
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Problem parameters
%==========================================================================
out_dir    = "./data_out/";
load("data_out/MSBP_solution/MSBPsol_1015_iCPUs.mat","marginals", ...
     "marginals_stats", "marginals_scales", "pts", "pts_xi", "pts_c", ...
     "gridxi", "gridc", "taus", "instr_scale", "llcreq_scale", "llcmiss_scale");
num_CPUs = 4;

%% Figure 2 : Plot interpolated marginals at specified time(s), varying c
%==========================================================================
figure(3)

ctxts = [[1,1,1,1]; [5,1,5,1]; [4,3,2,1]; [3,3,3,3]; [1,4,2,5]];


for k=1:numel(ctxts(:,1))
    c = ctxts(k,:);
    for p=1:num_CPUs
        subplot(num_CPUs, numel(ctxts(:,1)), (p-1)*numel(ctxts(:,1))+k);
        % scatter3(ilocs{k}(:,1), ilocs{k}(:,2), ilocs{k}(:,3), 1, weights{k}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, marginals{2,p}{c(1),c(2),c(3),c(4)}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, marginals{2,p}{c(1),c(2),c(3),c(4)}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, reshape(marginals{2,p}{c(1),c(2),c(3),c(4)},11*11*11,1), 'filled');
        marg_scaled = reshape(marginals{2,p}{c(p)},numel(gridxi)^3,1);
        marg_scaled = 40*marg_scaled / max(marg_scaled,[],"all") + 0.001;
        % Plot marginal distribution
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), ...
        scatter3(pts_xi(:,1)/marginals_scales{2,p}(1), ...
                 pts_xi(:,2)/marginals_scales{2,p}(2), ...
                 pts_xi(:,3)/marginals_scales{2,p}(3), ...
                 marg_scaled, marg_scaled, 'filled');
        hold on;
        
        % Plot marginal mean & standard deviation
        mean_c = marginals_stats{2,p}{c(p)}(1,:)./marginals_scales{2,p}(1:3);
        std_c  = marginals_stats{2,p}{c(p)}(2,:)./marginals_scales{2,p}(1:3);
        
        scatter3(mean_c(1), mean_c(2), mean_c(3), 50, 'black', 'filled', 'square');
        ellipsoid(mean_c(1), mean_c(2), mean_c(3), ...
                  std_c(1), std_c(2), std_c(3) );
        
        hold off;
        
        if( k == 1 )
            zlabel("CPU" + num2str(p),'FontSize',30)
        end
        if( p == 1 )
            title("$c=[" + num2str(c(1)) + "," + num2str(c(2)) ...
                  + "," + num2str(c(3)) + "," + num2str(c(4)) ...
                  + "]$",'FontSize',20);
        end
        
        % set(gca,'XLim',x_lim,'YLim',y_lim,'ZLim',z_lim);
        % h = scatter3(locs{k}(:,1), locs{k}(:,2), locs{k}(:,3), 40, 'filled');
        % set(h, 'MarkerEdgeAlpha', alphaI, 'MarkerFaceAlpha', alphaI);
        % disp(sum(imags{k,p}))
        % title("$\hat{\mu}_{" + num2str(si+(ei-si)*(k/(num_interp+1)),3) + "}$",'FontSize',30);
    end
end

return


%% Generate synthetic marginals
%==========================================================================
synth_profiles = cell(num_CPUs,numel(pts_c));

for i=1:num_CPUs
    for j=1:numel(pts_c)
        profile = zeros(numel(taus),7);
        for k=1:numel(taus)
            mean_t = marginals_stats{k,i}{j}(1,:)./marginals_scales{k,i}(1:3);
            std_t  = marginals_stats{k,i}{j}(2,:)./marginals_scales{k,i}(1:3);
            profile(k,:) = [taus(k), mean_t, std_t];
        end
        synth_profiles{i,j} = profile;
        writematrix(synth_profiles{i,j}, out_dir + ...
                    "synthetic_profiles_1022/dedup-synth_CPU" + num2str(i) + ...
                    "_c" + num2str(pts_c(j)) + ".txt");
    end
end


