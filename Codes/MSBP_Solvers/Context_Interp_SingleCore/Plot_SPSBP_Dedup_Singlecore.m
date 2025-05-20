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
set(groot,'defaultAxesFontSize',20)

%% Problem parameters
%==========================================================================
out_dir    = sprintf("./data_out/%s_synth_profiles/", BENCHMARK_NAME);
load("data_out/MSBP_solution/MSBPsol_1015_iCPUs.mat","marginals", ...
     "marginals_stats", "marginals_scales", "pts", "pts_xi", "pts_c", ...
     "gridxi", "gridc", "taus", "instr_scale", "llcreq_scale", "llcmiss_scale");
num_CPUs = 1;

%% Figure 2 : Plot interpolated marginals at specified time(s), varying c
%==========================================================================
figure(3)

ctxts = [[1,1];[2,2];[3,3]];

tau = 10;

for k=1:numel(ctxts(:,1))
    % c = ctxts(k,:);
    c1 = ctxts(k,1); c2 = ctxts(k,2);
    for p=1:num_CPUs
        subplot(num_CPUs, numel(ctxts(:,1)), (p-1)*numel(ctxts(:,1))+k);
        % scatter3(ilocs{k}(:,1), ilocs{k}(:,2), ilocs{k}(:,3), 1, weights{k}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, marginals{2,p}{c(1),c(2),c(3),c(4)}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, marginals{2,p}{c(1),c(2),c(3),c(4)}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, reshape(marginals{2,p}{c(1),c(2),c(3),c(4)},11*11*11,1), 'filled');
        marg_scaled = reshape(marginals{tau,p}{c1,c2},numel(gridxi)^3,1);
        marg_scaled = 40*marg_scaled / max(marg_scaled,[],"all") + 0.001;
        % Plot marginal distribution
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), ...
        scatter3(pts_xi(:,1)/marginals_scales{tau,p}(1), ...
                 pts_xi(:,2)/marginals_scales{tau,p}(2), ...
                 pts_xi(:,3)/marginals_scales{tau,p}(3), ...
                 marg_scaled, marg_scaled, 'filled');
        hold on;
        
        % Plot marginal mean & standard deviation
        mean_c = marginals_stats{tau,p}{c1,c2}(1,:)./marginals_scales{tau,p}(1:3);
        std_c  = marginals_stats{tau,p}{c1,c2}(2,:)./marginals_scales{tau,p}(1:3);
        
        scatter3(mean_c(1), mean_c(2), mean_c(3), 50, 'black', 'filled', 'square');
        scatter3(std_c(1), std_c(2), std_c(3), 50, 'black', 'filled', 'v');
        
        hold off;
        
        if( k == 1 )
            zlabel("CPU" + num2str(p),'FontSize',30)
        end
        if( p == 1 )
            title("$c=[" + num2str(c1) + "," + num2str(c2) + "]$",'FontSize',20);
        end
    end
end

return


%% Generate synthetic marginals
%==========================================================================
out_dir    = sprintf("./data_out/%s_synth_profiles/", BENCHMARK_NAME);
synth_profiles = cell(numel(gridc),numel(gridc));
for c1=gridc
    for c2=gridc
        
        %for j=1:numel(pts_c(:,1))
        profile = zeros(numel(taus),7);
        for k=1:numel(taus)
            mean_t = marginals_stats{k,1}{c1,c2}(1,:)./marginals_scales{k,1}(1:3);
            std_t  = marginals_stats{k,1}{c1,c2}(2,:)./marginals_scales{k,1}(1:3);
            profile(k,:) = [taus(k), mean_t, std_t];
        end
        synth_profiles{c1,c2} = profile;
        writematrix(synth_profiles{c1,c2}, out_dir + "dedup-synth_c" + num2str(bitshift(1,c1)-1) + "_" + num2str(c2*72) + ".txt");
        %end
    end
end

% for j=1:numel(pts_c(:,1))
%     profile = zeros(numel(taus),7);
%     for k=1:numel(taus)
%         mean_t = marginals_stats{k,1}{pts_c(j,1)}(1,:)./marginals_scales{k,1}(1:3);
%         std_t  = marginals_stats{k,1}{pts_c(j,1)}(2,:)./marginals_scales{k,1}(1:3);
%         profile(k,:) = [taus(k), mean_t, std_t];
%     end
%     synth_profiles{1,j} = profile;
%     writematrix(synth_profiles{1,j}, out_dir + "dedup-synth_c" + num2str(bitshift(1,pts_c(j,1))-1) + "_" + num2str(pts_c(j,2)*72) + ".txt");
% end


