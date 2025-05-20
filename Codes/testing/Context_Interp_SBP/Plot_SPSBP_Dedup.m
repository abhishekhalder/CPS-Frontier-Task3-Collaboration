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
load("data_out/MSBP_solution/MSBPsol_1013.mat","marginals", "pts", ...
     "pts_xi", "pts_c", "gridxi", "gridc", "taus", ...
     "instr_scale", "llcreq_scale", "llcmiss_scale");
num_CPUs = 4;

%% Figure 2 : Plot interpolated marginals at specified time(s), varying c
%==========================================================================
figure(2)

ctxts = [[1,1,1,1]; [5,1,5,1]; [4,3,2,1]; [3,3,3,3]; [1,4,2,5]];


for k=1:numel(ctxts(:,1))
    c = ctxts(k,:);
    for p=1:num_CPUs
        subplot(num_CPUs, numel(ctxts(:,1)), (p-1)*numel(ctxts(:,1))+k);
        % scatter3(ilocs{k}(:,1), ilocs{k}(:,2), ilocs{k}(:,3), 1, weights{k}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, marginals{2,p}{c(1),c(2),c(3),c(4)}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, marginals{2,p}{c(1),c(2),c(3),c(4)}, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, reshape(marginals{2,p}{c(1),c(2),c(3),c(4)},11*11*11,1), 'filled');
        marg_scaled = reshape(marginals{2,p}{c(1),c(2),c(3),c(4)},11*11*11,1);
        marg_scaled = 40*marg_scaled / max(marg_scaled,[],"all") + 0.001;
        scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), ...
                 marg_scaled, marg_scaled, 'filled');
        % scatter3(pts_xi(:,1), pts_xi(:,2), pts_xi(:,3), 20, 'filled');
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
%% Load additional data for comparison
addD = cell(num_interp,num_CPUs);

figure(4);
% endpoint measures
for k=1:num_interp
    % Load and scale data into spectral marginals
    for j=1:num_CPUs
        % addD{k,j} = readmatrix("../halder_outfiles_0103/canneal_7_72_MARG" + num2str(k-1) + "." + num2str(j) + dfile_sfix);
        addD{k,j} = readmatrix(dfile_pfix + "INT" + num2str(k-1) + "." + num2str(j) + dfile_sfix);
        
        if( size(addD{k,j},1) < n )                     % Pad the data if necessary
            addD{k,j} = padarray(addD{k,j},n-size(addD{k,j},1),0,'post')
        end
        
        addD{k,j}(:,1) = addD{k,j}(:,1) * instr_scale;
        addD{k,j}(:,2) = addD{k,j}(:,2) * llcreq_scale;
        addD{k,j}(:,3) = addD{k,j}(:,3) * llcmiss_scale;
        
        subplot(num_CPUs, num_interp, (j-1)*num_interp+k);
        % scatter3(addD{k,j}(:,1), addD{k,j}(:,2), addD{k,j}(:,3), 40, 1/n*ones(n,1), 'filled');
        scatter3(addD{k,j}(:,1), addD{k,j}(:,2), addD{k,j}(:,3), 40, 'filled');
        if( j == 1 )
            title("$t=" + num2str(taus(k)) + "$",'FontSize',20);
        end
        % locs{k,j} = [addD{k,j}(:,1), addD{k,j}(:,2), addD{k,j}(:,3)];
    end
end


%% Table 1: Compute Wasserstein distances.
% Get Entropy-Regularized Wasserstein distances between interpolations
% and measured.
epsilon_wass = 0.01;
wass  = zeros(num_interp,num_CPUs);
for k=1:num_interp
    for p=1:num_CPUs
        wass(k,p) = Wasserstein(addD{k,p}, dlocs{k,p}, 1/n*ones(n,1), dmags{k,p});
        fprintf("(k,p)=(%d,%d) | %f\n", k, p, wass(k,p));
    end
end
save('wass_sp.mat', 'wass');

%% Print the values
disp("--------Wasserstein Distances for Interpolations--------");
for p=1:num_CPUs
    fprintf('CPU%d | ', p);
    for k=1:num_interp
        fprintf('%5.5f ', wass(k,p));
    end
    fprintf('\n');
end
disp("--------------------------------------------------------");


% %% Figure 6: Plot 1D marginalized interpolated marginals vs. measured at specified times (sep. figure for each CPU)
% %==========================================================================
% f5 = figure(5);
% set(gcf,'color','w');
% set(0, 'DefaultLineLineWidth', 2.5);
% nBins = 100;
% for p=1:num_CPUs
%     figure(6+(p-1));
%     for j=1:3
%         subplot(3,1,j);
%         
%         % Plot the measured intermediate distributions
%         for k=1:num_interp
%             [X1, marg] = getMarginal1D(addD{k,p}(:,j), nBins);
%             marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
%             plot3(((k)-0.001)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "r");
%             hold on;
%             fill3(((k)-0.001)*ones(numel(X1),1), X1, marg, "r", 'FaceAlpha', 0.5);
%             if( ~any(addD{k,p}(:,j)) )
%                 line([(k)-0.001,(k)-0.001], [0,0], [0,1000], 'Color', 'r');
%             end
%         end
%         
%         % Plot interpolated distributions
%         for k=1:num_interp
%             %         [X1, marg] = getWeightedMarginal1D(locs{k}(:,j), mags{k}, nBins*1);
%             %         marg = smoothdata(marg);
%             %         plot3((si+(ei-si)*(k/(num_interp+1)))*ones(numel(X1),1), X1, marg, "g");
%             [X1, marg] = getWeightedMarginal1D(dlocs{k,p}(:,j), dmags{k,p}, nBins*1);
%             marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
%             plot3((k)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "b");
%             fill3((k)*ones(numel(X1),1), X1, marg, "b", 'FaceAlpha', 0.5);
%             if( ~any(dlocs{k,p}(:,j)) )
%                 line([(k),(k)], [0,0], [0,1000], 'Color', 'b');
%             end
%         end
%         
%         % Formatting
%         % set(gca,'FontSize',30)
%         set(gca,'YTick',[], 'ZTick', [])
%         zlim([0 400]);
%         xlim([1-0.002 num_interp+0.002]);
%         view(-12,35);
%         ylabel("$\xi_{" + num2str(j) + "}$")
%         % xlabel("$\tau$")
%         if ( j == 1 )
%             title("CPU"+num2str(p),'FontSize',30);
%             % title("$\hat{\mu}_{\hat{\tau}_j}$, $\mu_{\hat{\tau}_j}$ for $j\in[5]$");
%             %     elseif ( j == 2 )
%             %     	zticklabels(1/n*zticks);
%             %     elseif ( j == 3 )
%         end
%         if ( j == 3 )
%             set(gca,'XTick', 1:num_interp);
%             xticklabels(1:num_interp);
%         else
%             set(gca,'XTick',[])
%         end
%         % zticklabels(1/n*zticks);
%         hold off;
%     end
% end


% %% Figure 5: Plot 1D marginalized interpolated marginals vs. measured at specified times (all In one figure)
% %==========================================================================
% f5 = figure(25);
% set(gcf,'color','w');
% set(0, 'DefaultLineLineWidth', 2.5);
% nBins = 100;
% for p=1:num_CPUs
%     for j=1:3
%         subplot(6,2,(1+6*floor(p/3)+(1-mod(p,2)))+2*(j-1));
%         
%         % Plot the measured intermediate distributions
%         for k=1:num_interp
%             [X1, marg] = getMarginal1D(addD{k,p}(:,j), nBins);
%             marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
%             plot3(((k)-0.001)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "r");
%             hold on;
%             fill3(((k)-0.001)*ones(numel(X1),1), X1, marg, "r", 'FaceAlpha', 0.5);
%             if( ~any(addD{k,p}(:,j)) )
%                 line([(k)-0.001,(k)-0.001], [0,0], [0,1000], 'Color', 'r');
%             end
%         end
%         
%         % Plot interpolated distributions
%         for k=1:num_interp
%             %         [X1, marg] = getWeightedMarginal1D(locs{k}(:,j), mags{k}, nBins*1);
%             %         marg = smoothdata(marg);
%             %         plot3((si+(ei-si)*(k/(num_interp+1)))*ones(numel(X1),1), X1, marg, "g");
%             [X1, marg] = getWeightedMarginal1D(dlocs{k,p}(:,j), dmags{k,p}, nBins*1);
%             marg = smoothdata(marg); marg(1) = 0; marg(end) = 0;
%             plot3((k)*ones(numel(X1)-2,1), X1(2:end-1), marg(2:end-1), "b");
%             fill3((k)*ones(numel(X1),1), X1, marg, "b", 'FaceAlpha', 0.5);
%             if( ~any(dlocs{k,p}(:,j)) )
%                 line([(k),(k)], [0,0], [0,1000], 'Color', 'b');
%             end
%         end
%         
%         % Formatting
%         % set(gca,'FontSize',30)
%         set(gca,'YTick',[], 'ZTick', [])
%         zlim([0 400]);
%         xlim([1-0.002 num_interp+0.002]);
%         view(-12,35);
%         ylabel("$\xi_{" + num2str(j) + "}$")
%         % xlabel("$\tau$")
%         if ( j == 1 )
%             title("CPU"+num2str(p),'FontSize',30);
%             % title("$\hat{\mu}_{\hat{\tau}_j}$, $\mu_{\hat{\tau}_j}$ for $j\in[5]$");
%             %     elseif ( j == 2 )
%             %     	zticklabels(1/n*zticks);
%             %     elseif ( j == 3 )
%         end
%         if ( j == 3 )
%             set(gca,'XTick', 1:num_interp);
%             xticklabels(1:num_interp);
%         else
%             set(gca,'XTick',[])
%         end
%         % zticklabels(1/n*zticks);
%         hold off;
%     end
% end
% 
% 
%% Figure 6: 1D marginalized vs. interpolated, sep. graph for each one
%==========================================================================
nBins = 100;
for p=1:num_CPUs
    f5 = figure(5+p);
    set(gcf,'color','w');
    set(0, 'DefaultLineLineWidth', 1.5);
    for j=1:3
        %subplot(3,num_interp,); %(1+6*floor(p/3)+(1-mod(p,2)))+2*(j-1));
        
        for k=1:num_interp
            % Plot the measured intermediate distributions
            subplot(3,num_interp,(j-1)*(num_interp)+k);
            [X1, marg] = getMarginal1D(addD{k,p}(:,j), nBins);
            % marg = smoothdata(marg); 
            marg = smoothdata(normalize(marg,"norm")); 
            marg(1) = 0; marg(end) = 0;
            writematrix([X1' marg], out_dir + "measured_t" + num2str(k) + "_xi" + num2str(j) + "_CPU" + num2str(p) + ".txt");
            plot(X1(2:end-1), marg(2:end-1), "r");
            hold on;
            %fill(X1, marg, "r", 'FaceAlpha', 0.5);
            if( ~any(addD{k,p}(:,j)) )
                line([0,0], [0,1000], 'Color', 'r');
            end
            
            % Plot interpolated distributions
            [X1, marg] = getWeightedMarginal1D(dlocs{k,p}(:,j), dmags{k,p}, nBins);
            % marg = smoothdata(marg); 
            marg = smoothdata(normalize(marg,"norm")); 
            marg(1) = 0; marg(end) = 0;
            writematrix([X1' marg], out_dir + "interpolated_t" + num2str(k) + "_xi" + num2str(j) + "_CPU" + num2str(p) + ".txt");
            plot(X1(2:end-1), marg(2:end-1), "b");
            % fill(X1, marg, "b", 'FaceAlpha', 0.5);
            if( ~any(dlocs{k,p}(:,j)) )
                line([0,0], [0,1000], 'Color', 'b');
            end
            if( k == 1 )
                ylabel("$\xi_{" + num2str(j) + "}$", 'FontSize', 16)
            end
            if ( j == 1 )
                % title("CPU"+num2str(p),'FontSize',30);
                title("$\hat{\mu}_{\hat{\tau}_j}$, $\mu_{\hat{\tau}_j}$ for $j=" + num2str(k) + "$", 'FontSize', 16);
                %     elseif ( j == 2 )
                %     	zticklabels(1/n*zticks);
                %     elseif ( j == 3 )
            end
        end
        
        hold off;
    end
    sgtitle("CPU"+num2str(p),'FontSize',30)
end


%% Figure 7: Graph of Wasserstein distances for both SP and BC cases
figure(12);
set(gcf,'color','w');
set(0, 'DefaultLineLineWidth', 1.5);
save('wass_sp.mat', 'wass');
wass_bc = load("../Barycenter_Graph_SBP/wass_bc.mat").wass;
for p=1:num_CPUs
    semilogy(1:num_interp, max(eps, wass(:,p)), '-or');
    hold on;
    semilogy(1:num_interp, max(eps, wass_bc(:,p)), '-ob');
end
set(gca,'FontSize', 20)
set(gca,'Xtick',1:1:5)
% set(gca,'FontSize',30)
xlabel('Interpolation index $j$','FontSize',30)
ylabel('Wasserstein error ($W_j$)','FontSize',30)
ylim([1e-8, 1e-4])
hold off;

% %% Testing: Projection times for SP vs. BS cases
% %==========================================================================
% ptimes_bs = load("ptimes_bs.mat").ptimes;
% ptimes_sp = load("ptimes_sp.mat").ptimes;
% hist_nbins = 30;
% 
% p1 = histogram(ptimes_sp, hist_nbins, 'facealpha', 0.3, 'edgecolor', 'none');
% hold on;
% p2 = histogram(ptimes_bs, hist_nbins, 'facealpha', 0.7, 'edgecolor', 'none');
% title("Single-marginal Projection Times");
% xlabel("t (s)");
% ylabel("Count");
% legend([p1, p2], ["Series-Parallel", "Barycentric"]);
% hold off;


