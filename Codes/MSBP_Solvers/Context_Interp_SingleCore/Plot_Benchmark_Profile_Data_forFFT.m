%% Description
%
% For FFT with alloc. [2 2], plot original + additional 3D profile data
%
%==========================================================================
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',20)
% set(gcf,'position',[2459,55,3124,1041])
% set(groot,'defaultFontName','serif')
% fontname(gcf,"serif")
rng(0);

%% Problem parameters
%==========================================================================
BENCHMARK_NAME = "fft"; % MODIFY
%==========================================================================
dfile_pfix = sprintf("../%s_outfiles/marginals/%s_",BENCHMARK_NAME,BENCHMARK_NAME);
adfile_pfix = sprintf("../%s_outfiles/marginals_additional/%s_",BENCHMARK_NAME,BENCHMARK_NAME);
dfile_sfix = "_3dim.txt";
% out_dir    = "./data_out/";
%==========================================================================
VALID_CACHE = 2; % 1:20;
VALID_MEMBW = 2; % 1:20;
%==========================================================================
CTXT_SET    = VALID_CACHE.'+(1i*VALID_MEMBW);
for i=VALID_CACHE
    if( mod(i,5) ~= 0 & i ~= 1 )
        CTXT_SET(:,i) = 0;
        CTXT_SET(i,:) = 0;
    end
end
CTXT_SET    = setdiff(unique(CTXT_SET),[0]);
%==========================================================================
marg_times    = -0.1 + [0.25, 0.30, 0.35]; % 0:0.05:1;
% marg_times    = [0.1, 0.3, 0.5]; % 0:0.05:1;
% marg_times    = [0.1, 0.5, 0.9]; % 0:0.05:1;
marg_times_files = 0:0.05:1;
% taus          = 0.0:0.01:marg_times(end)-0.01;  % Times at which to interpolate
num_Timesteps = numel(marg_times);	% The number of time steps to solve over
num_CPUs      = 1;                  % Do not change this
nSample       = 100;                % The number of profiles
nSample_ad    = 500;                % The number of profiles
n             = numel(CTXT_SET) * nSample + nSample_ad;
%==========================================================================
numMarginal_Time = num_Timesteps;
numMarginal_CPU  = num_CPUs;
%==========================================================================


%% Load marginal data
%==========================================================================
rawD            = cell(numMarginal_Time,num_CPUs);
scaling_factors = cell(numMarginal_Time,numMarginal_CPU);
mu              = cell(numMarginal_Time,numMarginal_CPU);
locs            = cell(numMarginal_Time,numMarginal_CPU);
C               = cell(numMarginal_Time,numMarginal_CPU);
K               = cell(numMarginal_Time-1,numMarginal_CPU);
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        rawD{k,j} = zeros(n, 3);
        for i=1:numel(CTXT_SET)
            % Get context
            ctxt = [real(CTXT_SET(i)), imag(CTXT_SET(i))];
            ctxt_s = [bitshift(1,real(CTXT_SET(i)))-1, imag(CTXT_SET(i))*72];
            
            % Read marginal file
            mfile_ind = round(marg_times(k) / 0.05);
            curr_file = sprintf("%s%d_%d_MARG%d%s",dfile_pfix,ctxt_s(1),ctxt_s(2),mfile_ind,dfile_sfix);
            rawD_1C = importdata(curr_file);
            
            % Pad or truncate the data as necessary
            if( numel(rawD_1C) == 0 )
                rawD_1C = zeros(nSample,3);
            elseif( size(rawD_1C,1) < nSample )
                rawD_1C = padarray(rawD_1C,nSample-size(rawD_1C,1),0,'post')
            else
                rawD_1C = rawD_1C(1:nSample,:);
            end
            
            % Append to marginal
            blk = 1 + (i-1)*nSample;
            rawD{k,j}(blk:blk+nSample-1,:) = rawD_1C;


            % Read additional marginal file
            curr_file = sprintf("%s%d_%d_MARG%d%s",adfile_pfix,ctxt_s(1),ctxt_s(2),mfile_ind,dfile_sfix);
            rawD_1C = importdata(curr_file);
            
            % Pad or truncate the data as necessary
            if( numel(rawD_1C) == 0 )
                rawD_1C = zeros(nSample_ad,3);
            elseif( size(rawD_1C,1) < nSample_ad )
                rawD_1C = padarray(rawD_1C,nSample_ad-size(rawD_1C,1),0,'post')
            else
                rawD_1C = rawD_1C(1:nSample_ad,:);
            end
            
            % Append to marginal
            blk = 101 + (i-1)*nSample_ad;
            rawD{k,j}(blk:blk+nSample_ad-1,:) = rawD_1C;
        end
    end
end
%==========================================================================


%% Plot
%==========================================================================
rawD{1,1}(101,:) = rawD{1,1}(2,:); % Hacky fix, remove one sussy outlier
font = 'Noto Sans SemiCondensed Black';
% figure('DefaultTextFontName', font, 'DefaultAxesFontName', font);
f = figure('color','white','Position',[100 100 3500 1000], ...
           'DefaultTextFontName', font, 'DefaultAxesFontName', font);
ax = axes('Parent', f);

text(0.5, 0.5, 'Hello!');
% axes;
sph = [];
limits = [[Inf -Inf];[Inf -Inf];[Inf -Inf]];
sgtitle("Snapshots $\mu_\sigma$ for $\texttt{" + BENCHMARK_NAME + ...
    "}$, $\beta=(" + num2str(VALID_CACHE) + "," + num2str(VALID_MEMBW) + ")^\top$", 'FontSize', 30);

% Find axis limits
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        for i=1:3
            limits(i,:) = [min(limits(i,1),min(rawD{k,j}(:,i))) , max(limits(i,2),max(rawD{k,j}(:,i)))];
        end
    end
end

% Calculate weights
r = 1e2;
r1 = 1e5;
r2 = 1e3;
r3 = 1e2;
n = numel(rawD{i,1}(:,1));
weights = cell(numMarginal_Time,numMarginal_CPU);
for i=1:numMarginal_Time
    weights{i} = zeros(n,1);
    for j=1:n
        tmp = ( vecnorm(rawD{i,1}(j,1)-rawD{i,1}(:,1),2,2) < r1 );
        tmp = tmp & ( vecnorm(rawD{i,1}(j,2)-rawD{i,1}(:,2),2,2) < r2 );
        tmp = tmp & ( vecnorm(rawD{i,1}(j,3)-rawD{i,1}(:,3),2,2) < r3 );
        weights{i}(j) = sum( tmp );
        % weights{i}(j) = sum( vecnorm(rawD{i,1}(j,:)-rawD{i,1},2,2) < r );
    end
    weights{i} = weights{i} / n;
end

for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        sph(end+1) = subplot(numMarginal_CPU, numMarginal_Time, (j-1)*numMarginal_CPU+k);
    end
    
    scatter3(rawD{k,j}(:,1), rawD{k,j}(:,2), rawD{k,j}(:,3), 30, weights{k,j}, 'filled');
    hold on;
    if (k==1) 
        xlh = xlabel("Instructions retired", 'FontSize', 20, 'Rotation', 20, Interpreter='none'); 
        xlh.Position(:) = xlh.Position(:) - abs(xlh.Position(:) * 0.1);
        % xlh.Position = xlh.Position - [100 100];
    end
    if (k==1) ylabel("Cache requests", 'FontSize', 20, 'Rotation',-30, Interpreter='none'); end
    if (k==1) zlabel("Cache misses", 'FontSize', 20, Interpreter='none'); end
    % if (1==1)% (k==1) 
    %     xlabel("$\xi_1$", 'FontSize', 30);
    %     ylabel("$\xi_2$", 'FontSize', 30);
    %     zlabel("$\xi_3$", 'FontSize', 30);
    % end
    % 
    xlim(limits(1,:));
    ylim(limits(2,:));
    zlim(limits(3,:));

    f.CurrentAxes.XAxis.Exponent = 0;
    f.CurrentAxes.YAxis.Exponent = 0;
    f.CurrentAxes.ZAxis.Exponent = 0;

    xtickformat('%.1e');
    ytickformat('%.1e');
    ztickformat('%.1e');

    xticks(limits(1,:));
    yticks(limits(2,:));
    zticks(limits(3,:));

    xticklabels({'2.6e6', '3.2e6'});
    yticklabels({'2.1e4', '2.7e4'});
    zticklabels({'2.0e4', '2.8e4'});
    grid off;
    % if(k~=1) axis off; end
    if(k~=1)
        set(gca,'xticklabel',[])
        set(gca,'yticklabel',[])
        set(gca,'zticklabel',[])
    end
    hold off;
    
    if( j == 1 )
        title("$t_" + num2str(k) + "=" + sprintf("%0.2f", marg_times(k)) + "$", 'FontSize', 30);
    end
end

h = axes(f,'visible','off'); 
% c = colorbar(h,'Position',[0.93 0.168 0.022 0.7],'XTick', [0 1]);  % attach colorbar to h
c = colorbar(h,'location','southoutside','XTick', [0 1]);  % attach colorbar to h
colormap(c,'jet');
clim(h,[0,1]);

% sph(1).Position(1) = 0.1;



