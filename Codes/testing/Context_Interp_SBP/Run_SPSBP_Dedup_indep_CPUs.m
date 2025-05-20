%% Description
%
% We solve a Series-parallel (SP) tracking MSBP with Robert's synthetic
% data. In this file we assume that CPUs are dependent only on their own
% context, thereby reducing n.
%
%==========================================================================
close all; clear; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
rng(0);

%% Problem parameters
%==========================================================================
dfile_pfix = "../dedup_outfiles_1015/marginals/dedup_";
dfile_sfix = "_3dim.txt";
out_dir    = "./data_out/";

NUM_CACHE   = 5;
VALID_CACHE = [ 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111, 0b11111111, 0b111111111, ...
               0b1111111111, 0b11111111111, 0b111111111111, 0b1111111111111, 0b11111111111111, ...
               0b111111111111111, 0b1111111111111111, 0b11111111111111111, 0b111111111111111111, ...
               0b1111111111111111111 ,0b11111111111111111111 ];
VALID_CACHE = VALID_CACHE(1:NUM_CACHE);

marg_times    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
% marg_times    = [0.03, 0.15125, 0.27249999999999996, 0.39375000000000004, 0.515];     % Times at which marginals are placed
% marg_times    = [0.0, 0.03, 0.15125, 0.27249999999999996, 0.39375000000000004, 0.515, 0.63625, 0.7575000000000001, 0.87875, 1.0];     % Times at which marginals are placed
% taus          = 0.0:0.01:marg_times(end)-0.01;               % Times at which to interpolate
taus          = 0.0:0.04:marg_times(end)-0.01;               % Times at which to interpolate
% taus          = taus(63:64);
num_Timesteps = numel(marg_times);	% The number of time steps to solve over
num_CPUs      = 4;                  % The number of 'nodes' (sensors, CPUs, etc.) at each time step

numMarginal_Time = num_Timesteps; 
numMarginal_CPU  = num_CPUs;

% state space
nSample = 100;                                  % The number of scattered points at each spectral marginal.
n = NUM_CACHE * nSample;
num_interpedges = (num_CPUs*(num_CPUs-1)) / 2;  % The number of spectral marginal pairs to interpolate
                                                % between for the computation of the barycenter (equal to
                                                % to the number of edges in the complete graph K_{num_CPUs})

% Default data scaling parameters
instr_scale     = 1e-10;
llcreq_scale    = 1e-8;
llcmiss_scale   = 1e-8;
context_scale   = 0.0032;

autoscale = true;                               % If false, use defaults above
scaled_maxval = 0.1;                            % If autoscale==true, all components of data scaled to [0,scaled_maxval]

epsilon = 0.1;                                  % entropic regularization parameter
% epsilon = 0.05;                                  % entropic regularization parameter
%==========================================================================

rawD            = cell(numMarginal_Time,num_CPUs);              % raw data
scaling_factors = cell(numMarginal_Time,numMarginal_CPU);
mu              = cell(numMarginal_Time,numMarginal_CPU);
locs            = cell(numMarginal_Time,numMarginal_CPU);
C               = cell(numMarginal_Time,numMarginal_CPU);
K               = cell(numMarginal_Time-1,numMarginal_CPU);

% endpoint measures
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
        % Load and scale data into marginals
        rawD{k,j} = zeros(n, 3+1);
        for i=1:NUM_CACHE
            % for l=1:num_CPUs
                ctxt = [0 0 0 0];
                ctxt(j) = VALID_CACHE(i);
                % ctxt = VALID_CACHE(i);
                % rawD{k,j} = readmatrix(dfile_pfix + num2str(k+1) + "." + num2str(j) + dfile_sfix);
                % rawD_1C = readmatrix(dfile_pfix + num2str(ctxt(1)) + "_" + num2str(ctxt(2)) ...
                rawD_1C = importdata(dfile_pfix + num2str(ctxt(1)) + "_" + num2str(ctxt(2)) ...
                                     + "_" + num2str(ctxt(3)) + "_" + num2str(ctxt(4)) + "_MARG"  ...
                                     + num2str(k-1) + "." + num2str(j) + dfile_sfix);
                
                % Pad or truncate the data as necessary
                if( numel(rawD_1C) == 0 )
                    rawD_1C = zeros(nSample,3);
                elseif( size(rawD_1C,1) < nSample )
                    rawD_1C = padarray(rawD_1C,nSample-size(rawD_1C,1),0,'post')
                else
                    rawD_1C = rawD_1C(1:nSample,:);
                end
                
                % Affix context to \xi
                rawD_1C = [rawD_1C ones(nSample,1)*ctxt(j)];
                
                % Append to marginal
                blk = 1 + (i-1)*nSample;
                rawD{k,j}(blk:blk+nSample-1,:) = rawD_1C;
            % end
        end
        if( autoscale )
            if( max(rawD{k,j}(:,1)) ~= 0 ) instr_scale   = scaled_maxval / max(rawD{k,j}(:,1)); else instr_scale   = 1; end
            if( max(rawD{k,j}(:,2)) ~= 0 ) llcreq_scale  = scaled_maxval / max(rawD{k,j}(:,2)); else llcreq_scale  = 1; end
            if( max(rawD{k,j}(:,3)) ~= 0 ) llcmiss_scale = scaled_maxval / max(rawD{k,j}(:,3)); else llcmiss_scale = 1; end
            if( max(rawD{k,j}(:,4)) ~= 0 ) context_scale = scaled_maxval / max(rawD{k,j}(:,4)); else context_scale = 1; end
            fprintf("====Autoscaling Parameters====\n");
            fprintf('instr_scale   = %f\n', instr_scale);
            fprintf('llcreq_scale  = %f\n', llcreq_scale);
            fprintf('llcmiss_scale = %f\n', llcmiss_scale);
            fprintf('context_scale = %f\n', context_scale);
            fprintf("==============================\n\n");
        end
        
        % Scale the data for stability of the MSBP solver
        rawD{k,j}(:,1) = rawD{k,j}(:,1) * instr_scale;
        rawD{k,j}(:,2) = rawD{k,j}(:,2) * llcreq_scale;
        rawD{k,j}(:,3) = rawD{k,j}(:,3) * llcmiss_scale;
        rawD{k,j}(:,4) = rawD{k,j}(:,4) * context_scale;
        
        scaling_factors{k,j} = [instr_scale llcreq_scale llcmiss_scale context_scale];
        
        locs{k,j} = [rawD{k,j}(:,1), rawD{k,j}(:,2), rawD{k,j}(:,3), rawD{k,j}(:,4)];
        
        mu{k,j} = (1/n) * ones(n,1);
    end
end

% Generate cost matrices
for k=1:numMarginal_Time-1
    for j=1:numMarginal_CPU
        if k==1
            C{k,j} = pdist2(locs{1,1}, locs{2,j});
        elseif k==numMarginal_Time-1
            C{k,j} = pdist2(locs{k,j}, locs{k+1,1});
        else
            C{k,j} = pdist2(locs{k,j}, locs{k+1,j});
        end
        K{k,j} = exp(-C{k,j}/epsilon);
    end
end

%% Visualize Marginal Data
%==========================================================================
% figure(1)
% for k=1:numMarginal_Time
%     for j=1:numMarginal_CPU
%         subplot(numMarginal_Time,numMarginal_CPU,numMarginal_Time*(k-1)+j);
%         scatter(locs{k,j}(:,1), locs{k,j}(:,2));
%         title("locs\{" + num2str(k) + "," + num2str(j) + "\}");
%     end
% end

%% Algorithm parameters
%==========================================================================
maxIter = 10000; tol = 1e-12; maxtol = 1e5;
u = cell(numMarginal_Time,numMarginal_CPU);
err = cell(numMarginal_Time,numMarginal_CPU);
ptimes = [];

for k=1:numMarginal_Time
    if k==1 || k==numMarginal_Time
        u{k,1} = rand(n,1);
        err{k,1} = { []; [] };
    else
        for j=1:numMarginal_CPU
            u{k,j} = rand(n,1);
            err{k,j} = { []; [] };
        end
    end
end

iter_idx = 1;
t        = 1;
j        = 1;
tic;
while iter_idx <= maxIter
    fprintf('(Iter,t,j) = (%d, %d, %d)\n', iter_idx, t, j);
    
    u_old = u{t,j};
    
    % Calculate projection
    % tic;
    Proj = Proj1_scattered(t, j, numMarginal_Time, numMarginal_CPU, K, u);
    % ptimes(end+1) = toc;
    % Update iteration
    u{t,j} = u{t,j} .* mu{t,j} ./ Proj;
    
    % Calculate error
    err{t,j}{1}(end+1) = iter_idx;
    err{t,j}{2}(end+1) = max(1e-16, HilbertProjectiveMetric(u{t,j},u_old));
    
    disp(['Err ',num2str(err{t,j}{2}(end))])
    max_err = err{t,j}{2}(end);
    if (iter_idx >= numMarginal_Time*numMarginal_CPU)
        for k=1:numMarginal_Time
            if k==1 || k==numMarginal_Time
                max_err = max(max_err, err{k,1}{2}(end));
            else
                for l=1:numMarginal_CPU
                    max_err = max(max_err, err{k,l}{2}(end));
                end
            end
        end
    else
        max_err = 228;
    end
    disp(['Max_Err ',num2str(max_err)])                                                                                                                                                                                                                    
    
    % check convergence in Hilbert metric
    if (max_err < tol)
        break;
    elseif( isinf(err{t,j}{2}(end)) || isnan(err{t,j}{2}(end)))
        fprintf('Error: NaN or Inf detected in Hilbert metric on iteration (t,j)=(%d,%d). Stopping...\n', t, j);
        break;
    else
        iter_idx = iter_idx+1;  
        if j == numMarginal_CPU || (t==1 || t==numMarginal_Time)
            j = 1;
        else
            j = j + 1;
        end
        if j == 1
            t = mod(t, numMarginal_Time) + 1;
        end
    end
end
comptime_recursion = toc
% histogram(ptimes, 30, 'facealpha', 0.3, 'edgecolor', 'none');

%% Find the transport maps between adjacent marginals
%==========================================================================
M = cell(numMarginal_Time-1,numMarginal_CPU);
Msums = zeros(numMarginal_Time-1,numMarginal_CPU);

for k=1:numMarginal_Time-1
    for j=1:numMarginal_CPU
        if k==1
            M{k,j} = Proj2_scattered([1,1], [2,j], numMarginal_Time, numMarginal_CPU, K, u);
        elseif k==numMarginal_Time-1
            M{k,j} = Proj2_scattered([k,j], [k+1,1], numMarginal_Time, numMarginal_CPU, K, u);
        else
            M{k,j} = Proj2_scattered([k,j], [k+1,j], numMarginal_Time, numMarginal_CPU, K, u);
        end
        
        Msums(k,j) = sum(M{k,j},"all");
        disp(sum(M{k,j},"all"));
    end
end

%% Figure 1 : Plot convergence in Hilbert metric
%==========================================================================
% plot the covergence in Hilbert metric
figure(1)
for k=1:numMarginal_Time
	semilogy(err{k,1}{1}, err{k,1}{2}, 'LineWidth',2);
    hold on;
    if k>1 && k<numMarginal_Time
        for l=2:numMarginal_CPU
            semilogy(err{k,l}{1}, err{k,l}{2}, 'LineWidth',2);
            writecell(err{k,l}, out_dir + "f1_err" + num2str((k-1)*num_CPUs + (l-1)) + ".txt");
        end
    end
end
% set(gca,'FontSize',30)
xlabel('iteration index $j$','FontSize',30)
ylabel('Error','FontSize',30)
% legend('$d_{\rm{Hilbert}}(u_{0}^{j},u_{0}^{j+1})$','$d_{\rm{Hilbert}}(u_{1}^{j},u_{1}^{j+1})$','Interpreter','latex')
yline(tol);
hold off;


%% Rescale the data back to original values
for k=1:numMarginal_Time
    for j=1:numMarginal_CPU
%         rawD{k,j}(:,1) = rawD{k,j}(:,1) / scaling_factors{k,j}(1);
%         rawD{k,j}(:,2) = rawD{k,j}(:,2) / scaling_factors{k,j}(2);
%         rawD{k,j}(:,3) = rawD{k,j}(:,3) / scaling_factors{k,j}(3);
        rawD{k,j}(:,4) = rawD{k,j}(:,4) / scaling_factors{k,j}(4);
    end
end


%% Figure 2 : Plot interpolated marginals at specified times
%==========================================================================
num_interp = numel(taus);
r = 1e-3;
res = 0.01; gridxi = 0:res:scaled_maxval; gridc = 1:5; % (!) Grid parameters
[x1,x2,x3]    = ndgrid(gridxi,gridxi,gridxi);
[x4]          = ndgrid(gridc);
x1 = x1(:,:)';
x2 = x2(:,:)';
x3 = x3(:,:)';
x4 = x4(:,:)';
pts_xi = [x1(:) x2(:) x3(:)];
pts_c  = [x4(:)];
[x1,x2,x3,x4] = ndgrid(gridxi,gridxi,gridxi , gridc);
x1 = x1(:,:)';
x2 = x2(:,:)';
x3 = x3(:,:)';
x4 = x4(:,:)';
pts    = [x1(:) x2(:) x3(:) x4(:)];

marginals        = cell(num_interp, num_CPUs);
marginals_stats  = cell(num_interp, num_CPUs);
marginals_scales = cell(num_interp, num_CPUs);

% ts   = (0:(numIntermediate+1))/(numIntermediate+1);
ilocs    = cell(num_interp, num_CPUs);
imags    = cell(num_interp, num_CPUs);
weights  = cell(num_interp, num_CPUs);

nS       = 500;                     % Number of downsamples

dlocs    = cell(num_interp, num_CPUs);      % Downsampled locs
dmags    = cell(num_interp, num_CPUs);      % Downsampled mags
dweights = cell(num_interp, num_CPUs);      % Downsampled weights (for graphing)

for k=1:numel(taus)
    tic;
    fprintf('k=%d\n', k);
        
    tau = taus(k);
    % Find correct M to use
    M_ind = find(marg_times>tau,1)-1;
    % Find bounds in interpolation interval
    tau_int_b = marg_times(M_ind);
    tau_int_e = marg_times(M_ind+1);
    tau_scaled = (tau-tau_int_b) / (tau_int_e-tau_int_b);
    
    for p=1:num_CPUs
        ilocs{k,p}    = zeros(n^2,4);
        imags{k,p}    = zeros(n^2,1);
        weights{k,p}  = zeros(n^2,1);
        dlocs{k,p}    = zeros(nS,4);
        dmags{k,p}    = zeros(nS,1);
        dweights{k,p} = zeros(nS,1);
        M_int = M{M_ind,p};
        
        % Perform interpolation
        % =====================
        for i=1:n
            for j=1:n
                ilocs{k,p}((i-1)*n+j,:) = (1-tau_scaled)*rawD{M_ind,p}(i,:)+tau_scaled*rawD{M_ind+1,p}(j,:);
                imags{k,p}((i-1)*n+j)   = M_int(i,j);
            end
        end
        marginals_scales{k,p} = (1-tau_scaled)*scaling_factors{M_ind,p}+tau_scaled*scaling_factors{M_ind+1,p};
        % =====================
        
        % Perform downsamping
        % ===================
        t_locs = ilocs{k,p};
        t_mags = imags{k,p};
        for i=1:nS
            % disp(i);
            dlocs{k,p}(i,:) = t_locs(1,:);
            dmags{k,p}(i) = t_mags(1);
            t_locs(1,:) = [];
            t_mags(1) = [];
            [minValues,cIs] = mink(vecnorm(dlocs{k,p}(i,:)-t_locs,2,2), n^2/nS-1);
            for j=1:(n^2/nS-1)
                dlocs{k,p}(i,:) = (dmags{k,p}(i)*dlocs{k,p}(i,:) + t_mags(cIs(j)).*t_locs(cIs(j),:)) ...
                    / (dmags{k,p}(i)+t_mags(cIs(j)));
                dmags{k,p}(i) = dmags{k,p}(i) + t_mags(cIs(j));
            end
            t_locs(cIs,:) = [];
            t_mags(cIs)   = [];
        end
        for j=1:nS
            dweights{k,p}(j) = sum( (vecnorm(dlocs{k,p}(j,:)-dlocs{k,p},2,2) < r) .* dmags{k,p} );
        end
        dweights{k,p} = dweights{k,p} / nS;
        % ===================

        % Perform gridding
        % ================
        sigma = [];
        sigma = std(dlocs{k,p},0,1);
        bw = sigma * (4/((4+2)*size(dlocs{k,p},1)))^(1/(411+4)); bw(bw==0)=0.01; % zero bandwidth causes issues
        marg_fulldim = mvksdensity(dlocs{k,p},pts,'Bandwidth',bw);
        % ================
        
        % Compute 'denominator' of conditional probability
        marg_ctx = get_context_marg_iCPUs(gridc,pts,marg_fulldim,size(gridc,2));
        
        % Find distribution of \xi conditioned on c
        marg_c = cell(length(gridc),1);
        marg_c_stats = cell(length(gridc),1);
        for c=gridc
            marg_atc = get_xi_marg_iCPUs(gridxi,gridxi,gridxi,c,pts,marg_fulldim,size(gridxi,2));
            marg_c{c} = marg_atc / marg_ctx(c);
            
%             % Dirty mean calculation
%             marg_c_mean = 0;
%             for i=1:numel(gridxi)
%                 for j=1:numel(gridxi)
%                     for kk=1:numel(gridxi)
%                         loc = [gridxi(i), gridxi(j), gridxi(kk)];
%                         
%                         marg_c_mean = marg_c_mean + marg_c{c}(i,j,kk)*loc;
%                     end
%                 end
%             end
%             marg_c_mean = marg_c_mean / sum(marg_c{c},"all")
%             % ----------------------
            % Alternative - report highest probability value
            [~,I] = max(marg_c{c},[],"all","linear");
            [I1,I2,I3] = ind2sub(size(marg_c{c}),I);
            marg_c_mean = [gridxi(I1), gridxi(I2), gridxi(I3)];
            % ----------------------------------------------
            
            % marg_c_mean = mean(pts_xi.'*reshape(marg_c{c},numel(gridxi)^3,1)/sum(marg_c{c},"all"),2).';
            marg_c_std  = std(pts_xi,reshape(marg_c{c},numel(gridxi)^3,1),1);
            % Statistics of marginal
            marg_c_stats{c} = [marg_c_mean ; marg_c_std];
        end
        
        marginals{k,p} = marg_c;
        marginals_stats{k,p} = marg_c_stats;
    end
    toc
end

%% Export marginals and needful information
save("data_out/MSBP_solution/MSBPsol_1015_iCPUs.mat","marginals", ...
     "marginals_stats", "marginals_scales", "pts", "pts_xi", "pts_c", ...
     "gridxi", "gridc", "taus", "instr_scale", "llcreq_scale", "llcmiss_scale");
