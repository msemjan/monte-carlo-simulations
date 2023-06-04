% Reads files and calculates mean values

sufix                   = '';    % suffix for saved files
calculate_energy        = false; % calculate energy of a given configuration for verification
draw_lattice            = false;  % make a figure?
draw_loc_e              = false; % local energy of triangular plaquettes
draw_hist               = false; % histogram of values of spins
draw_sf                 = true;  % structure factor
multilattice            = -1;    % number of copies of a lattice
save_snaps_loc          = strcat('./figs-snaps-', date,'/'); % save directory
save_snaps_mat_loc      = strcat('./mats-snaps-', date,'/'); % save directory

mkdir(save_snaps_loc);
mkdir(save_snaps_mat_loc);
% J = -1;
h = 0;

d = dir();
for i = 3:length(d)
    if strfind(d(i).name, 'SIAKL')
        fprintf('%d: %s\n', i, d(i).name);
    end
end

files_to_process = input('\nSelect lattice to load!\n');

for file_idx = files_to_process
    lattice     = load(strcat(d(file_idx).name, '/lattice', sufix), '-ascii');
    sim_info    = load(strcat(d(file_idx).name, '/sim_config', sufix), '-ascii');
    
    latticeSize = [sim_info(1) sim_info(1) sim_info(2)/2];
    volume      = 3 * sim_info(1)^2 * sim_info(2);
    
    [tokens,match] = regexp(d(file_idx).name, 'J1_([\-0-9\.]+)', 'tokens','match');
    J1 = str2double(cell2mat(tokens{1}));
    [tokens,match] = regexp(d(file_idx).name, 'J2_([\-0-9\.]+)', 'tokens','match');
    J2 = str2double(cell2mat(tokens{1}));
    [tokens,match] = regexp(d(file_idx).name, 'MC_([0-9]+)x[0-9]+x[0-9]+', 'tokens','match');
    L = str2double(cell2mat(tokens{1}));
    title_ = strcat('J_1=',num2str(J1), ...
           ', J_2=', num2str(J2), ', L=', num2str(L));
    save_fig_suffix = strcat('_J1_',num2str(J1), ...
        '_J2_', num2str(J2), '_L', num2str(L)); % ,'_layer'sum0
    
    s1 = permute(reshape(lattice(:,1), latticeSize), [2 1 3]);
    s2 = permute(reshape(lattice(:,2), latticeSize), [2 1 3]);
    s3 = permute(reshape(lattice(:,3), latticeSize), [2 1 3]);
    s4 = permute(reshape(lattice(:,4), latticeSize), [2 1 3]);
    s5 = permute(reshape(lattice(:,5), latticeSize), [2 1 3]);
    s6 = permute(reshape(lattice(:,6), latticeSize), [2 1 3]);
    
%     addpath ~/usr/dev/matlab/

    if(draw_lattice)
       sum_layers;
       h = gca;
       title(title_);   
       if exist('save_fig_suffix') && exist('save_snaps_loc')
         saveas(h, strcat(save_snaps_loc, 'snap', save_fig_suffix, '.fig'),'fig'); 
         saveas(h, strcat(save_snaps_loc, 'snap', save_fig_suffix, '.eps'),'epsc'); 
       end
    end
    
    if(draw_loc_e)
       loc_energy;
%        figure; imagesc(energy(:,:,1)); colorbar();
       figure; imagesc(tri1); colorbar; title(strcat(title_, ' \nabla')); axis xy;
       
       h = gca;
%        title(title_);   
       if exist('save_fig_suffix') && exist('save_snaps_loc')
         saveas(h, strcat(save_snaps_loc, 'loc_e1_', save_fig_suffix, '.fig'),'fig'); 
         saveas(h, strcat(save_snaps_loc, 'loc_e1_', save_fig_suffix, '.eps'),'epsc'); 
       end
       
       figure; imagesc(tri2); colorbar; title(strcat(title_, ' \Delta')); axis xy;
              h = gca;
%        title(title_);   
       if exist('save_fig_suffix') && exist('save_snaps_loc')
         saveas(h, strcat(save_snaps_loc, 'loc_e2_', save_fig_suffix, '.fig'),'fig'); 
         saveas(h, strcat(save_snaps_loc, 'loc_e2_', save_fig_suffix, '.eps'),'epsc'); 
       end
    end

    if(draw_loc_e)
       loc_energy;
%        figure; imagesc(energy(:,:,1)); colorbar();
       figure; imagesc(tri1); colorbar; title(strcat(title_, ' \nabla')); axis xy;
       
       h = gca;
%        title(title_);   
       if exist('save_fig_suffix') && exist('save_snaps_loc')
         saveas(h, strcat(save_snaps_loc, 'loc_e1_', save_fig_suffix, '.fig'),'fig'); 
         saveas(h, strcat(save_snaps_loc, 'loc_e1_', save_fig_suffix, '.eps'),'epsc'); 
       end
       
       figure; imagesc(tri2); colorbar; title(strcat(title_, ' \Delta')); axis xy;
              h = gca;
%        title(title_);   
       if exist('save_fig_suffix') && exist('save_snaps_loc')
         saveas(h, strcat(save_snaps_loc, 'loc_e2_', save_fig_suffix, '.fig'),'fig'); 
         saveas(h, strcat(save_snaps_loc, 'loc_e2_', save_fig_suffix, '.eps'),'epsc'); 
       end
    end    
    
    if(draw_hist)
        figure; hist(lattice(:), 25); title(strcat(title_, ' Lattice spin distribution')); 
        h = gca;
%        title(title_);   
       if exist('save_fig_suffix') && exist('save_snaps_loc')
         saveas(h, strcat(save_snaps_loc, 'hist_', save_fig_suffix, '.fig'),'fig'); 
         saveas(h, strcat(save_snaps_loc, 'hist_', save_fig_suffix, '.eps'),'epsc'); 
       end
    end
    

   if exist('save_fig_suffix') && exist('save_snaps_mat_loc')
     save(strcat(save_snaps_mat_loc, 'snap_', save_fig_suffix, '.mat'), ...
         's1', 's2', 's3', 's4', 's5', 's6', 'sim_info', '-v7.3'); 
   end

    
    if(calculate_energy)
        time_series = load(strcat(d(file_idx).name, '/ts', sufix), '-ascii')';
        num_sweeps = sim_info(2);
        num_temp   = sim_info(3);
        
        egpu  = reshape(time_series(1, :), [num_sweeps, num_temp]);
        mgpu1 = reshape(time_series(2, :), [num_sweeps, num_temp]);
        mgpu2 = reshape(time_series(3, :), [num_sweeps, num_temp]);
        mgpu3 = reshape(time_series(4, :), [num_sweeps, num_temp]);
        
        s2_s = circshift(s2,[0 +1]);
        s3_s = circshift(s3,[+1 0]);
        
        m1       = sum(sum(s1));
        m2       = sum(sum(s2));
        m3       = sum(sum(s3));
        two_spin = sum(sum( s1 .* ( s2 + s3 + s2_s + s3_s ) ... 
                 + s2 .* ( s3 + circshift(s3,[+1 -1]) )));
             
        energy   = -J * two_spin - h * ( m1 + m2 + m3 );
        fprintf('Energy is %f and it should be %f (Diff=%f)\n', egpu(end, end)/volume, energy/volume, egpu(end, end)/volume - energy/volume);
        fprintf('M1 is %f and it should be %f (Diff=%f)\n', mgpu1(end, end)/volume, m1/volume, mgpu1(end, end)/volume - m1/volume);
        fprintf('M2 is %f and it should be %f (Diff=%f)\n', mgpu2(end, end)/volume, m2/volume, mgpu2(end, end)/volume - m2/volume);
        fprintf('M3 is %f and it should be %f (Diff=%f)\n', mgpu3(end, end)/volume, m3/volume, mgpu3(end, end)/volume - m3/volume);
    end
end
