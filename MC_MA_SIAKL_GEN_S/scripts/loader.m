% This script reads data from simulations, calculates the mean values, and
% saves the results into a specified folder for further processing.
%
% To run, modify the parameters, and run 'loader' in Matlab.
% The files to process are specified in the working_data.m file.

save_prefix         = './means/'; % Directory where to save the calculated mean files (and anything else to prepend into the file name
sufix               = '';         % A string that is suffixed to the file names
discard             = 0.0;        % Specify a number > 0 of MCS to discard for thermalization
cal_chain_order     = true;       % Specify if the chain order parameter should be calculated
cal_chain_energy    = true;       % Specify if the thermal values of the energy of chains should be calculated

fprintf('=================================================================================\n');
fprintf('Starting...\n');

% Anonymous functions for calculating minimum, median, and maximum of an array
min_arr = @(x, y, z) reshape(min(   [x(:), y(:), z(:)]'), size(x));
mid_arr = @(x, y, z) reshape(median([x(:), y(:), z(:)]'), size(x));
max_arr = @(x, y, z) reshape(max(   [x(:), y(:), z(:)]'), size(x));

% Files for processing
d = working_data;

% Loop over all files
for file_idx = 1:length(d)
  try 
    % Getting info from the name
    [tokens,match] = regexp(d{file_idx}, 'J1_([\-0-9\.]+)', 'tokens','match');
    J1 = cell2mat(tokens{1});
    [tokens,match] = regexp(d{file_idx}, 'J2_([\-0-9\.]+)', 'tokens','match');
    J2 = cell2mat(tokens{1});
    [tokens,match] = regexp(d{file_idx}, 'MC_([0-9]+)x[0-9]+x[0-9]+', 'tokens','match');
    L = cell2mat(tokens{1});

    % Loading data
    try 
      time_series = load(strcat(d{file_idx}, '/ts', sufix),         '-ascii');
      sim_info    = load(strcat(d{file_idx}, '/sim_config', sufix), '-ascii');
      temperature = load(strcat(d{file_idx}, '/temp', sufix),       '-ascii');
    catch ME
      fprintf('Error while reading files in "%s" directory\n', d{file_idx});
      ME
      ME.stack
      continue;
    end

    % More info
    volume = 3 * sim_info(2) * sim_info(1)^2;
    latticeSize = [sim_info(1) sim_info(1) sim_info(2)/2];
    num_sweeps = sim_info(3);
    num_temp   = length(temperature); % sim_info(4);
    start      = ceil(discard*num_sweeps) + 1;

    % Calculating means
    if(~cal_chain_energy)
      energy_ts       = reshape(time_series(:, 1), [num_sweeps, num_temp]);
      energy   = mean(energy_ts(start:end, :));
      varEnergy = var(energy_ts(start:end, :));
      specificHeat   = ((1./temperature').^2) .* varEnergy / volume;
      V4 = 1 - mean(energy_ts(start:end, :).^4) ./ (3*mean(energy_ts(start:end, :).^2));
    else
      energy_ts       = reshape(time_series(:, 1), [num_sweeps, num_temp]);
      energy_chain_ts = reshape(time_series(:, 10), [num_sweeps, num_temp]);

      energy          = mean(energy_ts(start:end, :) + energy_chain_ts(start:end, :));
      specificHeat    = ((1./temperature').^2) .* var(energy_ts(start:end, :) + energy_chain_ts(start:end, :)) / volume;
      specificHeat_plane = ((1./temperature').^2) .* var(energy_ts(start:end, :)) / volume;
      specificHeat_chain = ((1./temperature').^2) .* var(energy_chain_ts(start:end, :)) / volume;
      
      V4 = 1 - mean(((energy_ts(start:end, :) + energy_chain_ts(start:end, :))).^4) ...
            ./ (3*mean((energy_ts(start:end, :) + energy_chain_ts(start:end, :)).^2).^2);
    end
  
    % Calculations of sublattice magnetizations
    m1_ts       = reshape(time_series(:, 2), [num_sweeps, num_temp]);
    m2_ts       = reshape(time_series(:, 3), [num_sweeps, num_temp]);
    m3_ts       = reshape(time_series(:, 4), [num_sweeps, num_temp]);
    m4_ts       = reshape(time_series(:, 5), [num_sweeps, num_temp]);
    m5_ts       = reshape(time_series(:, 6), [num_sweeps, num_temp]);
    m6_ts       = reshape(time_series(:, 7), [num_sweeps, num_temp]);

    m1       = mean( m1_ts(start:end, :) + m4_ts(start:end, :));
    varM1    = var(  m1_ts(start:end, :) + m4_ts(start:end, :));
    m2       = mean( m2_ts(start:end, :) + m5_ts(start:end, :));
    varM2    = var(  m2_ts(start:end, :) + m5_ts(start:end, :));
    m3       = mean( m3_ts(start:end, :) + m6_ts(start:end, :));
    varM3    = var(  m3_ts(start:end, :) + m6_ts(start:end, :));
  
    % Binder cumulants
    binder2  = 1 - mean(( m1_ts(start:end, :) + m2_ts(start:end, :)    ...
                        + m3_ts(start:end, :) + m4_ts(start:end, :)    ...
                        + m5_ts(start:end, :) + m6_ts(start:end, :)).^2 ...
               ) ./ ( 3 * ...
                mean(abs( m1_ts(start:end, :) + m2_ts(start:end, :)    ...
                        + m3_ts(start:end, :) + m4_ts(start:end, :)    ...
                        + m5_ts(start:end, :) + m6_ts(start:end, :))).^2 );

    binder4  = 1 - mean(( m1_ts(start:end, :) + m2_ts(start:end, :)    ...
                        + m3_ts(start:end, :) + m4_ts(start:end, :)    ...
                        + m5_ts(start:end, :) + m6_ts(start:end, :)).^4 ...
               ) ./ ( 3 * ...
                   mean(( m1_ts(start:end, :) + m2_ts(start:end, :)    ...
                        + m3_ts(start:end, :) + m4_ts(start:end, :)    ...
                        + m5_ts(start:end, :) + m6_ts(start:end, :)).^2).^2 );
  
    % Quantity Y (for definition, check this paper: SEMJAN, M., ŽUKOVIČ, M. “Absence of long-range order in a general spin-S kagome
    % lattice Ising antiferromagnet“. Phys. Lett. A (2020) 384, 126615.)
    Y = mean( (m1_ts(start:end, :) + m4_ts(start:end, :)).^2 ...
            + (m2_ts(start:end, :) + m5_ts(start:end, :)).^2 ...
            + (m3_ts(start:end, :) + m6_ts(start:end, :)).^2 );

    % Total magnetization
    ts = m1_ts + m2_ts + m3_ts + m4_ts + m5_ts + m6_ts;
    varM      = var(ts(start:end, :));

    % Susceptibility (total and for sublattices)
    susceptibility = (1./temperature') .* varM  / volume;
    sus1           = (1./temperature') .* varM1 / volume;
    sus2           = (1./temperature') .* varM2 / volume;
    sus3           = (1./temperature') .* varM3 / volume;

    minM_ts  =  min_arr( ...
                   m1_ts(start:end, :) + m4_ts(start:end, :), ...
                   m2_ts(start:end, :) + m5_ts(start:end, :), ...
                   m3_ts(start:end, :) + m6_ts(start:end, :) ...
               ); 
    midM_ts  = mid_arr( ...
                   m1_ts(start:end, :) + m4_ts(start:end, :), ...
                   m2_ts(start:end, :) + m5_ts(start:end, :), ...
                   m3_ts(start:end, :) + m6_ts(start:end, :) ...
               ); 
    maxM_ts  = max_arr( ...
                   m1_ts(start:end, :) + m4_ts(start:end, :), ...
                   m2_ts(start:end, :) + m5_ts(start:end, :), ...
                   m3_ts(start:end, :) + m6_ts(start:end, :) ...
               ); 

    minM     = mean(minM_ts); 
    midM     = mean(midM_ts); 
    maxM     = mean(maxM_ts); 

    varMinM  = var(minM_ts); 
    varMidM  = var(midM_ts); 
    varMaxM  = var(maxM_ts); 
    
    ts       = m1_ts + m2_ts + m3_ts + m4_ts + m5_ts + m6_ts;
    varM     = var(ts(start:end, :));

    try
      chain_ts = reshape(time_series(:, 8), [num_sweeps, num_temp]);
      chain     = mean(chain_ts(start:end, :));
      sus_chain = (1./temperature') .* var(chain_ts(start:end, :)) / volume;
    catch ME
      fprintf('Was unable to load chain_ts\n')
      chain_ts = [];
      sus_chain = [];
      ME.stack
    end

    try
      triangle_ts = reshape(time_series(:, 9), [num_sweeps, num_temp]);
      triangle = mean(triangle_ts(start:end, :)) / (2*str2num(L)^3);
      sus_tri  =  (1./temperature') .* var(triangle_ts(start:end, :)) / volume;
    catch ME
      fprintf('Was unable to load triangle_ts\n')
      triangle_ts = [];
      triangle = [];
      sus_tri = [];
      ME.stack
    end
    % if draw
    %   title_ = strcat('J_1=',num2str(J1), ...
    %     ', J_2=', num2str(J2), ', L=', num2str(L));
    %   save_fig_suffix = strcat('_J1_',num2str(J1), ...
    %     '_J2_', num2str(J2), '_L', num2str(L));
    %   draw_figures;
    % end

    % Loading the final configuration
    lattice     = load(strcat(d{file_idx}, '/lattice'), '-ascii');

    s1 = permute(reshape(lattice(:,1), latticeSize), [2 1 3]);
    s2 = permute(reshape(lattice(:,2), latticeSize), [2 1 3]);
    s3 = permute(reshape(lattice(:,3), latticeSize), [2 1 3]);
    s4 = permute(reshape(lattice(:,4), latticeSize), [2 1 3]);
    s5 = permute(reshape(lattice(:,5), latticeSize), [2 1 3]);
    s6 = permute(reshape(lattice(:,6), latticeSize), [2 1 3]);

    % Saving time series
    % save(strcat(save_prefix, d{file_idx}, '.mat'), 'energy_ts',...
    % 'energy_chain_ts', 'm1_ts', 'm2_ts', 'm3_ts', 'm4_ts', 'm5_ts',...
    % 'm6_ts','chain_ts', 'triangle_ts', 's1', 's2', 's3', 's4', 's5', 's6',...
    % 'temperature', 'num_sweeps', 'num_temp', '-v7.3');

    % Saving thermal averages
    try
      % Saving means
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'energy', 'm1', 'm2', 'm3', 'specificHeat', ... 
      'susceptibility', 'volume', 'temperature', 'sus1', 'sus2', 'sus3', 'V4', ...
      'minM', 'midM', 'maxM', 'varMinM', 'varMidM', 'varMaxM', 'varM',  'latticeSize', '-v7.3');
    catch ME
      fprintf('Unable to save means\n');
      ME
      ME.stack
    end

    try
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'specificHeat_plane', ...
      'specificHeat_chain', '-append');
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'specificHeat_chain', ...
      'specificHeat_chain', '-append');
    catch ME
      fprintf('Unable to save specific heats of planes or chains...\n');
      ME
      ME.stack
    end

    try
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'chain', 'sus_chain', '-append');
    catch ME
      fprintf('Unable to save chains...\n');
      ME
      ME.stack
    end

    try
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'triangle', 'sus_tri', '-append');
    catch ME
      fprintf('Unable to save triangles...\n');
      ME
      ME.stack
    end

    try
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'Y', '-append');
    catch ME
      fprintf('Unable to save Y...\n');
      ME
      ME.stack
    end

    try
      save(strcat(save_prefix, d{file_idx}, '.mat'), 'binder2', 'binder4', '-append');
    catch ME
      fprintf('Unable to save Y...\n');
      ME
      ME.stack
    end

  catch ME
    fprintf('Unable to read files in the directory "%s"\n', d{file_idx});
    ME
    ME.stack
  end

  fprintf('.');
end

fprintf('\n');

fprintf('=================================================================================\n');
fprintf('Finished working.\n');
fprintf('=================================================================================\n');
