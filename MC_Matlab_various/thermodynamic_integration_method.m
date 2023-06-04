%   thermodynamic_integration_method(beta,dBeta,energy,N,s)
%       where
%           entropy = N*log(2*s+1) + beta.*energy - cumtrapz(dBeta,energy);
%
function entropy = thermodynamic_integration_method(beta, dBeta, energy, N, s)
    % Calculates entropy 
    entropy = N*log(2*s+1) + beta.*energy - cumtrapz(dBeta,energy);
end