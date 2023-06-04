% Code for calculation of natural logarithm of the partition function from the
% obtained energy density estimate.

PREDEF_VAL = 500;

% i .. tempereture index
lnZ = zeros(length(T),1);

% last term
if v(length(EE))-v(length(EE)-1) < PREDEF_VAL
  lnZ(i) = 1 + exp(v(length(EE))-v(length(EE)-1)); 
else
  lnZ(i) = v(length(EE))-v(length(EE)-1);
end

% other terms, except the first one
for j = (length(EE)-1):-1:2
  tmp = v(j)-v(j-1)+lnZ(i); 
  if tmp < PREDEF_VAL
    lnZ(i) = 1 + exp(tmp);
  else
    lnZ(i) = exp(tmp);
  end
end
  
% first term
if v(1) < PREDEF_VAL
  lnZ(i) = v(1) + exp(v1)*(1+lnZ(i);
else
  lnZ(i) = v(1)
end
