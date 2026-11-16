function y = nanvar(x)
% Variance of non-NaN values. Octave statistics 1.6.3 does not implement nanvar.
  x = x(~isnan(x));
  if numel(x) < 2
    y = NaN;
  else
    y = var(x);
  end
end
