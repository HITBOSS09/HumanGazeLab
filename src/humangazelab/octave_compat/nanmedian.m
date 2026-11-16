function y = nanmedian(x)
% Median of non-NaN values. Octave statistics 1.6.3 does not implement nanmedian.
  x = x(~isnan(x));
  if isempty(x)
    y = NaN;
  else
    y = median(x);
  end
end
