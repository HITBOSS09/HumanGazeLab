# Temporal uncertainty

The quantities below are computed on the frozen B3 maps: the leave-one-observer density at one released index, with the frozen bandwidth and floor. COCO-FreeView validation and OSIE held-out images were already inspected, so these summaries are exploratory. Tables: `outputs/protocolshift_b3_v1/entropy_by_index.csv`, `confident_but_poor.csv`, and `observer_spread_by_index.csv`.

## What each quantity is

Information gain is the log2 ratio of the probability in the bin that contains the fixation to the uniform probability on the grid. It is a score of one bin, not a continuous-pixel likelihood.

Predictive entropy is the Shannon entropy, in bits, of that discrete bin distribution. It is low when the map is peaked and high when the map is close to uniform. The maximum on the COCO-FreeView grid is log2(1092) = 10.09 bits. On the OSIE grid it is log2(300) = 8.23 bits. Entropy of a binned map is not a confidence interval for a continuous density, and it is not a posterior over model parameters. No parameter posterior was computed.

Half-mass coverage asks whether the fixation's bin is among the highest-probability bins that together hold half of the discrete probability. Ties on a flat map are left missing rather than assigned an arbitrary region.

Per-observer variation of the score is the standard deviation of the subject-equal means. Per-image variation is the standard deviation of image means. Both are in `b3_controls.csv`.

Inter-observer spatial disagreement is the standard deviation, across observers, of the fixation coordinates at one index on one image, then averaged across images. That quantity does not use the model.

Limited reference support is the count of other observers at that index. When the count is zero, B3 returns the uniform distribution and information gain is zero.

## Confident and poor

The rule was fixed for this summary as entropy at or below the 20th percentile of the same rows and information gain below zero. The percentile is computed on the same fixations, so it is a description, not a threshold chosen on a separate set.

On COCO-FreeView validation, 9,281 of 86,811 fixations (10.7%) meet the rule. Their mean information gain is −3.841 bits, close to the floor log2(0.05) = −4.322. The other fixations have mean information gain 0.079. Mean entropy is 6.928 bits. The fixation bin is inside the half-mass region for 20.4% of fixations.

On OSIE held-out images, 970 of 20,685 fixations (4.7%) meet the rule. Their mean information gain is −3.758 bits. The other fixations average 2.051 bits. Mean entropy is 5.308 bits. Half-mass coverage is 41.7%.

At COCO-FreeView index 5, where the fixation-mean information gain is already −0.131 bits, mean entropy is 7.062 bits and 47.1% of information gains are negative, while the mean number of other observers is still 8.87. The misses are not confined to indices that have lost their reference observers.

## Four different uncertainties

- Model uncertainty, in the sense of a posterior over parameters, was not estimated. Entropy here is a property of one fitted map.
- Inter-observer variability is the coordinate spread in `observer_spread_by_index.csv`. At the first scored index, mean horizontal standard deviation is 184 px on COCO-FreeView (0.110 of canvas width) and 80 px on OSIE index 0 (0.100 of canvas width). By index 5 the fractions of width are 0.174 and 0.198. Pixel spread and canvas-fraction spread are not the same statement.
- Limited reference support is `n_other`. It stays near 9 on COCO-FreeView through index 9 and is 14 at OSIE index 0. It falls only later.
- Experimental measurement uncertainty includes tracker noise, the unnamed duration unit, the unresolved OSIE duration cutoff, and the missing degrees-per-pixel conversion. None of those was turned into a likelihood term.
