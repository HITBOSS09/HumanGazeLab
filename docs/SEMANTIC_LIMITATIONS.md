# Limitations

These limits apply whether or not the descriptive tables show a gap. They are reasons not to treat a region-event score as a new gaze model.

## Annotations are not a partition

Objects overlap. Semantic labels overlap. A fixation can be positive for several labels at once. Each label is scored as its own binary event. The probabilities are not a distribution over labels, and they are not required to sum to one.

Unlabeled pixels are the complement of the annotated objects. Xu et al. (2014) did not extract sky, ocean, ground, wall, or regions they judged too small or too blurry. A fixation in that complement is not evidence that no meaningful object was there. The audit does not score background as a negative class.

Attribute values above 1 are kept in the manifest and then collapsed by the Matlab rule value > 0. The audit does not interpret 3 as "more face" than 1.

## The probability being tested is an integral

The model predicts a density on the fine grid. The event probability is the sum of the projected mass inside the mask. A well calibrated region event can hide a badly placed density inside the mask, and a badly calibrated region event can coexist with a density that still ranks fixations well. Information gain on the frozen table is a different number and is not recomputed here.

Log loss uses a clip at 1e-6. Brier score does not. The clip is only there to keep a probability of 0 or 1 from producing an infinite loss. It is not a calibrated probability.

## Dependence and reuse

Fifteen observers and 700 images generate the eligible fixations. An interval that resampled fixations would be too narrow. The reported intervals resample images, subjects, or both. They are not a mixed-effects model and they are not a causal effect of the semantic label.

OSIE performance of this checkpoint was already inspected in `outputs/osie_deepgaze3_v2`. Any split used here, including the five image folds of the area and center residual, is internal. It is not a confirmation set.

The three primary labels were named in the specification before scoring. The other labels were not. Looking at all twelve and then quoting the largest gap would be a different analysis. The tables keep the roles separate and do not adjust the three primary intervals for the exploratory family.

The failure images are the tails of a ranking that was fixed before the pictures were opened. They are examples of that ranking. They are not a random sample of errors, and they are not verified errors until the mask and the density have been checked. With two tails and two labels, the atlas is eight rows at most, fewer if a tie collapses them.

## Controls that are not a model

The area reference is the mask area on the 800×600 grid. The center reference is the frozen center-bias template, projected with the same mass-preserving map. The shifts keep approximate area and stay inside an 80-pixel radius, and they fail when the mask is too large to translate without losing 20% of its area. A large object can therefore have no matched control. That failure is reported. It is not filled in with a different shift.

The previous-fixation split is a fixed 100-pixel threshold. It was not chosen by scanning thresholds. The border rows change the outcome and leave the predicted mass unchanged, so they test label tolerance, not a second model.

The cross-fit residual removes a linear function of area fraction and centroid distance from the image-mean gap. A gap that shrinks there is explained by those two covariates only in that linear sense. It is not a proof that no semantic association exists, and a gap that remains is not a license to add a spatial correction.

## What a correction would not mean

An out-of-fold improvement in region-event Brier score or log loss would mean the binary event is easier to predict with the extra covariates. It would not mean the scanpath log-likelihood improved. A spatial correction has to be a normalized density on the same grid, scored with the same likelihood as the frozen model, on images that have not already been used to choose it. This audit does not do that.

ImageNet overlap of the OSIE photographs with the DenseNet-201 training images was not tested. The photographs are redistributed by the NUS-VIP release. The software license in that release is MIT for the code. That text is not treated here as a license for the photographs. Overlays are stored because the specification asked for them where redistribution is permitted; the release distributes the stimuli.
