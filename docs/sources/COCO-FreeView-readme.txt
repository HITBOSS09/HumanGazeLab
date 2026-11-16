Introduction:

COCO-FreeView is a laboratory-quality dataset of free viewing behavior. It contains the same natural images used in COCO-Search18, but labeled with 822,602 eye fixations from a free-viewing task. 
10 university students participated in the data collection. The experiment procedure was modified from COCO-Search18, with no target being cued, no response being required from the participant, and the viewing time for each image being fixed to 5 seconds.


If you use COCO-FreeView, please cite:

Chen, Y., Yang, Z., Chakraborty, S., Mondal, S., Ahn, S., Samaras, D., Hoai, M., & Zelinsky, G. (2022). Characterizing Target-Absent Human Attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5031-5040).

@inproceedings{chen2022characterizing, 
title={Characterizing Target-Absent Human Attention}, 
author={Chen, Yupei and Yang, Zhibo and Chakraborty, Souradeep and Mondal, Sounak and Ahn, Seoyoung and Samaras, Dimitris and Hoai, Minh and Zelinsky, Gregory}, 
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
pages={5031--5040}, 
year={2022} 
}


---------------v1.0----------------------  Date: 06/2022

Data format:

Fixation data is saved in .json format.
All images used in data collection were resized to 1680x1050 with zero padding and aspect ratio kept. 
Images can be downloaded at https://sites.google.com/view/cocosearch/

'trial_ind': index of the presentation of the image
'name': image name,
'subject': subject number,
'task': target object category corresponding to the same image in COCO-Search18,
'condition': target present or target absent condition corresponding to the same image in COCO-Search18,
'X': X coordinates of fixations from the first fixation to the last in this trial (including the initial center fixation),
'Y': Y coordinates of fixations from the first fixation to the last in this trial (including the initial center fixation),
'T': fixation durations from the first fixation to the last in this trial (including the initial center fixation),
'length': number of total fixations in this trial,
'split': training set = 'train', validation set = 'valid', testing set = 'test'.




