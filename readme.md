## Requirements


```
conda install -c conda-forge opencv=4.1.0
conda install pandas
conda install scipy
conda install matplotlib


### new
conda install jupyter conda scipy
pip install sklearn pandas pandas-datareader matplotlib pillow requests h5py
pip install --ignore-installed --upgrade tensorflow-gpu 
```

## Research Log:

https://openai.com/blog/emergent-tool-use/ Reading this.
Should take a look at attention.
https://arxiv.org/pdf/1909.07528.pdf - next the paper
Paper:

"intrinsic motivation" - ?? reading http://www.cs.cornell.edu/~helou/IMRL.pdf on this. 

autocurricula - ?? arXiv:1903.00742


not read yet:
https://arxiv.org/pdf/1707.06347.pdf

HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION https://arxiv.org/pdf/1506.02438.pdf

https://arxiv.org/pdf/1707.06347.pdf


kinda related maze: https://arxiv.org/pdf/1611.03673.pdf, https://arxiv.org/pdf/1810.02274.pdf
Must read on multiple objectives: https://arxiv.org/pdf/1809.04474.pdf


24.11.2019

I need to implement sample wieght map. Im working on classical convolutions.

** Sampling distribution of decision ? **

Weighed map;

https://stats.stackexchange.com/questions/284265/understanding-median-frequency-balancing
 |
 V
https://arxiv.org/pdf/1411.4734.pdf
also: Kampffmeyer_Semantic_Segmentation_of_CVPR_2016_paper.pdf
https://stackoverflow.com/questions/42591191/keras-semantic-segmentation-weighted-loss-pixel-map?rq=1
https://github.com/kwotsin/TensorFlow-ENet/blob/master/get_class_weights.py
## Experiments

 31.1.02.2019
 Begin  the experimentation with this log and git tracking.
 Run for about 800 iterations, proved to work a little bit,
 I will try to set a rule of stepping outside with a center 
 pixel outside to count as an not allowed move.
 
 R1 Changed to mentioned condition. First experimentation is not very conclusive.
 Increasing number of steps 6->8.
 R2.
 Really nice behaviour. 
 Reducing map iterations to 15->8.
 R3
 Ver interesting. Increasing steps 8->14
 R4
 The problem was that it was going back and forth a lot.
 Steps 14->10
 R5
 Same. Running again, for limited time.
 R6
 Almost random behaviour when stopped at 500 
 Going back to 8 
 R7
 As great as probably can be without recurrent layer.
 Lets increase gamma 0.5 -> 0.8 with steps 8->10.
 R8  Not to good. again, but less iterations.
 R9 forgot to turn it off, seems to be flying again
 R10 even worse. Lets change steps back to 10->8
 R11 Even better than anything so far. trying to give it een more time.
 R12 Worse than R11
 Increasing batch to 128 R13 better line of loss, and also nice behaviour 
 increasing steps again to 12 R14
 
 I think i need to read if batch reg would be useful here
 
 
 #12.12.2019
 
 122707
 So the first run just see how it works.