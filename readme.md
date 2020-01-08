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

124010
First run on GPU on Home Cluster.

125128
Increasing epochs to 100 and steps to 200
This shows signs of overfitting after 10 epoch

134448
Increasing batch to 128 
That did not help much

134219  
increasing dropout to 0.4
This helped a lot, altough the plateu is still there.

135924
Increasing to 0.5.
The plateu is stable (no overfitting), but it is stuck before 0.6 acc.

144130
decrease size of dense layers. ((9*2)**2)
slight increase overall

150400
decreasing to (9**2), in the meantime, adding split for the data 
Just slower learning curve, no plateu.

#13.12.2019

170228
I enhanced number of samples, lets see how that influences the network,
Going back to ((9*2)**2)

171259
lets change it to 14**2
This changes nothing, it is a problem of different matter.

170228
Lets add one more layer and see. and go back to (9*2)**2, we should is iff layers were limiting factor.
Does not change anything.

changing class weights by power of 2
This did not help at all as well

#14.12.2019

Lets change it to 3x3 input 1x1 output. And go back with weights.

#19.12.2019

Ok, changes to the output were applied, initial trials show improvement.
This means that its hard to guess 3x3 from 3x3.

#20.12.2019

The experimentation starts now.

130216
First one without the nans. I needed to add batch normalisation.
Looks ok, now its time to introduce validation as well.

135906
Added validation at 100 steps.
It seems to be doing really good.

xxxx
Increased dropout to 0.5, increased val steps to 400
Stopping. 400 is waay to much, this includes batches.

xxxx
Going to do 40. Stopping again, val acc started ad 0.01
Trying 80

Lost results, but the slope was much less steeper.

153856
0.0 dropout, incresed epochs to 200 to see when it will overfit
noticed some unstability over 120 in validation, increasing validation to 100

172138
300 epochs, val steps 100
it seems that adams learning rate diverges.

212148
changing optimiser to sgd with lr 0.00001 mom 0.9 nestr true
It seems the best thing so far, also really saturated.

#23.12.2019

205120
the same run, just dumped model every nth epoch

Thats it for the conv net. Its time to incorporate this to RL.

#08.01.2020

So the last model in rl uses window of 5x5, and so i will modify classic conv to match this.

First of all I need to recalculate weights for the classes.