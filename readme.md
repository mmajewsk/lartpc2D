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
Weight seem to be pretty much the same. keeping the old ones.

So i ran into the problem of the output of categorisation being of probability of 3 classes
Like [0.555, 0.12, 0.03], but desired output space is of categorical values.
I need to redo that.
But also will need to include that in replay process and bot, and visualisation.
Ok got that covered.
*iMPORTANT NOTE:* Visualisation pick now the max class to display

I also need to change observation to state like observation, because new network desires target to be included in learning.

#10.01.2020
So both models work. 
The loss does not make sense really for the movement. 
But this is a good start.

#13.01.2020
Im figuring out the loss
Right now it goes like this :

ETA: 0s -
loss: 1248.3007 -
output_movement_loss: 1247.9445 -
model_1_loss: 0.3562 -
output_movement_mse: 1247.9445 -
output_movement_mae: 17.5443 -
output_movement_acc: 0.3229 -
model_1_mse: 0.0928 -
model_1_mae: 0.1909 -
model_1_acc: 0.8333

lets see how it was before
It was like this:
32/32 [==============================] - 0s 64us/step - loss: 12.2132 - mae: 1.2951 - acc: 0.3438

Soo, I need to compare how rewards were given out. 
I think this might be due lack of binarisation.

#14.01.2020

I think binarisation has fixed it.  Also i changed it from keras to tensorflow.keras.
I need to check it tho, if it was binarised only from source or result as well.
Yes result was binarised, tho it used one-hot encoding.

#15.01.2020

Im letting it run for a little bit at home, so I could pick potential errors along the way.

#17.01.2020
Next objective is to fix replay
Next goal: train only movement.

#22.01.2020 
went back and recorded some sane models under 20200122113845_a54955582feaa6d15fc91e508e246b8b6c07fa77 

#23.01.2020 
Problem wit speed was solved.
Now the plotting from my own logging works as expected, but i am not sure about the mlflow one.
Im commiting this now.
So the old way was the right way. (meaning [[0.1,0.0,32]...[0.0,0.0,0.0]] - > [1,0,1]...[0,0,0] not: --> [1...0])
Actually, this is true, but only for the source, the result was the reducing way.

So new experimentation begins, with mlflow. 

#29.01.2020
After some experimentation, I am trying to use both and load them from disk.
Need to remember that tau learning may affect future results with categorisational part.
todays model up to 2020-01-29 18:25:09 are trained on wrong dataset