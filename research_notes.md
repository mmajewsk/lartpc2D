## Research Log:

24.11.2019

I need to implement sample wieght map. Im working on classical convolutions.

** Sampling distribution of decision ? **

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

#30.01.2020
I had the wrong categorisation mode working.

#03.02.2020
So it does not learn well, lest run it again after trail aaf733
Im splitting this project into two repos.
You might need to do File -> Settings -> Project -> Project structure -> Add content root
in pychar or use sys.path.append to use this dependancy

Thats it for today but need to remove and add submodule again.

#04.02.2020

Fixing some things in lartpc_game.
Now, I will attempt to read both networks at the same time.
Ok, worked on docs on lartpc_game.
I need more time to investigate what is going on with movement network, since they do not work as nice as they used to.

#05.02.2020

Trying this model;
2020-02-03 13:41:49
Saw it in replay, "not great, not terrible", its ok to go.
There is unresolved issue of reading two models:
(mov+cat) == model and (mov+cat) == target_model --> in actor

#26.02.2020

The joined model after the refactor works, i can experiment with it now.
The model picking code still needs some attention.

#01.04.2020 
so the next step is to do A2C agent, which is actually policy gradient based
also, what you have implemented is DDQN

#16.04.2020
Working on a2c

#18.04.2020
Still working on a2c comming from this examples
https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py
The important not here is that it learns after each episode.
I think I read about this being different from dqn, I must check that.
Yes, this was referenced here, as i thought it means that we don't need multi
armed bandit.

Ok so, I ne ed to rewrite actions and action factory
The idea for now to implement is to make so that f.e. action factory would bethe
only place that creates the actions.
And that factory is stored within the agent 

Ok, I rewrote ModelAction2D to QAction2D, next step is to create PolicyAction2D



I am implementing a2c training, the std, normalisation is weird, but i think the
authour did not know how to comprehend div by 0.
Anyway, Im going to keep it that way

The previous implementation used trace length equal to one, since there was no
need to use next steps. Im gonna investigate if its not an error now. (Its not)

# 28.04.2020

So what's left to do is to pick up an architecture for actor and critic.
Tried to find how A2C is defined in openai baseline https://github.com/openai/baselines
but the code is really making that hard to pull off

This is bearable but small https://github.com/germain-hug/Deep-RL-Keras
This is pytorch but seems ok https://github.com/TianhongDai/reinforcement-learning-algorithms/

# 29.04.2020

So it seems that the only difference in learning is that when you use dqn you
have to use mse.
In case of actor critic categorical crossentropy is the same as the actual
learning function.
I should check if tweaking existing dqn would change it into simple policy gradient.
Im starting to rewrite train script.

# 30.04.2020
In order to make that in classic a2c sense, I have to make it so that experience
is not repeated, and one sars is learned only once.
Which after short thought, I will not do, instead I will try to learn it
batches.
@TODO This is important for the potential paper. It might be that this is A3C
Whats left is to test this darn thing.

# 05.05.2020
So the problem is that i have might mistakenly converted with "from_game" action
method, check that out !
Runs, now lets find out if it trains at all and fix logging 

# 07.05.2020

# 25.05.2020
So A2C is showing negative loss, this seems to be because of the negative
reward.
BUT, the actor target must be categorical [0.0-1.0]. Im looking for problems
with negative rewards if tis problem is solved somehow or is it done exactly the same.

# 26.05.2020
So the pong example shows that the negative reward is present also.

# 08.06.2020
 https://github.com/pythonlessons/Reinforcement_Learning
 So i was talking about this example, but openai gym stable baselines has alos
 an implementation that i can try.
 https://github.com/hill-a/stable-baselines

Ok so negative loss is ok
https://stable-baselines.readthedocs.io/en/master/guide/examples.html?highlight=pong#id2
Per the example in google colab, but ill investigate what does mean entropy
loss, and what does it mean, and how it is implemented here.

Negative loss is ok, but makes a2c perform badly. Lets just focus back at DQN,
and show that A2C does not really have much sense.
