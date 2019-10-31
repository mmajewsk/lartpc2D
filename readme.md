## Requirements


```
conda install -c conda-forge opencv=4.1.0
conda install pandas
conda install scipy
conda install matplotlib
```


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