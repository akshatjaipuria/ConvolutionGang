This folder contains a stepwise analysis of building a CNN model for MNIST digits dataset. We perform this in five steps, with the following target:  
1. 99.4% Test accuracy.  
2. Less than or equal to 15 Epochs.  
3. Less than 10000 Parameters.  

## Notebook 1
```
Targets:  
1. Check if the code works properly.
2. Set up a basic model.
3. Analyzing the dataset. 

Results: 
Best train accuracy :  99.3%
Best test accuracy :  98.82%
Number of parameters :  120,608

Analysis:
1. Code works fine without any issues.
2. Model is heavier than what might be required for this dataset.
```
## Notebook 2
```
Targets:
1. Reduce the parameters count, close to the target.
2. Add batch normalization to improve the efficiency of the model.

Results: 
Best train accuracy :  99.92%
Best test accuracy :  99.17%
Number of parameters : 11,516

Analysis:
1. Less parameters, but yet greater than the target.
2. Slight improvement in test accuracy.
3. We clearly see over-fitting.
```
## Notebook 3
```
Targets:
1. Fixing over-fitting.
2. Use dropout, with appropriate drop percentage to tackle over-fitting.

Results: 
Best train accuracy :  99.15%
Best test accuracy :  99.21%
Number of parameters : 11,516

Analysis:
1. We have nearly eliminated over-fitting.
2. Number of parameters are still more than the target.
3. The target accuracy still isn't achieved.
```
## Notebook 4
```
Targets:
1. Replacing the last layer with GAP + 1x1 convolution to remove the big kernel.
2. Altering the layers a bit and changing the no. of parameters to fit in our range.

Results: 
Best train accuracy :  99.19%
Best test accuracy :  99.37%
Number of parameters : 7,432

Analysis:
1. Model is good, with parameters count within the target.
2. We are close to the target accuracy.
3. With lesser parameters, we also reduced the dropout value.
```
## Notebook 5
```
Targets:
1. To  get the desired accuracy.
2. Find the balance between the augmentation and dropout.
3. Do not over regularize the model and changing the channel count if required.

Results: 
Best train accuracy :  99.16%
Best test accuracy :  99.46%
Number of parameters : 8,832

Analysis:
1. Model works great and we have achieved the target at 10th epoch.
2. The most appropriate augmentations we found was a bit of rotation and random cut, using the 22x22 centre pixels with a probability of 10%.
3. Also, dropout value had to be reduced to 2%.
```

## Final Model's trainig epochs:
```
 0%|          | 0/469 [00:00<?, ?it/s]
EPOCH: 1
Loss=0.0670732781291008 Batch_id=468 Accuracy=89.80: 100%|██████████| 469/469 [00:15<00:00, 37.92it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0659, Accuracy: 9829/10000 (98.29%)

EPOCH: 2
Loss=0.12398222833871841 Batch_id=468 Accuracy=97.79: 100%|██████████| 469/469 [00:14<00:00, 32.87it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0383, Accuracy: 9893/10000 (98.93%)

EPOCH: 3
Loss=0.03551769629120827 Batch_id=468 Accuracy=98.28: 100%|██████████| 469/469 [00:15<00:00, 29.68it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0303, Accuracy: 9910/10000 (99.10%)

EPOCH: 4
Loss=0.021600447595119476 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:15<00:00, 30.90it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0344, Accuracy: 9899/10000 (98.99%)

EPOCH: 5
Loss=0.0048032947815954685 Batch_id=468 Accuracy=98.62: 100%|██████████| 469/469 [00:15<00:00, 32.79it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0264, Accuracy: 9922/10000 (99.22%)

EPOCH: 6
Loss=0.04661647602915764 Batch_id=468 Accuracy=98.74: 100%|██████████| 469/469 [00:16<00:00, 29.15it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0244, Accuracy: 9930/10000 (99.30%)

EPOCH: 7
Loss=0.061236124485731125 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:15<00:00, 30.26it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0313, Accuracy: 9899/10000 (98.99%)

EPOCH: 8
Loss=0.017667658627033234 Batch_id=468 Accuracy=98.86: 100%|██████████| 469/469 [00:14<00:00, 31.74it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0215, Accuracy: 9938/10000 (99.38%)

EPOCH: 9
Loss=0.06397314369678497 Batch_id=468 Accuracy=98.88: 100%|██████████| 469/469 [00:15<00:00, 32.83it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0209, Accuracy: 9935/10000 (99.35%)

EPOCH: 10
Loss=0.03540671989321709 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:16<00:00, 29.25it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0203, Accuracy: 9941/10000 (99.41%)

EPOCH: 11
Loss=0.009100005030632019 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:15<00:00, 30.78it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0198, Accuracy: 9939/10000 (99.39%)

EPOCH: 12
Loss=0.008323957212269306 Batch_id=468 Accuracy=99.06: 100%|██████████| 469/469 [00:15<00:00, 30.97it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9932/10000 (99.32%)

EPOCH: 13
Loss=0.03727330267429352 Batch_id=468 Accuracy=99.04: 100%|██████████| 469/469 [00:15<00:00, 31.05it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0208, Accuracy: 9932/10000 (99.32%)

EPOCH: 14
Loss=0.03586318716406822 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:15<00:00, 30.40it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0189, Accuracy: 9946/10000 (99.46%)

EPOCH: 15
Loss=0.07491981238126755 Batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:15<00:00, 30.92it/s]
Test set: Average loss: 0.0180, Accuracy: 9942/10000 (99.42%)
```
