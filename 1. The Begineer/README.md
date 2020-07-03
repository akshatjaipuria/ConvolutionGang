## Task
WRITE AND TRAIN A CNN FOR MNIST DATASET SUCH THAT IT ACHIEVES
- 99.4% validation accuracy
- Less than 20k Parameters 
- Less than 20 Epochs
- No fully connected layer
## Reslut
```
Maximum validation accuracy reached : 99.43 at epoch 17th  
Total number of parameters : 10,040
```
## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
       BatchNorm2d-2            [-1, 8, 28, 28]              16
           Dropout-3            [-1, 8, 28, 28]               0
            Conv2d-4           [-1, 16, 28, 28]           1,152
       BatchNorm2d-5           [-1, 16, 28, 28]              32
           Dropout-6           [-1, 16, 28, 28]               0
            Conv2d-7           [-1, 16, 28, 28]           2,304
       BatchNorm2d-8           [-1, 16, 28, 28]              32
           Dropout-9           [-1, 16, 28, 28]               0
        MaxPool2d-10           [-1, 16, 14, 14]               0
           Conv2d-11            [-1, 8, 14, 14]             128
      BatchNorm2d-12            [-1, 8, 14, 14]              16
          Dropout-13            [-1, 8, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           1,152
      BatchNorm2d-15           [-1, 16, 14, 14]              32
          Dropout-16           [-1, 16, 14, 14]               0
           Conv2d-17           [-1, 16, 14, 14]           2,304
      BatchNorm2d-18           [-1, 16, 14, 14]              32
          Dropout-19           [-1, 16, 14, 14]               0
        MaxPool2d-20             [-1, 16, 7, 7]               0
           Conv2d-21              [-1, 8, 7, 7]             128
      BatchNorm2d-22              [-1, 8, 7, 7]              16
          Dropout-23              [-1, 8, 7, 7]               0
           Conv2d-24             [-1, 16, 5, 5]           1,152
      BatchNorm2d-25             [-1, 16, 5, 5]              32
          Dropout-26             [-1, 16, 5, 5]               0
           Conv2d-27             [-1, 10, 3, 3]           1,440
AdaptiveAvgPool2d-28             [-1, 10, 1, 1]               0
================================================================
Total params: 10,040
Trainable params: 10,040
Non-trainable params: 0
----------------------------------------------------------------
```

## Logs:  
```
Epoch: 1/20..
loss=0.16503207385540009 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.61it/s]
Train set: Average loss: 0.1034, Accuracy: 58318/60000 (97.197%)
Test set: Average loss: 0.0986, Accuracy: 9751/10000 (97.510%)

Epoch: 2/20..  
loss=0.046510498970746994 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.36it/s]
Train set: Average loss: 0.0664, Accuracy: 58856/60000 (98.093%)
Test set: Average loss: 0.0625, Accuracy: 9824/10000 (98.240%)

Epoch: 3/20..  
loss=0.11855870485305786 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.66it/s]
Train set: Average loss: 0.0568, Accuracy: 58951/60000 (98.252%)
Test set: Average loss: 0.0531, Accuracy: 9850/10000 (98.500%)

Epoch: 4/20..  
loss=0.06984870880842209 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.43it/s]
Train set: Average loss: 0.0386, Accuracy: 59292/60000 (98.820%)
Test set: Average loss: 0.0372, Accuracy: 9881/10000 (98.810%)

Epoch: 5/20..  
loss=0.03781529888510704 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.92it/s]
Train set: Average loss: 0.0382, Accuracy: 59338/60000 (98.897%)
Test set: Average loss: 0.0368, Accuracy: 9886/10000 (98.860%)

Epoch: 6/20..  
loss=0.013056889176368713 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.78it/s]
Train set: Average loss: 0.0334, Accuracy: 59377/60000 (98.962%)
Test set: Average loss: 0.0313, Accuracy: 9910/10000 (99.100%)

Epoch: 7/20..  
loss=0.040621861815452576 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.71it/s]
Train set: Average loss: 0.0308, Accuracy: 59444/60000 (99.073%)
Test set: Average loss: 0.0316, Accuracy: 9896/10000 (98.960%)

Epoch: 8/20..  
loss=0.023167630657553673 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.05it/s]
Train set: Average loss: 0.0282, Accuracy: 59486/60000 (99.143%)
Test set: Average loss: 0.0300, Accuracy: 9903/10000 (99.030%)

Epoch: 9/20..  
loss=0.04272851347923279 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.62it/s]
Train set: Average loss: 0.0288, Accuracy: 59448/60000 (99.080%)
Test set: Average loss: 0.0287, Accuracy: 9908/10000 (99.080%)

Epoch: 10/20..  
loss=0.0061875381506979465 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.27it/s]
Train set: Average loss: 0.0276, Accuracy: 59467/60000 (99.112%)
Test set: Average loss: 0.0306, Accuracy: 9898/10000 (98.980%)

Epoch: 11/20..  
loss=0.014786005020141602 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.96it/s]
Train set: Average loss: 0.0223, Accuracy: 59589/60000 (99.315%)
Test set: Average loss: 0.0253, Accuracy: 9920/10000 (99.200%)

Epoch: 12/20..  
loss=0.05867987498641014 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.38it/s]
Train set: Average loss: 0.0304, Accuracy: 59425/60000 (99.042%)
Test set: Average loss: 0.0336, Accuracy: 9901/10000 (99.010%)

Epoch: 13/20..  
loss=0.014841449446976185 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.20it/s]
Train set: Average loss: 0.0223, Accuracy: 59592/60000 (99.320%)
Test set: Average loss: 0.0261, Accuracy: 9917/10000 (99.170%)

Epoch: 14/20..  
loss=0.03210768103599548 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.04it/s]
Train set: Average loss: 0.0204, Accuracy: 59633/60000 (99.388%)
Test set: Average loss: 0.0241, Accuracy: 9922/10000 (99.220%)

Epoch: 15/20..  
loss=0.06482108682394028 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.76it/s]
Train set: Average loss: 0.0189, Accuracy: 59648/60000 (99.413%)
Test set: Average loss: 0.0222, Accuracy: 9932/10000 (99.320%)

Epoch: 16/20..  
loss=0.012506385333836079 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.84it/s]
Train set: Average loss: 0.0185, Accuracy: 59674/60000 (99.457%)
Test set: Average loss: 0.0221, Accuracy: 9922/10000 (99.220%)

Epoch: 17/20..  
loss=0.04388311505317688 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.77it/s]
Train set: Average loss: 0.0168, Accuracy: 59708/60000 (99.513%)
Test set: Average loss: 0.0219, Accuracy: 9943/10000 (99.430%)

Epoch: 18/20..  
loss=0.013021404854953289 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.09it/s]
Train set: Average loss: 0.0176, Accuracy: 59660/60000 (99.433%)
Test set: Average loss: 0.0218, Accuracy: 9927/10000 (99.270%)

Epoch: 19/20..  
loss=0.03376510739326477 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.64it/s]
Train set: Average loss: 0.0157, Accuracy: 59716/60000 (99.527%)
Test set: Average loss: 0.0203, Accuracy: 9925/10000 (99.250%)

Epoch: 20/20..  
loss=0.0334927961230278 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.77it/s]
Train set: Average loss: 0.0172, Accuracy: 59684/60000 (99.473%)
Test set: Average loss: 0.0214, Accuracy: 9933/10000 (99.330%)
```
