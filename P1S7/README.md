## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           9,216
             ReLU-10           [-1, 32, 32, 32]               0
      BatchNorm2d-11           [-1, 32, 32, 32]              64
          Dropout-12           [-1, 32, 32, 32]               0
        MaxPool2d-13           [-1, 32, 16, 16]               0
           Conv2d-14           [-1, 32, 16, 16]           9,216
             ReLU-15           [-1, 32, 16, 16]               0
      BatchNorm2d-16           [-1, 32, 16, 16]              64
          Dropout-17           [-1, 32, 16, 16]               0
        <b>Conv2d-18           [-1, 32, 16, 16]           9,216</b>
             ReLU-19           [-1, 32, 16, 16]               0
      BatchNorm2d-20           [-1, 32, 16, 16]              64
          Dropout-21           [-1, 32, 16, 16]               0
        MaxPool2d-22             [-1, 32, 8, 8]               0
           Conv2d-23             [-1, 16, 8, 8]             512
             ReLU-24             [-1, 16, 8, 8]               0
      BatchNorm2d-25             [-1, 16, 8, 8]              32
          Dropout-26             [-1, 16, 8, 8]               0
           Conv2d-27             [-1, 64, 8, 8]           9,216
             ReLU-28             [-1, 64, 8, 8]               0
      BatchNorm2d-29             [-1, 64, 8, 8]             128
          Dropout-30             [-1, 64, 8, 8]               0
           Conv2d-31             [-1, 64, 8, 8]             576
           Conv2d-32             [-1, 64, 8, 8]           4,096
             ReLU-33             [-1, 64, 8, 8]               0
      BatchNorm2d-34             [-1, 64, 8, 8]             128
          Dropout-35             [-1, 64, 8, 8]               0
        MaxPool2d-36             [-1, 64, 4, 4]               0
           Conv2d-37             [-1, 32, 4, 4]           2,048
             ReLU-38             [-1, 32, 4, 4]               0
      BatchNorm2d-39             [-1, 32, 4, 4]              64
          Dropout-40             [-1, 32, 4, 4]               0
           Conv2d-41             [-1, 32, 4, 4]           9,216
             ReLU-42             [-1, 32, 4, 4]               0
      BatchNorm2d-43             [-1, 32, 4, 4]              64
          Dropout-44             [-1, 32, 4, 4]               0
           Conv2d-45             [-1, 16, 4, 4]           4,608
             ReLU-46             [-1, 16, 4, 4]               0
      BatchNorm2d-47             [-1, 16, 4, 4]              32
          Dropout-48             [-1, 16, 4, 4]               0
        AvgPool2d-49             [-1, 16, 1, 1]               0
           Conv2d-50             [-1, 10, 1, 1]             170
================================================================
Total params: 63,866
Trainable params: 63,866
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.44
Params size (MB): 0.24
Estimated Total Size (MB): 3.69
----------------------------------------------------------------
```
## Training Epochs
```
 0%|          | 0/391 [00:00<?, ?it/s]
EPOCH: 1
Loss=1.274531602859497 Batch_id=390 Accuracy=41.45: 100%|██████████| 391/391 [00:18<00:00, 21.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.4156, Accuracy: 4865/10000 (48.65%)

EPOCH: 2
Loss=1.3408780097961426 Batch_id=390 Accuracy=60.32: 100%|██████████| 391/391 [00:18<00:00, 21.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 1.0631, Accuracy: 6215/10000 (62.15%)

EPOCH: 3
Loss=0.9597965478897095 Batch_id=390 Accuracy=67.10: 100%|██████████| 391/391 [00:18<00:00, 21.03it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.8596, Accuracy: 6970/10000 (69.70%)

EPOCH: 4
Loss=0.6965003609657288 Batch_id=390 Accuracy=71.31: 100%|██████████| 391/391 [00:18<00:00, 21.47it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7803, Accuracy: 7294/10000 (72.94%)

EPOCH: 5
Loss=0.6791541576385498 Batch_id=390 Accuracy=73.83: 100%|██████████| 391/391 [00:18<00:00, 21.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7365, Accuracy: 7421/10000 (74.21%)

EPOCH: 6
Loss=0.6915105581283569 Batch_id=390 Accuracy=75.20: 100%|██████████| 391/391 [00:17<00:00, 21.77it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.7027, Accuracy: 7554/10000 (75.54%)

EPOCH: 7
Loss=0.7693012952804565 Batch_id=390 Accuracy=76.54: 100%|██████████| 391/391 [00:18<00:00, 21.71it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6805, Accuracy: 7663/10000 (76.63%)

EPOCH: 8
Loss=0.6374693512916565 Batch_id=390 Accuracy=77.81: 100%|██████████| 391/391 [00:18<00:00, 21.32it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6351, Accuracy: 7804/10000 (78.04%)

EPOCH: 9
Loss=0.5569256544113159 Batch_id=390 Accuracy=78.57: 100%|██████████| 391/391 [00:18<00:00, 21.17it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6313, Accuracy: 7835/10000 (78.35%)

EPOCH: 10
Loss=0.5136169791221619 Batch_id=390 Accuracy=79.45: 100%|██████████| 391/391 [00:18<00:00, 21.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6284, Accuracy: 7834/10000 (78.34%)

EPOCH: 11
Loss=0.6290264129638672 Batch_id=390 Accuracy=80.17: 100%|██████████| 391/391 [00:18<00:00, 21.33it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6244, Accuracy: 7880/10000 (78.80%)

EPOCH: 12
Loss=0.47280532121658325 Batch_id=390 Accuracy=80.48: 100%|██████████| 391/391 [00:18<00:00, 21.38it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6389, Accuracy: 7876/10000 (78.76%)

EPOCH: 13
Loss=0.46835580468177795 Batch_id=390 Accuracy=81.01: 100%|██████████| 391/391 [00:18<00:00, 21.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5953, Accuracy: 7930/10000 (79.30%)

EPOCH: 14
Loss=0.5992981791496277 Batch_id=390 Accuracy=81.59: 100%|██████████| 391/391 [00:18<00:00, 21.00it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5671, Accuracy: 8037/10000 (80.37%)

EPOCH: 15
Loss=0.5516589879989624 Batch_id=390 Accuracy=81.87: 100%|██████████| 391/391 [00:18<00:00, 20.97it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5725, Accuracy: 8029/10000 (80.29%)

EPOCH: 16
Loss=0.6272042393684387 Batch_id=390 Accuracy=82.30: 100%|██████████| 391/391 [00:18<00:00, 21.24it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5465, Accuracy: 8137/10000 (81.37%)

EPOCH: 17
Loss=0.4658827781677246 Batch_id=390 Accuracy=82.71: 100%|██████████| 391/391 [00:18<00:00, 21.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5458, Accuracy: 8162/10000 (81.62%)

EPOCH: 18
Loss=0.4635300040245056 Batch_id=390 Accuracy=83.04: 100%|██████████| 391/391 [00:18<00:00, 21.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5425, Accuracy: 8161/10000 (81.61%)

EPOCH: 19
Loss=0.4257860779762268 Batch_id=390 Accuracy=83.27: 100%|██████████| 391/391 [00:18<00:00, 20.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.5361, Accuracy: 8209/10000 (82.09%)

EPOCH: 20
Loss=0.5714861154556274 Batch_id=390 Accuracy=83.63: 100%|██████████| 391/391 [00:18<00:00, 21.15it/s]
Test set: Average loss: 0.5317, Accuracy: 8196/10000 (81.96%)
```
