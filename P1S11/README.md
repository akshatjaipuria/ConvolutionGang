### Tasks
1. Write a code that draws cyclic lr curve.
2. Write code for a custom resnet architecture as follows:
```
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
```
3.Uses One Cycle Policy such that:  
  Total Epochs = 24  
  Max at Epoch = 5  
  LRMIN = FIND  
  LRMAX = FIND  
  NO Annihilation  

4.Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

### Cyclic LR curve
![graph](https://github.com/akshatjaipuria/EVA/blob/master/P1S11/files/cyclic_lr_plot.png)

### Training logs
```
  0%|          | 0/98 [00:00<?, ?it/s]EPOCH: 1
LR: [0.0009999999999999992]
Loss=1.3740323781967163 Batch_id=97 Accuracy=37.75: 100%|██████████| 98/98 [00:22<00:00,  4.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 1.3939, Accuracy: 5119/10000 (51.19%)

EPOCH: 2
LR: [0.001862825146503981]
Loss=1.0918467044830322 Batch_id=97 Accuracy=57.14: 100%|██████████| 98/98 [00:22<00:00,  4.43it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 1.0412, Accuracy: 6275/10000 (62.75%)

EPOCH: 3
LR: [0.004120426260042766]
Loss=0.8826957941055298 Batch_id=97 Accuracy=65.67: 100%|██████████| 98/98 [00:22<00:00,  4.40it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.9729, Accuracy: 6647/10000 (66.47%)

EPOCH: 4
LR: [0.006907063335488947]
Loss=0.8405880928039551 Batch_id=97 Accuracy=71.22: 100%|██████████| 98/98 [00:22<00:00,  4.40it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.9383, Accuracy: 6986/10000 (69.86%)

EPOCH: 5
LR: [0.009154122798231294]
Loss=0.7143046259880066 Batch_id=97 Accuracy=74.57: 100%|██████████| 98/98 [00:22<00:00,  4.40it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.6736, Accuracy: 7781/10000 (77.81%)

EPOCH: 6
LR: [0.009999993594945831]
Loss=0.5544305443763733 Batch_id=97 Accuracy=78.24: 100%|██████████| 98/98 [00:22<00:00,  4.38it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.6337, Accuracy: 7873/10000 (78.73%)

EPOCH: 7
LR: [0.009937369869797048]
Loss=0.4421062469482422 Batch_id=97 Accuracy=80.80: 100%|██████████| 98/98 [00:22<00:00,  4.32it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.6117, Accuracy: 7987/10000 (79.87%)

EPOCH: 8
LR: [0.009753706261960138]
Loss=0.4279489517211914 Batch_id=97 Accuracy=82.57: 100%|██████████| 98/98 [00:22<00:00,  4.30it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.5935, Accuracy: 8054/10000 (80.54%)

EPOCH: 9
LR: [0.009454012635881602]
Loss=0.44568124413490295 Batch_id=97 Accuracy=84.18: 100%|██████████| 98/98 [00:22<00:00,  4.27it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.5165, Accuracy: 8305/10000 (83.05%)

EPOCH: 10
LR: [0.009046463852437882]
Loss=0.41635704040527344 Batch_id=97 Accuracy=85.02: 100%|██████████| 98/98 [00:23<00:00,  4.25it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4943, Accuracy: 8343/10000 (83.43%)

EPOCH: 11
LR: [0.008542176780040937]
Loss=0.3595554232597351 Batch_id=97 Accuracy=86.64: 100%|██████████| 98/98 [00:23<00:00,  4.23it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4705, Accuracy: 8476/10000 (84.76%)

EPOCH: 12
LR: [0.00795490705544747]
Loss=0.3183010220527649 Batch_id=97 Accuracy=87.82: 100%|██████████| 98/98 [00:23<00:00,  4.24it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4044, Accuracy: 8641/10000 (86.41%)

EPOCH: 13
LR: [0.007300673865846474]
Loss=0.3250465989112854 Batch_id=97 Accuracy=89.07: 100%|██████████| 98/98 [00:23<00:00,  4.25it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4459, Accuracy: 8573/10000 (85.73%)

EPOCH: 14
LR: [0.006597322987191624]
Loss=0.3286028206348419 Batch_id=97 Accuracy=90.18: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3961, Accuracy: 8729/10000 (87.29%)

EPOCH: 15
LR: [0.005864039997953725]
Loss=0.3647056221961975 Batch_id=97 Accuracy=90.64: 100%|██████████| 98/98 [00:23<00:00,  4.25it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4025, Accuracy: 8641/10000 (86.41%)

EPOCH: 16
LR: [0.0051208269465530965]
Loss=0.2581229507923126 Batch_id=97 Accuracy=91.38: 100%|██████████| 98/98 [00:23<00:00,  4.23it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.4709, Accuracy: 8553/10000 (85.53%)

EPOCH: 17
LR: [0.00438795674762012]
Loss=0.1672387421131134 Batch_id=97 Accuracy=92.02: 100%|██████████| 98/98 [00:23<00:00,  4.23it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3607, Accuracy: 8846/10000 (88.46%)

EPOCH: 18
LR: [0.0036854201897316597]
Loss=0.21145375072956085 Batch_id=97 Accuracy=92.90: 100%|██████████| 98/98 [00:23<00:00,  4.23it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3452, Accuracy: 8871/10000 (88.71%)

EPOCH: 19
LR: [0.00303238063881079]
Loss=0.17497682571411133 Batch_id=97 Accuracy=93.45: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3579, Accuracy: 8860/10000 (88.60%)

EPOCH: 20
LR: [0.002446651311459579]
Loss=0.1486731767654419 Batch_id=97 Accuracy=94.01: 100%|██████████| 98/98 [00:23<00:00,  4.22it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3643, Accuracy: 8888/10000 (88.88%)

EPOCH: 21
LR: [0.0019442093768457615]
Loss=0.17371328175067902 Batch_id=97 Accuracy=94.51: 100%|██████████| 98/98 [00:23<00:00,  4.25it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3500, Accuracy: 8894/10000 (88.94%)

EPOCH: 22
LR: [0.0015387601411772367]
Loss=0.12637236714363098 Batch_id=97 Accuracy=95.22: 100%|██████████| 98/98 [00:23<00:00,  4.26it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3366, Accuracy: 8958/10000 (89.58%)

EPOCH: 23
LR: [0.001241363202675764]
Loss=0.15002043545246124 Batch_id=97 Accuracy=95.50: 100%|██████████| 98/98 [00:23<00:00,  4.24it/s]
  0%|          | 0/98 [00:00<?, ?it/s]
Test set: Average loss: 0.3308, Accuracy: 8968/10000 (89.68%)

EPOCH: 24
LR: [0.0010601307745677067]
Loss=0.0912422314286232 Batch_id=97 Accuracy=95.71: 100%|██████████| 98/98 [00:23<00:00,  4.24it/s]

Test set: Average loss: 0.3232, Accuracy: 9007/10000 (90.07%)
```
### Training History Plot

![Plot](https://github.com/akshatjaipuria/EVA/blob/master/P1S11/files/plot.png)
