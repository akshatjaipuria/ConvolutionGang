### Tasks
1. Migrating augmentation from torch vision to albumentation
2. Adding the implementation of GradCam in the module

### Training logs
```
EPOCH: 1
Loss=1.2432620525360107 Batch_id=390 Accuracy=46.20: 100%|██████████| 391/391 [00:36<00:00, 10.64it/s]
Loss=1.1304576396942139 Batch_id=1 Accuracy=61.72:   1%|          | 2/391 [00:00<00:37, 10.45it/s]
Test set: Average loss: 1.0838, Accuracy: 6105/10000 (61.05%)

EPOCH: 2
Loss=0.9001601934432983 Batch_id=390 Accuracy=64.75: 100%|██████████| 391/391 [00:36<00:00, 10.69it/s]
Loss=0.8501405715942383 Batch_id=1 Accuracy=70.70:   1%|          | 2/391 [00:00<00:36, 10.68it/s]
Test set: Average loss: 0.8771, Accuracy: 6849/10000 (68.49%)

EPOCH: 3
Loss=0.7098191380500793 Batch_id=390 Accuracy=72.21: 100%|██████████| 391/391 [00:36<00:00, 10.66it/s]
Loss=0.760125994682312 Batch_id=1 Accuracy=71.88:   1%|          | 2/391 [00:00<00:36, 10.77it/s]
Test set: Average loss: 0.7192, Accuracy: 7494/10000 (74.94%)

EPOCH: 4
Loss=0.5880144238471985 Batch_id=390 Accuracy=76.66: 100%|██████████| 391/391 [00:36<00:00, 10.70it/s]
Loss=0.7331576943397522 Batch_id=1 Accuracy=76.95:   0%|          | 1/391 [00:00<00:39,  9.78it/s]
Test set: Average loss: 0.6482, Accuracy: 7713/10000 (77.13%)

EPOCH: 5
Loss=0.5758970379829407 Batch_id=390 Accuracy=79.47: 100%|██████████| 391/391 [00:36<00:00, 10.65it/s]
Loss=0.5207703709602356 Batch_id=1 Accuracy=83.20:   1%|          | 2/391 [00:00<00:36, 10.69it/s]
Test set: Average loss: 0.6004, Accuracy: 7961/10000 (79.61%)

EPOCH: 6
Loss=0.3757728636264801 Batch_id=390 Accuracy=84.42: 100%|██████████| 391/391 [00:36<00:00, 10.62it/s]
Loss=0.4072847068309784 Batch_id=1 Accuracy=83.98:   1%|          | 2/391 [00:00<00:36, 10.62it/s]
Test set: Average loss: 0.4750, Accuracy: 8391/10000 (83.91%)

EPOCH: 7
Loss=0.3902896046638489 Batch_id=390 Accuracy=85.80: 100%|██████████| 391/391 [00:36<00:00, 10.61it/s]
Loss=0.39393842220306396 Batch_id=1 Accuracy=86.72:   1%|          | 2/391 [00:00<00:36, 10.60it/s]
Test set: Average loss: 0.4531, Accuracy: 8514/10000 (85.14%)

EPOCH: 8
Loss=0.32198503613471985 Batch_id=390 Accuracy=87.43: 100%|██████████| 391/391 [00:36<00:00, 10.64it/s]
Loss=0.24291110038757324 Batch_id=1 Accuracy=90.62:   1%|          | 2/391 [00:00<00:36, 10.76it/s]
Test set: Average loss: 0.5520, Accuracy: 8253/10000 (82.53%)

EPOCH: 9
Loss=0.2691170573234558 Batch_id=390 Accuracy=88.10: 100%|██████████| 391/391 [00:36<00:00, 10.64it/s]
Loss=0.41782182455062866 Batch_id=1 Accuracy=87.11:   1%|          | 2/391 [00:00<00:36, 10.67it/s]
Test set: Average loss: 0.7178, Accuracy: 7830/10000 (78.30%)

EPOCH: 10
Loss=0.33136916160583496 Batch_id=390 Accuracy=88.97: 100%|██████████| 391/391 [00:36<00:00, 10.66it/s]
Loss=0.2539178133010864 Batch_id=0 Accuracy=89.84:   0%|          | 1/391 [00:00<00:39,  9.86it/s]
Test set: Average loss: 0.4698, Accuracy: 8522/10000 (85.22%)

EPOCH: 11
Loss=0.22787928581237793 Batch_id=390 Accuracy=91.91: 100%|██████████| 391/391 [00:36<00:00, 10.68it/s]
Loss=0.2545255124568939 Batch_id=1 Accuracy=92.19:   1%|          | 2/391 [00:00<00:36, 10.76it/s]
Test set: Average loss: 0.4111, Accuracy: 8690/10000 (86.90%)

EPOCH: 12
Loss=0.19037838280200958 Batch_id=390 Accuracy=92.73: 100%|██████████| 391/391 [00:36<00:00, 10.72it/s]
Loss=0.25193023681640625 Batch_id=1 Accuracy=94.14:   1%|          | 2/391 [00:00<00:36, 10.69it/s]
Test set: Average loss: 0.4359, Accuracy: 8630/10000 (86.30%)

EPOCH: 13
Loss=0.22276481986045837 Batch_id=390 Accuracy=93.40: 100%|██████████| 391/391 [00:36<00:00, 10.74it/s]
Loss=0.10490256547927856 Batch_id=1 Accuracy=94.92:   1%|          | 2/391 [00:00<00:36, 10.56it/s]
Test set: Average loss: 0.4663, Accuracy: 8599/10000 (85.99%)

EPOCH: 14
Loss=0.21892663836479187 Batch_id=390 Accuracy=93.70: 100%|██████████| 391/391 [00:36<00:00, 10.68it/s]
Loss=0.1872207373380661 Batch_id=1 Accuracy=94.14:   1%|          | 2/391 [00:00<00:35, 10.81it/s]
Test set: Average loss: 0.4435, Accuracy: 8699/10000 (86.99%)

EPOCH: 15
Loss=0.1914360225200653 Batch_id=390 Accuracy=94.16: 100%|██████████| 391/391 [00:36<00:00, 10.70it/s]
Loss=0.1388688087463379 Batch_id=1 Accuracy=94.14:   1%|          | 2/391 [00:00<00:37, 10.43it/s]
Test set: Average loss: 0.4400, Accuracy: 8680/10000 (86.80%)

EPOCH: 16
Loss=0.14351984858512878 Batch_id=390 Accuracy=95.36: 100%|██████████| 391/391 [00:36<00:00, 10.69it/s]
Loss=0.08870675414800644 Batch_id=0 Accuracy=96.88:   0%|          | 1/391 [00:00<00:41,  9.49it/s]
Test set: Average loss: 0.4094, Accuracy: 8807/10000 (88.07%)

EPOCH: 17
Loss=0.13826881349086761 Batch_id=390 Accuracy=95.81: 100%|██████████| 391/391 [00:36<00:00, 10.70it/s]
Loss=0.10200934112071991 Batch_id=1 Accuracy=96.09:   1%|          | 2/391 [00:00<00:37, 10.25it/s]
Test set: Average loss: 0.4232, Accuracy: 8772/10000 (87.72%)

EPOCH: 18
Loss=0.21869663894176483 Batch_id=390 Accuracy=96.11: 100%|██████████| 391/391 [00:36<00:00, 10.71it/s]
Loss=0.12849020957946777 Batch_id=1 Accuracy=95.31:   1%|          | 2/391 [00:00<00:36, 10.54it/s]
Test set: Average loss: 0.4147, Accuracy: 8812/10000 (88.12%)

EPOCH: 19
Loss=0.14340369403362274 Batch_id=390 Accuracy=96.08: 100%|██████████| 391/391 [00:36<00:00, 10.68it/s]
Loss=0.06204645335674286 Batch_id=1 Accuracy=98.44:   1%|          | 2/391 [00:00<00:36, 10.69it/s]
Test set: Average loss: 0.4376, Accuracy: 8769/10000 (87.69%)

EPOCH: 20
Loss=0.2549378275871277 Batch_id=390 Accuracy=96.33: 100%|██████████| 391/391 [00:36<00:00, 10.69it/s]
Loss=0.10859446227550507 Batch_id=1 Accuracy=96.09:   1%|          | 2/391 [00:00<00:36, 10.58it/s]
Test set: Average loss: 0.4441, Accuracy: 8757/10000 (87.57%)

EPOCH: 21
Loss=0.059770144522190094 Batch_id=390 Accuracy=96.81: 100%|██████████| 391/391 [00:36<00:00, 10.71it/s]
Loss=0.08618944138288498 Batch_id=1 Accuracy=96.48:   1%|          | 2/391 [00:00<00:36, 10.71it/s]
Test set: Average loss: 0.4112, Accuracy: 8837/10000 (88.37%)

EPOCH: 22
Loss=0.10622861236333847 Batch_id=390 Accuracy=97.03: 100%|██████████| 391/391 [00:36<00:00, 10.72it/s]
Loss=0.08638603985309601 Batch_id=1 Accuracy=96.88:   1%|          | 2/391 [00:00<00:36, 10.54it/s]
Test set: Average loss: 0.4095, Accuracy: 8829/10000 (88.29%)

EPOCH: 23
Loss=0.03786652907729149 Batch_id=390 Accuracy=97.14: 100%|██████████| 391/391 [00:36<00:00, 10.70it/s]
Loss=0.0507931113243103 Batch_id=1 Accuracy=98.05:   1%|          | 2/391 [00:00<00:37, 10.48it/s]
Test set: Average loss: 0.4172, Accuracy: 8824/10000 (88.24%)

EPOCH: 24
Loss=0.11318135261535645 Batch_id=390 Accuracy=97.22: 100%|██████████| 391/391 [00:36<00:00, 10.63it/s]
Loss=0.05261882022023201 Batch_id=1 Accuracy=98.44:   1%|          | 2/391 [00:00<00:36, 10.75it/s]
Test set: Average loss: 0.4153, Accuracy: 8830/10000 (88.30%)

EPOCH: 25
Loss=0.06124243885278702 Batch_id=390 Accuracy=97.38: 100%|██████████| 391/391 [00:36<00:00, 10.66it/s]
Test set: Average loss: 0.4184, Accuracy: 8827/10000 (88.27%)
```
### Training History Plot

![Plot](https://github.com/akshatjaipuria/EVA/blob/master/P1S9/files/logs.png)

### GradCAM Results

![GradCam](https://github.com/akshatjaipuria/EVA/blob/master/P1S9/files/grad_cam_3.png)
![GradCam](https://github.com/akshatjaipuria/EVA/blob/master/P1S9/files/grad_cam_4.png)
