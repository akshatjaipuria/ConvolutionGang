## Task 1 - Training ResNet18 on Tiny-ImageNet Dataset from scratch

The model reaches a maximum validation accuracy of **56.23%** on Tiny-ImageNet using **ResNet 18** model.

### Training History

![Plot](https://github.com/akshatjaipuria/EVA/blob/master/P1S12/files/logs.png)

## Part 2 - Finding YOLO v2 Anchor Boxes

Finding anchor boxes for a dataset of 50 dog images using **K-Means Clustering Algorithm**.

### Mean IoU vs Clusters

![kmeans_iou](files/iou_vs_clusters.png)

For the dataset created, it was found that the most suitable number of clusters could be 6.

| Number of Clusters (k) | Mean IoU |             Cluster Plot            |
| :--------------------: | :------: | :---------------------------------: |
|           4            |   0.81   | ![4_clusters](files/4_clusters.png) |
|           5            |   0.83   | ![5_clusters](files/5_clusters.png) |
|           6            |   0.85   | ![6_clusters](files/6_clusters.png) |
