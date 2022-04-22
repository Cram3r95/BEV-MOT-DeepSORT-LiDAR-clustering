NOTES: This repository is uncomplete and messy. I will study again Multi-Object Tracking from September to December, cannot improve neither this repository nor SmartMOT now. If you want to try, please, first take a look of the DeepSORT algorithm and late fusion concepts. 

# BEV-MOT-DeepSORT-LiDAR-clustering

This is repository of my MSc final project titled "Predictive Techniques for Scene Understanding by using Deep Learning".

The present work proposes an accurate and real-time Deep Learning based Multi-Object Tracking architecture in the context of self-driving applications. 
A sensor fusion is performed merging 2D Visual Object Tracking (based on the CenterNet and Deep SORT algorithms) using a ZED camera, and 3D proposals using 
a LiDAR point cloud over the ROS framework and Docker containers.

<img src="images/Architecture.PNG" width="900" height="400" />

A comparison between the traditional Precision-Tracking strategy, Deep Learning based Visual Object Tracking and sensor fusion approach with LiDAR 
is carried out comparing the obtained pose estimations for each of them. Moreover, the proposals have been validated on the KITTI benchmark dataset for 
vehicle tracking, on the CARLA simulator for pedestrian tracking and on the Campus of the University of Alcalá using our autonomous vehicle 
developed in the SmartElderlyCar project.
