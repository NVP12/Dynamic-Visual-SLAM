# Visual SLAM in Dynamic Environments

<p align="center">
    <img src="assets/Raphson_Static.png" alt="Project Demo" width="700"/>
</p>

![SLAM](https://img.shields.io/badge/SLAM-Dynamic%20Mapping-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Mask%20R--CNN%20%7C%20YOLO-red)
![Optimization](https://img.shields.io/badge/Optimization-Bundle%20Adjustment-green)

## 📌 Overview

SLAM (Simultaneous Localization and Mapping) allows robots to create 2D/3D maps of environments while determining their location within these maps. While traditional SLAM techniques assume a static environment, this project aims to overcome that limitation by implementing a dynamic SLAM pipeline capable of excluding dynamic objects (such as people) from the map using semantic segmentation.

The base framework used in this project is ORB-SLAM3, which has been extended by integrating the SegFormer semantic segmentation model from Nvidia. The result is an enhanced Visual SLAM algorithm that can filter out dynamic objects, ensuring better localization and more accurate mapping in dynamic environments.

(**LOOK UP THE PRESENTATION LINK AT THE BOTTOM FOR A MORE INTUITIVE UNDERSTANDING**)

> 🧠 Built during my graduate coursework at the University of Minnesota (CSCI 5561 - Computer Vision).  
> 🔍 Core modules include: ORB feature tracking, dynamic object detection using SegFormer, visual odometry, and bundle adjustment.

For the owner: https://docs.google.com/document/d/1d98b0ul8DAbsA6pQdtcuoIs5RCLXsdhCS047mxy8j9w/edit?tab=t.0

---

## 🎯 Objectives

- Enable robust localization in non-static scenes
- Filter dynamic features that degrade map quality
- Combine classical SLAM with deep learning-based perception
- Validate with public datasets and real-world video input

---

## 🚀 Features

- ORB-based keypoint extraction and matching
- Real-time dynamic SLAM
- Integration with SegFormer for dynamic object detection
- Improved accuracy in indoor environments
- Evaluated using the TUM RGB-D dataset

---
## Presentation
https://docs.google.com/presentation/d/1OyZ4Ogx_ZLECr_UY7OXu8BlClwnN3eTwHzgSxzgS4i4/edit?usp=drive_link
