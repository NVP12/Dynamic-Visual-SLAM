# Dynamic Visual SLAM in Dynamic Environments

![SLAM](https://img.shields.io/badge/SLAM-Dynamic%20Mapping-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Mask%20R--CNN%20%7C%20YOLO-red)
![Optimization](https://img.shields.io/badge/Optimization-Bundle%20Adjustment-green)

## 📌 Overview

This project implements a robust **Visual SLAM pipeline** capable of operating in **dynamic environments**, where moving objects (e.g., people, vehicles) can interfere with traditional feature-based localization and mapping. By detecting and filtering dynamic objects, the system maintains accurate localization and map consistency.
(*LOOK UP THE PRESENTATION LINK AT THE BOTTOM FOR A MORE INTUITIVE UNDERSTANDING*)

> 🧠 Built during my graduate coursework at the University of Minnesota (CSCI 5561 - Computer Vision).  
> 🔍 Core modules include: ORB feature tracking, dynamic object detection using YOLO/Mask R-CNN, visual odometry, and bundle adjustment.

---

## 🎯 Objectives

- Enable robust localization in non-static scenes
- Filter dynamic features that degrade map quality
- Combine classical SLAM with deep learning-based perception
- Validate with public datasets and real-world video input

---

## 🚀 Features

- ORB-based keypoint extraction and matching
- YOLOv5 / Mask R-CNN for dynamic object segmentation
- Visual odometry with frame-to-frame pose estimation
- Keyframe selection and graph-based optimization
- Bundle adjustment for global consistency
- Visualization with Pangolin (3D map, trajectory, keypoints)

---
## Presentation
https://docs.google.com/presentation/d/1OyZ4Ogx_ZLECr_UY7OXu8BlClwnN3eTwHzgSxzgS4i4/edit?usp=drive_link
