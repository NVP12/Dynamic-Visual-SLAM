#!/usr/bin/env python3
import sys # System specific modules
import os # Operating specific functions
import glob
import time # Python timing module
import copy # For deepcopying arrays
import shutil # High level folder operation tool
from pathlib import Path # To find the "home" directory location
import argparse # To accept user arguments from commandline
import natsort # To ensure all images are chosen loaded in the correct order
import yaml # To manipulate YAML files for reading configuration files
import copy # For making deepcopies of openCV matrices, python lists, numpy arrays etc.
import numpy as np # Python Linear Algebra module
import cv2 # OpenCV

#* ROS2 imports
import ament_index_python.packages
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

# If you have more files in the submodules folder
# from .submodules.py_utils import fn1 # Import helper functions from files in your submodules folder

# Import a custom message interface
# from your_custom_msg_interface.msg import CustomMsg #* Note the camel caps convention

# Import ROS2 message templates
from sensor_msgs.msg import Image # http://wiki.ros.org/sensor_msgs
from std_msgs.msg import String, Float64 # ROS2 string message template
from cv_bridge import CvBridge, CvBridgeError # Library to convert image messages to numpy array
from geometry_msgs.msg import Pose, PoseArray



#imports for semantic segmentation
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from PIL import Image as PILImage
import requests



class SemanticSegNode(Node):
    def __init__(self, node_name = "semantic_node"):
        super().__init__(node_name)
        
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.eval()

        self.pub_exp_config_name = "/mono_py_driver/experiment_settings_to_common" 
        self.sub_exp_config_name = "/mono_py_driver/experiment_settings" 
        self.sub_exp_ack_name = "/mono_py_driver/exp_settings_ack"
        self.pub_exp_ack_name = "/mono_py_driver/exp_settings_ack_to_driver"
        self.sub_img_to_agent_name = "/mono_py_driver/img_msg"
        self.pub_img_to_agent_name = "/mono_py_driver/img_msg_to_common"
        self.sub_timestep_to_agent_name = "/mono_py_driver/timestep_msg"
        self.pub_timestep_to_agent_name = "/mono_py_driver/timestep_msg_to_common"
        
        
        self.publish_exp_config_ = self.create_publisher(String, self.pub_exp_config_name, 1)
        self.publish_exp_ack_ = self.create_publisher(String, self.pub_exp_ack_name, 1)
        self.publish_img_msg_ = self.create_publisher(Image, self.pub_img_to_agent_name, 1)
        self.publish_timestep_msg_ = self.create_publisher(Float64, self.pub_timestep_to_agent_name, 1)
        self.publish_bad_points_ = self.create_publisher(PoseArray, "/bad_points", 1)
        
        self.subscribe_exp_config_ = self.create_subscription(String, self.sub_exp_config_name, self.exp_config_callback ,10)
        self.subscribe_exp_ack_ = self.create_subscription(String, self.sub_exp_ack_name, self.exp_ack_callback ,10)
        self.subscribe_img_msg_ = self.create_subscription(Image, self.sub_img_to_agent_name, self.img_callback ,10)
        self.subscribe_timestep_msg_ = self.create_subscription(Float64, self.sub_timestep_to_agent_name, self.timestep_callback ,10)
              
        
        
    def exp_config_callback(self, msg):
        exp_config = String()
        exp_config.data = msg.data
        self.publish_exp_config_.publish(exp_config)
    
    
    
    
    def exp_ack_callback(self, msg):
        ack_msg = String()
        ack_msg.data = msg.data
        self.publish_exp_ack_.publish(ack_msg)
        
    
    
    
    def img_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            bridge = CvBridge()
            gray_image = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Convert OpenCV image (BGR) to PIL image (RGB)
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Preprocess for SegFormer
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted = logits.argmax(dim=1)

            # Get segmentation mask and resize to original image shape
            mask = predicted[0].cpu().numpy()
            resized_mask = cv2.resize(mask.astype(np.uint8), (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create a binary mask where "person" class (12) is 1
            person_mask = (resized_mask == 12).astype(np.uint8)

            # Dilate the person mask to include neighboring pixels
            kernel = np.ones((10, 10), np.uint8)  # (5,5) dilation; adjust as needed
            dilated_mask = cv2.dilate(person_mask, kernel, iterations=1)

            # Find all (y, x) locations where dilated mask == 1
            mask_indices = np.argwhere(dilated_mask == 1)

            # Create a PoseArray message
            pose_array_msg = PoseArray()
            pose_array_msg.header = msg.header
        
            for y, x in mask_indices:  # np.argwhere returns (row, col) = (y, x)
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = 0.0  # Z unused
                pose_array_msg.poses.append(pose)

            # Publish the PoseArray (bad points)
            self.publish_bad_points_.publish(pose_array_msg)

            # Also publish the processed image
            processed_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publish_img_msg_.publish(processed_msg)

            self.get_logger().info(f"Published {len(pose_array_msg.poses)} bad points (including neighborhood).")

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")



    
    
    def timestep_callback(self, msg):
        timestep_msg = Float64()
        timestep_msg.data = msg.data
        self.publish_timestep_msg_.publish(timestep_msg)
        
        
        
        
        
        
        
        
        
        
def main(args = None):
    rclpy.init(args=args) # Initialize node
    node = SemanticSegNode("semantic_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
            
        
if __name__ == "__main__":
    main()
