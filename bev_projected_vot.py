#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:38:44 2020

@author: Carlos Gomez-Huelamo

Code to process a ROS image with CenterNet and Deep Sort techniques in order to perform Visual Object Tracking
and project it onto the Bird's Eye View space.

Communications are based on ROS (Robot Operating Sytem)

Inputs:  RGB(A) image topic (e.g. ZED camera topic: /zed/zed_node/left/image_rect_color)
Outputs: Tracked obstacles (e.g. /t4ac/perception/tracking/vot_projected_tracked_obstacles) topic 
         and monitors information (collision prediction, )

Note that each obstacle shows an unique ID in addition to its semantic information (person, car, ...), 
in order to make easier the decision-making process.

Executed via Python3.6 (python3.6 bev_projected_vot.py --arguments ...)
"""

# General imports

import numpy as np
import os
import cv2
import sys, time
from PIL import Image
from argparse import ArgumentParser

# ROS imports

import rospy
import sensor_msgs.msg

from t4ac_msgs.msg import BEV_tracker, BEV_trackers_list

# CenterNet and DeepSort imports

CENTERNET_PATH = './CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts
from deep_sort import DeepSort
from util import draw_bboxes

# Model and architecture (Best: resdcn_18) (other models: ctdet_coco_dla_2x.pth, arch = 'dla_34')

model_path = './CenterNet/models/ctdet_coco_resdcn18.pth'
arch = 'resdcn_18'

#model_path = './CenterNet/models/ctdet_coco_dla_2x.pth'
#arch = 'dla_34'

# Task: 'ctdet' (normal use, recommended) or 'multi_pose' for human pose estimation

task = 'ctdet'
opt = opts().init('{} --load_model {} --arch {}'.format(task, model_path, arch).split(' '))

# vis_thresh

opt.vis_thresh = 0.4

with open('./yolov3-model/coco.names','r') as coco_names:
    classes = coco_names.read().splitlines()

def bbox_to_xywh_cls_conf(bbox):
    """
    """
    type_object_list = [1,2,3] # person_id = 1, bicycle_id = 2, car_id = 3
    
    bbox_of_interest = []

    k = 0
    
    for i in type_object_list:
        bbox_object = bbox[i] 
        r,c = bbox_object.shape
        aux = np.zeros((r,1))
        aux.fill(i)
        bbox_object = np.concatenate([bbox_object,aux],1)
        
        if (k==0):
            bbox_of_interest = bbox_object
        else:
            bbox_of_interest = np.concatenate([bbox_of_interest,bbox_object])
        
        k = k+1
    
    bbox = bbox_of_interest
    
    if any(bbox[:,4] > opt.vis_thresh):
    
        bbox = bbox[bbox[:,4] > opt.vis_thresh, :]
        bbox[:,2] = bbox[:,2] - bbox[:,0]
        bbox[:,3] = bbox[:,3] - bbox[:,1]
        
        return bbox[:,:4], bbox[:,4], bbox[:,5]
        
    else:   
        return None, None, None
         
class image_feature:
    def __init__(self, args,opt):
        
        # Detector and Tracker
        
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        
        # ROS publishers
        
        self.pub_image_with_vot = rospy.Publisher('/t4ac/perception/image_visual_mot', sensor_msgs.msg.Image, queue_size = 1)
        self.pub_tracker_list = rospy.Publisher('/t4ac/perception/tracked_obstacles', BEV_trackers_list, queue_size=20)
        
        # ROS subscriber
        
        self.image_topic = args.image_topic
        self.image_subscriber = rospy.Subscriber(self.image_topic, sensor_msgs.msg.Image, self.callback, queue_size = 5)
        
        # Detection file 
        
        self.filename = 'tracked_obstacles.txt'
        self.path = pwd + '/results/' + self.filename
        
        self.write_video = True
        
        self.start = float(0)
        self.end = float(0)
        self.end_aux = float(0)
        
        # Left eye (ZED camera - Amplied Projection Matrix (3 x 4 -> 4 x 4, adding 0,0,0,1)
        
        self.camera_height = 1.64 # ZED camera in Techs4AgeCar (Real-World)
        P = np.matrix([[672.8480834960938,0.0,664.2965087890625,0.0],
                       [0.0,672.8480834960938,347.0620422363281,0.0],
                       [0.0,0.0,1.0,0.0],
                       [0.0,0.0,0.0,1.0]])
    
        self.inv_proj_matrix = np.linalg.inv(P)
        
        self.avg_fps = float(0)
        self.frame_no = 0

    def image_to_realworld(self, color, object_score, object_id, object_type, obj_coordinates):
        
        # 2D to Bird's Eye View (LiDAR frame, z-filtered) projection
        
        tracked_obstacle = BEV_tracker()
        tracked_obstacle.type = object_type
        tracked_obstacle.object_id = object_id
  
        # 3D box dimensions that ties the object 
        #TODO: Improve this 3D information recovery
        
        tracked_obstacle.w = 0 
        tracked_obstacle.l = 0
        tracked_obstacle.o = 0
   
        # Object colors (since the colours in Yolo detection are in BGR, so they must be converted to RGB for ROS topic )
        #TODO: Publish as a visualization marker when the 3D information (width, length, orientation) is correct
        #tracked_obstacle.color.a = 1.0
        #tracked_obstacle.color.r = color[0]/255 
        #tracked_obstacle.color.g = color[1]/255
        #tracked_obstacle.color.b = color[2]/255

        # Image world to Bird's Eye View (LiDAR frame, z-filtered) projection
        
        centroid_x = (obj_coordinates[0]+obj_coordinates[2])/2
        pixels = np.matrix([[centroid_x], [obj_coordinates[3]], [0], [1]]) # 4 x 1
        p_camera = np.dot(self.inv_proj_matrix,pixels) 
        K = self.camera_height/p_camera[1]
        p_camera_meters = np.dot(p_camera,K) # Camera frame  
        
        # If the camera is not parallel to the floor
        # if (p_camera_meters[2] > 4.1):
        #    correction = 1 +((p_camera_meters[2]-4)*0.65/46)
        #    p_camera_meters[2] = p_camera_meters[2]/correction
        #    p_camera_meters[3] = correction
        
        # Publish coordinates in BEV frame (LiDAR frame)
        
        # Note that LiDAR x-axis corresponds to ZED z-axis, y-axis to (-)x-axis and z-axis to (-)y-axis
        
        tracked_obstacle.x = float(p_camera_meters[2]) 
        tracked_obstacle.y = float(-p_camera_meters[0])
        
        # Append single tracked obstacle to tracked obstacle list
        
        self.tracked_obstacles_list.tracked_obstacle_list.append(tracked_obstacle)
        
    def callback(self, image_rosmsg):
        """
        """
        self.frame_no += 1
        
        aux_score = []

        self.image_subscriber.unregister()
        ros_image = image_rosmsg
        
        # Image from camera topic
        
        mode = 'RGBA'
        
        if self.topic == "/zed/zed_node/left/image_rect_color":
            mode = 'RGBA' # RGB
        
        image = Image.frombytes(mode, (ros_image.width, ros_image.height), ros_image.data)
        image = image.convert('RGB')
        
        self.image_width, self.image_height = image.size
        self.area = 0,0,self.image_width,self.image_height
   
        if (self.write_video):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("recorded_tracking.avi", fourcc, 20, (self.image_width, self.image_height))

        xmin, ymin, xmax, ymax = self.area
        
        ori_im = np.array(image)

        self.start = time.time()
        
        im = ori_im[ymin:ymax, xmin:xmax, :]
        
        results = self.detector.run(im)['results']
        bbox_xywh, cls_conf, type_object = bbox_to_xywh_cls_conf(results)
        
        aux_score = cls_conf
        
        self.tracked_obstacles_list = BEV_trackers_list()
        self.tracked_obstacles_list.header.stamp = image_rosmsg.header.stamp

        if bbox_xywh is not None: # At least one object was detected
            self.outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            
            if len(self.outputs) > 0: # At least an object was tracked
                bbox_xyxy = self.outputs[:,:4]
                identities = self.outputs[:,-1]
                
                ori_im, colours = draw_bboxes(aux_score, type_object, classes, ori_im, bbox_xyxy, identities, offset=(xmin,ymin))
                
                r,c = bbox_xyxy.shape
 
                for i in range(r):
                    obj_coordinates = bbox_xyxy[i]
                    
                    try:
                        kind = int(type_object[i]) if type_object is not None else 0 # Object type
                        
                        if (kind > 0):
                            object_type = classes[kind-1] 
                            
                            score = float(aux_score[i]) if aux_score is not None else 0 # Object score confidence
                            score = round(score, 2)

                            object_id = int(identities[i]) if identities is not None else 0 # Object ID 

                            color = colours[i]
    
                            image_feature.image_to_realworld(self, color, score, object_id, object_type, obj_coordinates)
                    except:
                        print(" ")
                 
                im_from_array = Image.fromarray(ori_im)
        
                output_ros_image = sensor_msgs.msg.Image()
                output_ros_image.header.frame_id = image_rosmsg.header.frame_id
                output_ros_image.header.stamp = image_rosmsg.header.stamp
                output_ros_image.encoding = 'bgr8' 
                output_ros_image.width, output_ros_image.height = im_from_array.size
                output_ros_image.step = 3 * output_ros_image.width
                output_ros_image.data = im_from_array.tobytes()
            
                self.image_pub.publish(output_ros_image)
            
            else: # None object was tracked
                tracked_obstacle = BEV_tracker()
                
                tracked_obstacle.type = "nothing"
                
                self.tracked_obstacles_list.header.stamp = rospy.Time.now()
                self.tracked_obstacles_list.tracked_obstacles_list.append(tracked_obstacle)
                
                self.pub_image_with_vot.publish(image_rosmsg)
                
        else: # None object was detected
            tracked_obstacle = BEV_tracker()
            
            tracked_obstacle.type = "nothing"
            
            self.tracked_obstacles_list.header.stamp = rospy.Time.now()
            self.tracked_obstacles_list.tracked_obstacles_list.append(tracked_obstacle)
            
            self.pub_image_with_vot.publish(image_rosmsg)

        self.pub_tracker_list.publish(self.tracked_obstacle_list)     
        
        self.end = time.time()
        
        fps = 1/(self.end-self.start)
        
        self.avg_fps += fps 
        
        print("CenterNet time: {}s, fps: {}, avg fps: {}".format(round(self.end-self.start,3), round(fps,3), round(self.avg_fps/self.frame_no,3)))
        
        # Tracking and Detection visualization
        
        #cv2.imshow("CenterNet + DeepSORT", ori_im)
        #cv2.waitKey(1)
        
        #if (self.write_video):
        #    self.output.write(ori_im)

        self.image_subscriber = rospy.Subscriber(self.image_topic, sensor_msgs.msg.Image, self.callback, queue_size = 5)

def main(args):
    print(args)
    
    image_feature(args,opt)
    rospy.init_node('tracking_node', anonymous=True)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        #print(" ")
        rospy.loginfo("Shutting down ROS Image feature detector module")

if __name__ == '__main__':

    # Detection and Tracking on ROS topic
      
    parser = ArgumentParser()
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--image_topic', default="/zed/zed_node/left/image_rect_color")
    
    args = parser.parse_args()
    
    print("\nProcessing ROS topic:", args.topic," ")
    
    main(args)

    
    
    
    
    
    
    



