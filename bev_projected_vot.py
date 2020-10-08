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

# YOLO imports

from yolov3_ros.msg import yolo_list
from yolov3_ros.msg import yolo_obstacle

from PIL import Image
from argparse import ArgumentParser


# ROS imports

import rospy
import sensor_msgs.msg

# CenterNet and DeepSort imports

CENTERNET_PATH = '/root/ros_ws/src/centerNet-deep-sort/CenterNet/src/lib'
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

# input_type

opt.input_type = 'ros' # for webcam, 'webcam', for ip camera, 'ipcam', for video, 'vid', for ROS topic, 'ros'

#------------------------------
# for video
# opt.vid_path = 'MOT16-11.mp4'  #
opt.vid_path = '/home/robesafe/compartido_con_docker/Nuevos_Ficheros_CGH/Videos/Overtaking_frontal_1032.mp4'
#------------------------------
# for webcam  (webcam device index is required)
opt.webcam_ind = 0
#------------------------------
# for ipcamera (camera url is required.this is dahua url format)
opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
#------------------------------

with open('/root/ros_ws/src/centerNet-deep-sort/yolov3-model/coco.names','r') as coco_names:
    classes = coco_names.read().splitlines()
    # If we use just coco_names.readlines(), the output would have \n

def bbox_to_xywh_cls_conf(bbox): # In bbox it can be found all objects (car, persons, bycicles, etc.) procesed by YOLOv3
    # Now, we can add the corresponding objects to be tracked. Note that you have to check the id of the required object to be tracked in coco.names file
  
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
        
class Detector(object):
    def __init__(self, opt):
        
        
        self.vdo = cv2.VideoCapture()
        
        # CenterNet detector
        
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        
        self.write_video = True
        
    def open(self, video_path):
    
        # Webcam 
        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)
        
        # Ip camera
        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()
                
            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))
        
        # Video    
        elif opt.input_type == 'vid':
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)
        
        # ROS topic    
        else:
            print(" ")
        
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0,0,self.im_width,self.im_height
        
        if (self.write_video):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("recorded_tracking.avi", fourcc, 20, (self.im_width, self.im_height))
        # return self.vdo.isOpened()
        
    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        
        aux_score = []
        
        while self.vdo.grab():
        
            frame_no += 1
            
            start = time.time()
            
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax]
            # im = ori_im[ymin:ymax, xmin:xmax, :]
            
            results = self.detector.run(im)['results'] 
            bbox_xywh, cls_conf, type_object = bbox_to_xywh_cls_conf(results)
            
            aux_score = cls_conf
            # bbox_xywh represents the upper left corner and bottom right corner u,v coordinates in camera frame
            # cls_conf represents the score confidence for each detected object
            
            #print(cls_conf)
            
            if bbox_xywh is not None:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                
                if len(outputs) > 0: # At least an object was detected
                    bbox_xyxy = outputs[:,:4] # :4 means the first four columns (0 to 3 column)
                    identities = outputs[:,-1] # Objects identifiers (1, 2, ...). -1 means the last column

                    ori_im, colours = draw_bboxes(aux_score, type_object, classes, ori_im, bbox_xyxy, identities, offset=(xmin,ymin))
                    
            end = time.time()
            
            fps = 1/(end-start)
            
            avg_fps += fps 
            
            print("CenterNet time: {}s, fps: {}, avg fps: {}".format(round(end-start,3), round(fps,3), round(avg_fps/frame_no,3)))
            
            # Tracking and Detection visualization
            
            cv2.imshow("YOLOv3 + DeepSORT + CenterNet", ori_im)
            cv2.waitKey(1)
            
            if (self.write_video):
                self.output.write(ori_im)
        
class image_feature:
    def __init__(self, args,opt):
        
        # Detector and Tracker
        
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        
        # ROS publishers
        
        self.image_pub = rospy.Publisher('/perception/image_visual_mot', sensor_msgs.msg.Image, queue_size = 1)
        self.yolo_list_pub = rospy.Publisher('/perception/list_visual_mot', yolo_list, queue_size=20)
        
        # ROS subscriber
        
        self.topic = args.topic
        self.subscriber = rospy.Subscriber(self.topic, sensor_msgs.msg.Image, self.callback, queue_size = 5)
        
        # Detection file 
        
        self.filename = 'yolov3_detection.txt'
        self.path = '/root/ros_ws/src/centerNet-deep-sort/' + self.filename
        
        self.write_video = True
        
        self.start = float(0)
        self.end = float(0)
        self.end_aux = float(0)
        
        # Left eye (ZED camera - Amplied Projection Matrix (3 x 4 -> 4 x 4, adding 0,0,0,1)
        
        P = np.matrix([[672.8480834960938,0.0,664.2965087890625,0.0],
                       [0.0,672.8480834960938,347.0620422363281,0.0],
                       [0.0,0.0,1.0,0.0],
                       [0.0,0.0,0.0,1.0]])
    
        self.inv_proj_matrix = np.linalg.inv(P)
        
        self.avg_fps = float(0)
        self.frame_no = 0
        
    def callback(self, rosmsg):
        
        self.frame_no += 1
        
        aux_score = []

        self.subscriber.unregister()
        ros_image = rosmsg
        
        # Image from camera topic
        
        mode = 'RGBA'
        
        if self.topic == "/zed//zed_node/left/image_rect_color":
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
        # image = Resize((416,416), Image.BILINEAR)(image)
        #cv2.imwrite("./prueba.png", ori_im)
        
        self.start = time.time()
        
        im = ori_im[ymin:ymax, xmin:xmax, :]
        
        results = self.detector.run(im)['results']
        bbox_xywh, cls_conf, type_object = bbox_to_xywh_cls_conf(results)
        
        aux_score = cls_conf
        
        self.tracked_obstacle_list = yolo_list()
        
        if bbox_xywh is not None: # At least one object was detected
            self.outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            
            if len(self.outputs) > 0: # At least an object was tracked
                bbox_xyxy = self.outputs[:,:4]
                identities = self.outputs[:,-1]
                
                ori_im, colours = draw_bboxes(aux_score, type_object, classes, ori_im, bbox_xyxy, identities, offset=(xmin,ymin))
                
                r,c = bbox_xyxy.shape
               
                #self.end_aux = time.time()
 
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
    
                            image_feature.Image_to_RealWorld(self, color, score, object_id, obj_coordinates, object_type, ori_im)
                    except:
                        print(" ")
                 
                im_from_array = Image.fromarray(ori_im) #ToPILImage()(image)
        
                yolov3_tracking_output_image = sensor_msgs.msg.Image()
                yolov3_tracking_output_image.header.frame_id = rosmsg.header.frame_id
                yolov3_tracking_output_image.header.stamp = rosmsg.header.stamp
                yolov3_tracking_output_image.encoding = 'bgr8' 
                (yolov3_tracking_output_image.width, yolov3_tracking_output_image.height) = im_from_array.size
                yolov3_tracking_output_image.step = 3 * yolov3_tracking_output_image.width
                yolov3_tracking_output_image.data = im_from_array.tobytes()
            
                self.image_pub.publish(yolov3_tracking_output_image)
            
            else: # None object was tracked
                tracked_obstacle = yolo_obstacle()
                
                tracked_obstacle.type = "nothing"
                
                self.tracked_obstacle_list.yolo_list.append(tracked_obstacle)
                self.tracked_obstacle_list.header.stamp = rospy.Time.now()
                
                self.image_pub.publish(rosmsg)
                
        else: # None object was detected
            tracked_obstacle = yolo_obstacle()
            
            tracked_obstacle.type = "nothing"
            
            self.tracked_obstacle_list.yolo_list.append(tracked_obstacle)
            self.tracked_obstacle_list.header.stamp = rospy.Time.now()
            
            self.image_pub.publish(rosmsg)
            
        self.yolo_list_pub.publish(self.tracked_obstacle_list)     
        
        self.end = time.time()
        
        fps = 1/(self.end-self.start)
        #fps_aux = 1/(self.end_aux-self.start)
        
        self.avg_fps += fps 
        #avg_fps_aux += fps_aux
        
        print("CenterNet time: {}s, fps: {}, avg fps: {}".format(round(self.end-self.start,3), round(fps,3), round(self.avg_fps/self.frame_no,3)))
        
        # Tracking and Detection visualization
        
        #cv2.imshow("YOLOv3 + DeepSORT + CenterNet", ori_im)
        #cv2.waitKey(1)
        
        #if (self.write_video):
        #    self.output.write(ori_im)

        self.subscriber = rospy.Subscriber(self.topic, sensor_msgs.msg.Image, self.callback, queue_size = 5)

    def Image_to_RealWorld(self, color, object_score, object_id, obj_coordinates, object_type, ori_im):
        
        # 2D to 3D projection
        
        tracked_obstacle = yolo_obstacle()
        
        camera_height = 1.64 # ZED camera in SmartElderlyCar (Real-World)

        tracked_obstacle.type = object_type
        
        # Bounding Box coordinates in camera frame
        
        tracked_obstacle.x1 = obj_coordinates[0]
        tracked_obstacle.y1 = obj_coordinates[1]
        tracked_obstacle.x2 = obj_coordinates[2]
        tracked_obstacle.y2 = obj_coordinates[3]
        
        # 3D box dimensions that ties the object
        
        tracked_obstacle.h = 2.0 
        tracked_obstacle.w = 1.0
        tracked_obstacle.l = 1.0
        
        # Object score
        
        tracked_obstacle.probability = object_score
        
        # Object ID
        
        tracked_obstacle.object_id = object_id
        
        # Object colors (since the colours in Yolo detection are in BGR, so they must be converted to RGB for ROS topic )
        
        tracked_obstacle.color.a = 1.0
        tracked_obstacle.color.r = color[0]/255 
        tracked_obstacle.color.g = color[1]/255
        tracked_obstacle.color.b = color[2]/255

        # Image world to Real world
        
        centroid_x = (tracked_obstacle.x1+tracked_obstacle.x2)/2

        pixels = np.matrix([[centroid_x], [tracked_obstacle.y2], [1], [1]])
      
        p_camera = self.inv_proj_matrix * pixels
        
        K = camera_height/p_camera[1]
        
        p_camera_meters = p_camera*K    
        
        # If the camera is not parallel to the floor
        # if (p_camera_meters[2] > 4.1):
        #    correction = 1 +((p_camera_meters[2]-4)*0.65/46)
        #    p_camera_meters[2] = p_camera_meters[2]/correction
        #    p_camera_meters[3] = correction
        
        # Publish coordinates in real world frame (LiDAR coordinates)
        
        # Note that LiDAR x-axis corresponds to ZED z-axis, y-axis to (-)x-axis and z-axis to (-)y-axis
        
        tracked_obstacle.tx = float(p_camera_meters[2]) 
        tracked_obstacle.ty = float(-p_camera_meters[0])
        tracked_obstacle.tz = float(-p_camera_meters[1])
        
        # Append single tracked obstacle to tracked obstacle list
        
        self.tracked_obstacle_list.yolo_list.append(tracked_obstacle)
        self.tracked_obstacle_list.header.stamp = rospy.Time.now()

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

    # Detection and Tracking on Video, IP_Camera or Webcam
        
    if (opt.input_type == 'webcam' or opt.input_type == 'ipcam' or opt.input_type == 'vid'):

        # Initialize visualization
        
        cv2.namedWindow("YOLOv3 + DeepSORT + CenterNet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv3 + DeepSORT + CenterNet", 800, 600)
        
        det = Detector(opt)
        det.open(opt.vid_path)
        det.detect()

    # Detection and Tracking on ROS topic
    
    elif opt.input_type == 'ros':
        
        parser = ArgumentParser()
        parser.add_argument('--num-workers', type=int, default=4)
        parser.add_argument('--batch-size', type=int, default=1)
        parser.add_argument('--cpu', action='store_true')
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--topic', default="/zed/zed_node/left/image_rect_color")
        
        args = parser.parse_args()
        
        print("\nProcessing ROS topic:", args.topic," ")
        
        main(args)

    
    
    
    
    
    
    



