/***

Carlos Gómez Huélamo December 2019

3D tracking, precision tracking and monitors

SmartElderlyCar (SEC) - Tech4AgeCar (T4AC) project

Simulation

***/

// Includes //

// General purpose includes
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <string.h>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <vector>

// OpenCV includes
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include <cv_bridge/cv_bridge.h>

// ROS includes
#include <ros/ros.h>

#include <geodesy/utm.h>
#include <geodesy/wgs84.h>

#include <geographic_msgs/GeoPoint.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Transform.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Path.h"

#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/PointCloud2.h>

#include <std_msgs/ColorRGBA.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Time.h>

#include "tf/tf.h"
#include "tf/transform_listener.h"
#include <tf/transform_broadcaster.h>

#include <visualization_msgs/Marker.h>
#include <rviz_visual_tools/rviz_visual_tools.h>

// PCL includes
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/impl/point_types.hpp>

// Precision tracking includes
#include <precision_tracking/track_manager_color.h>
#include <precision_tracking/tracker.h>
#include <precision_tracking/high_res_timer.h>
#include <precision_tracking/sensor_specs.h>

// YOLO includes
#include "yolov3_centernet_ros/yolo_list.h"
#include "yolov3_centernet_ros/yolo_obstacle.h"

// CARLA includes
#include "carla_msgs/CarlaObjectLocation.h"
#include "carla_msgs/CarlaObjectLocationList.h"

// SEC (SmartElderlyCar) includes
#include <sec_msgs/Route.h>
#include <sec_msgs/Lanelet.h>
#include <sec_msgs/RegElem.h>
#include <sec_msgs/Distance.h>
#include <sec_msgs/CarControl.h>
#include <sec_msgs/ObstacleArray.h>
#include "map_manager_base.hpp"

// Precision tracking includes
#include <precision_tracking/track_manager_color.h>
#include <precision_tracking/tracker.h>
#include <precision_tracking/high_res_timer.h>
#include <precision_tracking/sensor_specs.h>

// End Includes //


// Defines //

#define VER_OBJETOS_AGRUPADOS 0
#define VER_CLUSTERES 0
#define VER_TRAYECTORIA 0 
#define KALMAN 0
#define VIEWER_3D 0
#define LaneletFilter 0
#define DEBUG 0

#define PI  3.1415926
#define THMIN     10.0
#define THMAX     70.0
#define SENSOR_HEIGHT 1.73 // LiDAR

#define CAR_LENGTH 3.7
#define CAR_WIDTH 1.7
#define CAR_HEIGHT 1.5

#define PEDESTRIAN_LENGTH 0.6
#define PEDESTRIAN_WIDTH 0.6
#define PEDESTRIAN_HEIGHT 1.85

#define MIN_MONITOR(x,y) (x < y ? x : y) // Returns x if x less than y. Otherwise, returns y
#define MAX_MONITOR(x,y) (x > y ? x : y) // Returns x if x greater than y. Otherwise, returns y
#define INSIDE 0
#define OUTSIDE 1

#define TIME_PRECISION_TRACKING 1.5

// End Defines //


// Structures //

typedef struct
{
	double x; // x UTM with respect to the map origin
	double y; // y UTM with respect to the map origin
}Area_Point;

typedef struct Kalman_Points_History
{
	// Store up to 10 measurements on each Kalman filter
	
	// Position of the obstacle
	double x[10];
	double y[10];
	double z[10];

	// Dimensions of the obstacle
	double w[10];
	double h[10];
	double d[10];

	// Prediction of the obstacle
	double predicted_x[10];
	double predicted_y[10];
	double predicted_z[10];
	double time[10];
	string type;
}kalman_points_history;

typedef struct Points_Kalman
{
	double x;
	double y;
	double z;
	double w;
	double h;
	double d;
	string type;
}points_kalman;

typedef struct Points
{
	int x;
	int z;
}points[100];

struct Object
{
	float centroid_x; // Local centroid (with respect to the "base_link" frame)
	float centroid_y;
	float centroid_z;
	float global_centroid_x; // Global centroid (with respect to the "map" frame)
	float global_centroid_y;
	float global_centroid_z;
	double r;
	double g;
	double b;
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
	float w;
	float h;
	float d;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	string type;
	int id;
	int pedestrian_state; // If the object is a pedestrian, store its state
}object;

struct Merged_Object
{
	int cluster[10];
	float centroid_x; // Local centroid (with respect to the "base_link" frame)
	float centroid_y;
	float centroid_z;
	float global_centroid_x; // Global centroid (with respect to the "map" frame)
	float global_centroid_y;
	float global_centroid_z;
	float w;
	float h;
	float d;
	double r;
	double g;
	double b;
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	string type;
	int id;
};

struct Cluster
{       
	int cluster[10];
	int repetitions;
	float centroid_x[10];
	float centroid_y[10];
	float centroid_z[10];
	double r;
	double g;
	double b;
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
	float w;
	float h;
	float d;
	int id;
}cluster;

struct Precision_Trackers
{
	int id;
	double time;
	Eigen::Vector3f centroid; // x, y, z position
	Eigen::Vector3f previous_centroid;
	Eigen::Vector3f previous_velocity;
	Eigen::Vector3f estimated_velocity; // x, y, z speed
	Eigen::Vector3f size; // Height, Width, Depth
	string type;
	int pedestrian_state;
}precision_trackers;

// End Structures //


// ROS communication // 

// ROS Publishers

ros::Publisher pub_LiDAR_Pointcloud_Coloured_XYZ_Filtered;
ros::Publisher pub_LiDAR_Pointcloud_Coloured_XYZ_Angle_Filtered;
ros::Publisher pub_LiDAR_Obstacles;
ros::Publisher pub_LiDAR_Obstacles_Velocity_Marker;
ros::Publisher pub_Detected_Pedestrian;
ros::Publisher pub_Safe_Merge;
ros::Publisher pub_Front_Car;
ros::Publisher pub_Front_Car_Distance;
ros::Publisher pub_Safe_Lane_Change;
ros::Publisher pub_Distance_Overtake;

// ROS Subscribers

ros::Subscriber monitorizedlanes_sub; // Monitorized lanelets
ros::Subscriber route_sub; // Route
ros::Subscriber waiting_sub; // Empty message waiting (STOP behaviour)

// End ROS communication //


// Global variables //

// Transform variables

tf::StampedTransform transformBaseLinkBaseCamera, transformOdomBaseLink, transformBaseLinkOdom, transformMaptoVelodyne, transformVelodynetoMap;				
tf::Transform tfBaseLinkBaseCamera, tfOdomBaseLink;
tf::TransformListener *listener;

// SEC variables

sec_msgs::Route pedestrian_crossing_lanelets; 
sec_msgs::Route merging_lanelets; // Merging role lanelets
sec_msgs::Route route_lanelets; // Monitorized lanelets that are in the planified route
sec_msgs::Route left_lanelets; // Left lanelets of the planified route
sec_msgs::Route right_lanelets; // Right lanelets of the planified route
sec_msgs::Route route_left_lanelets;
sec_msgs::Route route_right_lanelets;
sec_msgs::Route all_lefts;
sec_msgs::Route all_rights;
sec_msgs::Route route; // Received route
sec_msgs::Route monitorized_lanelets; // Important lanelet for each use case (STOP, give way, etc.)

sec_msgs::ObstacleArray Obstacles;
sec_msgs::Obstacle current_obstacle;

sec_msgs::RegElem current_regulatory_element;

// Monitors

bool merging_monitor;
bool pedestrian_crossing_monitor;
int stop = 0; // 0 = Inactive, 1 Active (Car cannot cross the STOP), 2 Merging monitor (Car can cross the stop if merging monitor allows)
std_msgs::Bool lane_change;
int id_lanelet_pedestrian_crossing = 0;
int id_lanelet_merging = 0;
int global_pedestrian_crossing_occupied;
int merging_occupied;

// Visualization variables

//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer")); // 3D viewer

// Namespaces //

using namespace std;

namespace rvt = rviz_visual_tools;

namespace rviz_visual_tools
	{
	class RvizVisualToolsDemo 
	{
		private:
		  rvt::RvizVisualToolsPtr visual_tools_;
		  string name_;
		public:
		  
		  // Constructor

		  RvizVisualToolsDemo() : name_("rviz_tracking")
		  {
			visual_tools_.reset(new rvt::RvizVisualTools("/map", "/rviz_visual_tools"));
			visual_tools_->loadMarkerPub();  // create publisher before waiting

			// Clear messages
			visual_tools_->deleteAllMarkers();
			visual_tools_->enableBatchPublishing();
		  }

		  void publishLabelHelper(const Eigen::Isometry3d& pose, const string& label)
	  	  {
			Eigen::Isometry3d pose_copy = pose;
			pose_copy.translation().x() -= 0.2;
			visual_tools_->publishText(pose_copy, label, rvt::WHITE, rvt::LARGE, false);
	  	  }

		  void Show_WireFrame(geometry_msgs::Point32 location, const string label)
		  {
			Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

			pose.translation() = Eigen::Vector3d(location.x, location.y, location.z);

			double depth = PEDESTRIAN_LENGTH, width = PEDESTRIAN_WIDTH, height = PEDESTRIAN_HEIGHT;

			visual_tools_->publishWireframeCuboid(pose, depth, width, height, rvt::RED);
			
			publishLabelHelper(pose, label);

			//visual_tools_->trigger();
		  }
	};
}

// End Namespaces //


// Geographic variables 

geodesy::UTMPoint odomUTMmsg;
nav_msgs::Odometry previous_odom;
geodesy::UTMPoint utm_origin;
double lat_origin, lon_origin;
shared_ptr<LaneletMap> loadedMap;
nav_msgs::OccupancyGrid localMap;
geographic_msgs::GeoPoint geo_origin;

// General purpose variables

cv::Mat imgProjection;
vector<std_msgs::ColorRGBA> colours;

vector<Object> only_laser_objects, merged_objects, output_objects;
//vector<Tracking_Points> tracking_points, tracking_points_prev, tracking_points_aux, tracking_points_lidar, tracking_points_prev_lidar, tracking_points_aux_lidar;
vector<Precision_Trackers> pTrackers;
vector<cv::KalmanFilter> kfs;
vector<int> kfsTime;
precision_tracking::Params params;
Eigen::Vector3f estimated_velocity;
vector<precision_tracking::Tracker> trackers;
int indexpTrackers = 0;
int flag_tracking_points = 0;
int number_of_clusters = 0; // Number of clusters after the first filter

Area_Point polygon_area[] = {0,0,
			     0,0,
			     0,0,
			     0,0};

int Number_of_sides = 4; // Of the area you have defined. Here is a rectangle, so area[] has four rows

// End Global variables


// Declarations of functions // 

// General use functions

geometry_msgs::Point32 Global_To_Local_Coordinates(geometry_msgs::PointStamped );
geometry_msgs::Point32 Local_To_Global_Coordinates(geometry_msgs::PointStamped );
float get_Centroids_Distance(pcl::PointXYZ , pcl::PointXYZ );
void Inside_Polygon(Area_Point *, int , Area_Point, bool &);
void Obstacle_in_Lanelet(pcl::PointCloud<pcl::PointXYZRGB>::Ptr , geometry_msgs::PointStamped , geometry_msgs::Point32 , geometry_msgs::PointStamped , geometry_msgs::PointStamped , geometry_msgs::PointStamped , geometry_msgs::PointStamped , ros::Time , sec_msgs::Lanelet );

// Kalman functions

cv::KalmanFilter initKalman(float, float, float, float, float, float, float, float, float, float, float, float, float );
void updateKalman(cv::KalmanFilter &, float , float , float , float , float , float , bool );
Points_Kalman getKalmanPrediction(cv::KalmanFilter );

// Point Cloud filters

pcl::PointCloud<pcl::PointXYZRGB> xyz_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr );
pcl::PointCloud<pcl::PointXYZRGB> angle_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr );
void cluster_filter (pcl::PointCloud<pcl::PointXYZRGB>::Ptr , float , int , int , vector<Object> *, int *);
void segmentation_filter (pcl::PointCloud<pcl::PointXYZRGB>::Ptr , float , float , float , float , float , float , float , int , int , vector<Object> *, int *, string);
vector<Merged_Object> merging_z(vector<Object> );

// Cluster functions

vector<Merged_Object> merging_z(vector<Object> );

// Calbacks

void route_cb(const sec_msgs::Route::ConstPtr& );
void waiting_cb(const std_msgs::Empty );
void regelement_cb(const sec_msgs::RegElem::ConstPtr& , const sec_msgs::Route::ConstPtr& );
//void sensor_fusion_and_monitors_cb(ESPECIFICAR);
void clustering_precision_tracking_monitors_cb(const sensor_msgs::PointCloud2::ConstPtr& , const nav_msgs::Odometry::ConstPtr& );

// End Declarations of functions //


// Main //

int main (int argc, char ** argv)
{
	// Initialize ROS

	ros::init(argc, argv, "sensor_fusion_and_monitors_node");
	ros::NodeHandle nh;

	// Map origin latitude and longitude by parameters

    	nh.param<double>("/lat_origin",lat_origin,40.5126566);
    	nh.param<double>("/lon_origin",lon_origin,-3.34460735);

	// Initialize origin of the map

	geographic_msgs::GeoPoint geo_origin;
	geo_origin.latitude = lat_origin;
	geo_origin.longitude = lon_origin;
	geo_origin.altitude = 0;
	utm_origin = geodesy::UTMPoint(geo_origin); // Convert to geodesy::UTMPoint

	// Transform listener

	listener = new tf::TransformListener(ros::Duration(5.0));

	// Publishers //

	pub_LiDAR_Pointcloud_Coloured_XYZ_Filtered = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_coloured_xyz_filtered", 1);
	pub_LiDAR_Pointcloud_Coloured_XYZ_Angle_Filtered = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_coloured_xyz_angle_filtered", 1);
	pub_LiDAR_Obstacles = nh.advertise<sec_msgs::ObstacleArray>("/obstacles", 1, true);
	pub_Detected_Pedestrian = nh.advertise<std_msgs::Bool>("/pedestrian",1);
	pub_Safe_Merge = nh.advertise<std_msgs::Bool>("/safeMerge",1);
	pub_Front_Car = nh.advertise<sec_msgs::Obstacle>("/frontCarCurrentLane", 1, true);
	pub_Front_Car_Distance = nh.advertise<std_msgs::Float64>("/frontCarCurrentLane_distance", 1, true);
	pub_Safe_Lane_Change = nh.advertise<std_msgs::Bool>("/safeLaneChange",1);
	pub_Distance_Overtake = nh.advertise<sec_msgs::Distance>("/distOvertake", 1);

	// Tracking publishers

	pub_LiDAR_Obstacles_Velocity_Marker = nh.advertise<visualization_msgs::Marker>("/Obstacle_marker_vel", 1, true); // Only LiDAR
	/*tracked_objects_vel_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_objects_marker_vel", 1, true); // Only vision
	tracked_objects_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_objects_marker", 1, true); // Only vision
	tracked_merged_objects_vel_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_merged_objects_marker_vel", 1, true); // Sensor fusion LiDAR and Camera
	tracked_merged_objects_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_merged_objects_marker", 1, true); // Sensor fusion LiDAR and Camera*/
	
	// End Publishers //

 	// Subscribers //

 	message_filters::Subscriber<sec_msgs::RegElem> regelem_sub_; // Regulatory elements of current monitorized lanelets
	message_filters::Subscriber<sec_msgs::Route> regelemLanelet_sub_; // Monitorized lanelets
	message_filters::Subscriber<sec_msgs::Distance> regelemDist_sub_; // Distance to regulatory elements
	message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_; // Coloured LiDAR point cloud
	message_filters::Subscriber<sensor_msgs::PointCloud2> velodyne_cloud_sub_; // LiDAR point cloud
	message_filters::Subscriber<nav_msgs::Odometry> odom_sub_; // Odometry
	message_filters::Subscriber<yolov3_centernet_ros::yolo_list> vision_sub_; // Detection and Tracking with camera (CenterNet + Deep Sort + YOLO)
        message_filters::Subscriber<carla_msgs::CarlaObjectLocationList> carla_sub_; 

	regelem_sub_.subscribe(nh, "/currentRegElem", 1);
	regelemLanelet_sub_.subscribe(nh, "/monitorizedLanelets", 1);
	cloud_sub_.subscribe(nh, "/velodyne_coloured", 1); // Colored point cloud (based on semantic segmentation)
	velodyne_cloud_sub_.subscribe(nh, "/velodyne_points", 1);
	odom_sub_.subscribe(nh, "/carla/ego_vehicle/odometry", 1); // Note that in CARLA, each dynamic object has its own odometry topic. /odom has all objects odometries
	waiting_sub = nh.subscribe<std_msgs::Empty>("/waitingAtStop", 1, &waiting_cb);
	route_sub = nh.subscribe<sec_msgs::Route>("/route", 1, &route_cb);
	//vision_sub_.subscribe(nh, "/yolov3_tracking_list", 1);
        //carla_sub_.subscribe(nh, "/carla/hero/location_list", 1); 

	//carla_sub_ = nh.subscribe<carla_msgs::CarlaObjectLocation>("/carla/hero/location", 1, &Procesar_carla_cb); 

        //yolo_sub_ = nh.subscribe<nav_msgs::Path>("/yolov3_tracking_list", 1, &Procesar_yolo_cb);

        // End Subscribers //

	// Callbacks //

	// Callback 1: Synchonize monitorized lanelets and current regulatory element (Exact time)

	typedef message_filters::sync_policies::ExactTime<sec_msgs::RegElem, sec_msgs::Route> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync_(MySyncPolicy(10), regelem_sub_, regelemLanelet_sub_);
	sync_.registerCallback(boost::bind(&regelement_cb, _1, _2));

	/*// Callback 2: Synchronize LiDAR point cloud and camera information (including detection and tracking) (Approximate time)

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, yolov3_centernet_ros::yolo_list> MySyncPolicy2;
	message_filters::Synchronizer<MySyncPolicy2> sync2_(MySyncPolicy2(200), velodyne_cloud_sub_, vision_sub_);
	sync2_.registerCallback(boost::bind(&sensor_fusion_and_monitors_cb, _1, _2));*/

	// Callback 2: Synchonize LiDAR point cloud and ego-vehicle odometry. Perform tracking and monitors (Approximate time)

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy2;
	message_filters::Synchronizer<MySyncPolicy2> sync2_(MySyncPolicy2(100), cloud_sub_, odom_sub_);
	sync2_.registerCallback(boost::bind(&clustering_precision_tracking_monitors_cb, _1, _2));

	// Load map

	string map_frame = "";
	//string map_path = ros::package::getPath("sec_map_manager") + "/maps/uah_lanelets_v42.osm"; // Load this path if /map_path argument does not exit in the network
	string map_path = ros::package::getPath("sec_map_manager") + "/maps/Town03_CARLA.osm";

        nh.param<string>("/map_path", map_path, map_path);
	loadedMap = make_shared<LaneletMap>(map_path);

	// Initialize colours

	std_msgs::ColorRGBA colour;

	// 0

	colour.a=1.0;
	colour.r=1.0;
	colour.g=0.0;
	colour.b=0.0;
	colours.push_back(colour);

	// 1

	colour.a=1.0;
	colour.r=0.0;
	colour.g=1.0;
	colour.b=0.0;
	colours.push_back(colour);
	
	// 2

	colour.a=1.0;
	colour.r=0.0;
	colour.g=0.0;
	colour.b=1.0;
	colours.push_back(colour);

	// 3

	colour.a=1.0;
	colour.r=1.0;
	colour.g=1.0;
	colour.b=0.0;
	colours.push_back(colour);

	// 4

	colour.a=1.0;
	colour.r=1.0;
	colour.g=0.0;
	colour.b=1.0;
	colours.push_back(colour);

	// 5

	colour.a=1.0;
	colour.r=0.0;
	colour.g=1.0;
	colour.b=1.0;
	colours.push_back(colour);

	// 6

	colour.a=1.0;
	colour.r=0.5;
	colour.g=0.0;
	colour.b=0.0;
	colours.push_back(colour);

	// 7

	colour.a=1.0;
	colour.r=0.0;
	colour.g=0.5;
	colour.b=0.0;
	colours.push_back(colour);

	// 8

	colour.a=1.0;
	colour.r=0.0;
	colour.g=0.0;
	colour.b=0.5;
	colours.push_back(colour);

	// 9

	colour.a=1.0;
	colour.r=0.5;
	colour.g=0.5;
	colour.b=0.0;
	colours.push_back(colour);

	// 3D viewer configuration

	/*if (VIEWER_3D)
	{
		viewer->setBackgroundColor (0.0, 0.0, 0.0);
		viewer->addCoordinateSystem (1.0);
		viewer->initCameraParameters ();
		viewer->setCameraPosition (-10, 0, 5, 0.3, 0, 0.95);
	}*/
	 
	// ROS Spin

	ros::spin ();

}

// End Main //


// Definitions of functions // 

// General use functions 

// Transform global coordinates ("map" frame) to local coordinates ("base_link" frame)
geometry_msgs::Point32 Global_To_Local_Coordinates(geometry_msgs::PointStamped point_global)
{
	// Parameters:
	// point_global: geometry_msgs::PointStamped point in global coordinate (with respect to the "map" frame)

	// Returns this point in local coordinates (with respect to the "base_link" frame)

	tf::Vector3 aux, aux2;
	geometry_msgs::PointStamped point_local;
	geometry_msgs::Point32 point32_local;

	aux.setX(point_global.point.x);
	aux.setY(point_global.point.y);
	aux.setZ(point_global.point.z);

	aux2 = transformBaseLinkOdom * aux;

	point_local.point.x = aux2.getX();
	point_local.point.y = aux2.getY();
	point_local.point.z = aux2.getZ();

	point32_local.x = point_local.point.x;
	point32_local.y = point_local.point.y;
	point32_local.z = point_local.point.z;

	return(point32_local);
}

geometry_msgs::Point32 Local_To_Global_Coordinates(geometry_msgs::PointStamped point_local)
{
	// Parameters:
	// point_global: geometry_msgs::PointStamped point in global coordinate (with respect to the "map" frame)

	// Returns this point in local coordinates (with respect to the "base_link" frame)

	tf::Vector3 aux, aux2;
	geometry_msgs::PointStamped point_global;
	geometry_msgs::Point32 point32_global;

	aux.setX(point_local.point.x);
	aux.setY(point_local.point.y);
	aux.setZ(point_local.point.z);

	aux2 = transformOdomBaseLink * aux;

	point_global.point.x = aux2.getX();
	point_global.point.y = aux2.getY();
	point_global.point.z = aux2.getZ();

	point32_global.x = point_global.point.x;
	point32_global.y = point_global.point.y;
	point32_global.z = point_global.point.z;

	return(point32_global);
}

// Obtain the distance between two 3D points 
float get_Centroids_Distance(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
	// Parameters: 
	// p1, p2: Two 3D points

	// Returns the Euclidean distance between both 3D points

	float root;
	root = float(sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)+pow(p1.z-p2.z,2)));

	return root;
}

// Check if a point is inside a certain area (https://jsbsan.blogspot.com/2011/01/saber-si-un-punto-esta-dentro-o-fuera.html)
void Inside_Polygon(Area_Point *polygon, int N, Area_Point p, bool &detection)
{
	// Parameters:
	// polygon: Pointer that points to the beggining of the array (Area_point type) that contains the defined area vertices
	// N: Number of vertices of the area
	// p: Detected point to be evaluated if is inside this area or not
	// detection: Boolean variable

	// Returns 1 if p is inside the defined area, or 0 in the contrary case

	int counter = 0;
	double xinters;
	Area_Point p1, p2;

	p1 = polygon[0]; // = *(polygon+0). Note that polygon is a pointer that points to the beggining of that array.
	
	for (int i=1; i<=N; i++)
	{
		p2 = polygon[i%N];

		if ((p.y > MIN_MONITOR(p1.y,p2.y)) && (p.y <= MAX_MONITOR(p1.y,p2.y)) && (p.x <= MAX_MONITOR(p1.x,p2.x)) && (p1.y != p2.y))             
		{
			xinters = p1.x + (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y);

			if ((p1.x == p2.x) || (p.x <= xinters))
			{
				counter++;
			}
		}

		p1 = p2; // p1 is updated
	}

	if (counter % 2 == 0) // even number of intersections -> The point is outside
	{
		detection = false;
	}
	else
	{
		detection = true; 
	}	
}

void Obstacle_in_Lanelet(pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaclesInLanelet_Ptr, geometry_msgs::PointStamped point_local, geometry_msgs::Point32 point32_global, geometry_msgs::PointStamped v1, geometry_msgs::PointStamped v2, geometry_msgs::PointStamped v3, geometry_msgs::PointStamped v4, ros::Time stamp, sec_msgs::Lanelet lanelet)
{
	current_obstacle.header.stamp = stamp;
	current_obstacle.header.frame_id = "/base_link";

	//cout<<"Point local x: "<<point_local.point.x;
	//cout<<"Point local y: "<<point_local.point.y;

	current_obstacle.pose.position.x = point_local.point.x;
	current_obstacle.pose.position.y = point_local.point.y;
	current_obstacle.pose.position.z = point_local.point.z;

	current_obstacle.laneletID = lanelet.id;

	geometry_msgs::Point32 pointaux32;
	pcl::PointXYZ pointaux;

	// Store the vertices of the obstacle

	current_obstacle.shape.points.clear();

	pointaux32.x = v1.point.x;
	pointaux32.y = v1.point.y;
	pointaux32.z = v1.point.z;

	current_obstacle.shape.points.push_back(pointaux32);

	pointaux32.x = v2.point.x;
	pointaux32.y = v2.point.y;
	pointaux32.z = v2.point.z;

	current_obstacle.shape.points.push_back(pointaux32);

	pointaux32.x = v3.point.x;
	pointaux32.y = v3.point.y;
	pointaux32.z = v3.point.z;

	current_obstacle.shape.points.push_back(pointaux32);

	pointaux32.x = v4.point.x;
	pointaux32.y = v4.point.y;
	pointaux32.z = v4.point.z;

	current_obstacle.shape.points.push_back(pointaux32);

	Obstacles.obstacles.push_back(current_obstacle);

	// Obstacle in lanelet

	pointaux.x=point32_global.x;
	pointaux.y=point32_global.y;
	pointaux.z=point32_global.z;

	ObstaclesInLanelet_Ptr->points.push_back(pointaux);
}

// Kalman functions

// Initialize a Kalman filter and put the initial values in the matrices used by the algorithm
cv::KalmanFilter initKalman(float x, float y, float z, float w, float h, float d, float sigmaR1, float sigmaR2, float sigmaR3, float sigmaQ1, float sigmaQ2, float sigmaQ3, float sigmaP)
{
	// Parameters:
	// x, y, z: Centroid of the object
	// w, h, d: Dimensions of the object
	// sigmaR1, sigmaR2, sigmaR3: Error Noise Covariance
	// sigmaQ1, sigmaQ2, sigmaQ3: Process Noise Covariance
	// sigmaP: Error Posterior Matrix

	// Creates Kalman Filter with 6 measures

	cv::KalmanFilter kf(12,6,0); // (int dynamParams, int measureParams, int controlParams = 0)
	
	// dynamParams: Dimensionality of the state
	// measureParams: Dimensionality of the measurement
	// controlParams: Dimensionality of the control vector
	// type: Type of the created matrices that should be CV_32F (by default) or CV_64F

	// Transition matrix
	
	kf.transitionMatrix = (cv::Mat_<float>(12,12) << 1,0,0,0,0,0,1,0,0,0,0,0,
	0,1,0,0,0,0,0,1,0,0,0,0,
	0,0,1,0,0,0,0,0,1,0,0,0,
	0,0,0,1,0,0,0,0,0,1,0,0,
	0,0,0,0,1,0,0,0,0,0,1,0,
	0,0,0,0,0,1,0,0,0,0,0,1,
	0,0,0,0,0,0,1,0,0,0,0,0,
	0,0,0,0,0,0,0,1,0,0,0,0,
	0,0,0,0,0,0,0,0,1,0,0,0,
	0,0,0,0,0,0,0,0,0,1,0,0,
	0,0,0,0,0,0,0,0,0,0,1,0,
	0,0,0,0,0,0,0,0,0,0,0,1);

	// Measurement matrix

	kf.measurementMatrix = (cv::Mat_<float>(6,12) << 1,0,0,0,0,0,0,0,0,0,0,0,
	0,1,0,0,0,0,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,0,0,0,0,
	0,0,0,1,0,0,0,0,0,0,0,0,
	0,0,0,0,1,0,0,0,0,0,0,0,
	0,0,0,0,0,1,0,0,0,0,0,0);

	// Process noise covariance

	kf.processNoiseCov = (cv::Mat_<float>(12,12) << sigmaQ1,0,0,0,0,0,0,0,0,0,0,0,
	0,sigmaQ1,0,0,0,0,0,0,0,0,0,0,
	0,0,sigmaQ2,0,0,0,0,0,0,0,0,0,
	0,0,0,sigmaQ2,0,0,0,0,0,0,0,0,
	0,0,0,0,sigmaQ3,0,0,0,0,0,0,0,
	0,0,0,0,0,sigmaQ3,0,0,0,0,0,0,
	0,0,0,0,0,0,sigmaQ1,0,0,0,0,0,
	0,0,0,0,0,0,0,sigmaQ1,0,0,0,0,
	0,0,0,0,0,0,0,0,sigmaQ2,0,0,0,
	0,0,0,0,0,0,0,0,0,sigmaQ2,0,0,
	0,0,0,0,0,0,0,0,0,0,sigmaQ3,0,
	0,0,0,0,0,0,0,0,0,0,0,sigmaQ3);

	// Measurement noise covariance

	kf.measurementNoiseCov = (cv::Mat_<float>(6,6) << sigmaR1,0,0,0,0,0,
	0,sigmaR1,0,0,0,0,
	0,0,sigmaR2,0,0,0,
	0,0,0,sigmaR2,0,0,
	0,0,0,0,sigmaR3,0,
	0,0,0,0,0,sigmaR3);

	// Error posterior matrix

	kf.errorCovPost = (cv::Mat_<float>(12,12) << sigmaP,0,0,0,0,0,0,0,0,0,0,0,
	0,sigmaP,0,0,0,0,0,0,0,0,0,0,
	0,0,sigmaP,0,0,0,0,0,0,0,0,0,
	0,0,0,sigmaP,0,0,0,0,0,0,0,0,
	0,0,0,0,sigmaP,0,0,0,0,0,0,0,
	0,0,0,0,0,sigmaP,0,0,0,0,0,0,
	0,0,0,0,0,0,sigmaP,0,0,0,0,0,
	0,0,0,0,0,0,0,sigmaP,0,0,0,0,
	0,0,0,0,0,0,0,0,sigmaP,0,0,0,
	0,0,0,0,0,0,0,0,0,sigmaP,0,0,
	0,0,0,0,0,0,0,0,0,0,sigmaP,0,
	0,0,0,0,0,0,0,0,0,0,0,sigmaP);

	// Initialize filter with given data

	// State post: Corrected state (x(k)): x(k) = x'(k)+K(k)*(z(k)-H*x'(k))

	kf.statePost.at<float>(0) = x;
	kf.statePost.at<float>(1) = y;
	kf.statePost.at<float>(2) = z;
	kf.statePost.at<float>(3) = w;
	kf.statePost.at<float>(4) = h;
	kf.statePost.at<float>(5) = d;
	kf.statePost.at<float>(6) = 0;
	kf.statePost.at<float>(7) = 0;
	kf.statePost.at<float>(8) = 0;
	kf.statePost.at<float>(9) = 0;
	kf.statePost.at<float>(10) = 0;
	kf.statePost.at<float>(11) = 0;
	
	// State Pre: Predicted state (x'(k)): x(k) = A*x(k-1)+B*u(k)
	 
	kf.statePre.at<float>(0) = x;
	kf.statePre.at<float>(1) = y;
	kf.statePre.at<float>(2) = z;
	kf.statePre.at<float>(3) = w;
	kf.statePre.at<float>(4) = h;
	kf.statePre.at<float>(5) = d;
	 
	return kf;
}

// Update Kalman filter
void updateKalman(cv::KalmanFilter &kf, float x, float y, float z, float w, float h, float d, bool useMeasurement)
{
	// Parameters:
	// kf: Kalman filter calculated in initKalman
	// x, y, z: Center of the obstacle
	// w, h, d: Dimensions of the object
	// useMeasurement: If true, use measurements. If false, use previous prediction

	cv::Mat prediction; // cv::Mat n-dimensional dense array class
	cv::Mat measurement(6,1,CV_32FC1);

	// Kalman prediction

	prediction = kf.predict(); // predict Public member function computes a predicted state

	if (useMeasurement == false) // Use as measurements the previous prediction
	{
		measurement.at<float>(0) = kf.statePre.at<float>(0);
		measurement.at<float>(1) = kf.statePre.at<float>(1);
		measurement.at<float>(2) = kf.statePre.at<float>(2);
		measurement.at<float>(3) = kf.statePre.at<float>(3);
		measurement.at<float>(4) = kf.statePre.at<float>(4);
		measurement.at<float>(5) = kf.statePre.at<float>(5);
	}
	else // Use as measurements the current measurements
	{
		measurement.at<float>(0) = x;
		measurement.at<float>(1) = y;
		measurement.at<float>(2) = z;
		measurement.at<float>(3) = w;
		measurement.at<float>(4) = h;
		measurement.at<float>(5) = d;
	}

	// Correct the Kalman filter based on the measurements

	// They current measurements are usually used as cv::Mat measurements matrix since it is usually better to correct
	// the Kalman filter based on the measurements of a good sensor rather than on the prediction

	kf.correct(measurement); 
}

// Obtain a Kalman prediction for each object
Points_Kalman getKalmanPrediction(cv::KalmanFilter kf)
{
	// Parameters
	// kf: Kalman Filter
	
	// Returns the prediction in Kalman Points format

	Points_Kalman predicted_object;

	predicted_object.x = kf.statePre.at<float>(0);
	predicted_object.y = kf.statePre.at<float>(1);
	predicted_object.z = kf.statePre.at<float>(2);
	predicted_object.w = kf.statePre.at<float>(3);
	predicted_object.h = kf.statePre.at<float>(4);
	predicted_object.d = kf.statePre.at<float>(5);

	return predicted_object;
}

// Point Cloud functions

// Filter the input LiDAR point cloud in terms of xyz dimensions
pcl::PointCloud<pcl::PointXYZRGB> xyz_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_filtered_cloud)
{
	// Parameters:
	// non_filtered_cloud: Input LiDAR point cloud 

	// Returns a xyz filtered point cloud

	pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;

	for (int i=0; i<non_filtered_cloud->points.size(); i++)
	{
		pcl::PointXYZRGB aux_point; // A single point
		aux_point = non_filtered_cloud->points[i];

		// Z must be above the sidewalk. If the frame of the point cloud is /base_link, located on the floor, in order to avoid the sidewalk we must filter
		// above 0.2 m (sidewalk height)

		if (aux_point.z > 0.2)
		{
			filtered_cloud.points.push_back(aux_point);
		}
	}

	return filtered_cloud;
}

// Filter the input LiDAR point cloud in terms of angle
pcl::PointCloud<pcl::PointXYZRGB> angle_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_filtered_cloud) 
{
	// Parameters:
	// non_filtered_cloud: Input LiDAR point cloud 

	// Returns an angle filtered point cloud

	pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;

	float field_of_view = 80; // Out of this field of view (centered in the LiDAR) the point cloud is discarded

	// Take into account the field of view (in degrees) of your camera

	for (int i=0; i<non_filtered_cloud->points.size(); i++)
	{
		pcl::PointXYZRGB aux_point; // A single point
		aux_point = non_filtered_cloud->points[i];

		double aux_point_angle = atan2(aux_point.x, aux_point.y); // Angle with respect to the "base_link" frame

		if ((aux_point_angle<((field_of_view/2)*(M_PI/180))) && (aux_point_angle<((-field_of_view/2)*(M_PI/180))))
		{
			filtered_cloud.push_back(aux_point);
		}

		filtered_cloud.push_back(aux_point);
	}
	return filtered_cloud;
}

// Extract clusters from the coloured XYZ filtered LiDAR point cloud according to the input cluster parameters
void cluster_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud, float tolerance, int min_cluster, int max_cluster, vector<Object> *output_objects, int *output_objects_number)
{
	// Parameters:
	// filtered_cloud: Coloured XYZ filtered LiDAR point cloud that contains the clusters
	// tolerance: Tolerance of clusters
	// min_cluster: Minimum size of a cluster
	// max_cluster: Maximum size of a cluster
	// output_objects: Pointer that points to the array that contains the clusters
	// output_objects_number: Number of objects

	// This function only takes into account the size of the clusters, not its colour

}

// Extract clusters from the coloured XYZ angle filtered LiDAR point cloud according to the input cluster parameters
void segmentation_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud, float r_max, float r_min, float g_max, float g_min, float b_max, float b_min, float tolerance, int min_cluster, int max_cluster, vector<Object> *output_objects, int *number_output_objects, string type)
{
	// Parameters
	// filtered_cloud: Coloured XYZ angle filtered LiDAR point cloud that contains the clusters
	// r_max, r_min, g_max, g_min, b_min, b_max: RGB colour limits (0 - 255)
	// tolerance: Tolerance of clusters
	// min_cluster: Minimum size of cluster
	// max_cluster: Maximum size of cluster
	// output_objects: Obtained clusters
	// number_output_objects: Number of obtained clusters

	// Delete points that are not between the colour limits

	pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
	color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, r_max)));
	color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::GT, r_min)));
	color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, g_max)));
	color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::GT, g_min)));
	color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::LT, b_max)));
	color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, b_min)));

	pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem;
	condrem.setCondition(color_cond);
	condrem.setInputCloud (filtered_cloud);
	condrem.setKeepOrganized(false);
	condrem.filter (*filtered_cloud);

	if (DEBUG)
	{
		cout<<"Point Cloud RGB after filtering has: "<<filtered_cloud->points.size()<<" data points"<<endl;
	}

	// Extract clusters from point cloud

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (filtered_cloud);
	vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (tolerance); 
	ec.setMinClusterSize (min_cluster); 
	ec.setMaxClusterSize (max_cluster);
	ec.setSearchMethod (tree);
	ec.setInputCloud (filtered_cloud);
	ec.extract (cluster_indices);

	// Store the clusters

	for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
 
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			cloud_cluster->points.push_back (filtered_cloud->points[*pit]); 
		}

		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
 
		// Initialize point cloud vertices. Set to +/- INFINITY to ensure a proper behaviour for the first cluster

		float x_min = INFINITY; 
		float y_min = INFINITY;
		float z_min = INFINITY;
		float x_max = -INFINITY;
		float y_max = -INFINITY;
		float z_max = -INFINITY;
 
		float centroid_x = -INFINITY;
		float centroid_y = -INFINITY;
		float centroid_z = -INFINITY;
		float length_x = -INFINITY;
		float width_y = -INFINITY;
		float height_z = -INFINITY;
 
		for (int i = 0; i < cloud_cluster->points.size(); i++)
		{
			if (cloud_cluster->points[i].x < x_min)		
			{
				x_min = cloud_cluster->points[i].x;
			}
 
			if (cloud_cluster->points[i].y < y_min)		
			{
				y_min = cloud_cluster->points[i].y;
			}
 
			if (cloud_cluster->points[i].z < z_min)	
			{
				z_min = cloud_cluster->points[i].z;		
			}
			if (cloud_cluster->points[i].x > x_max)
			{
				x_max = cloud_cluster->points[i].x;
			}
			if (cloud_cluster->points[i].y > y_max)
			{
				y_max = cloud_cluster->points[i].y;
			}
			if (cloud_cluster->points[i].z > z_max)
			{
				z_max = cloud_cluster->points[i].z;
			}
		}

		// Centroid

		//centroid_x = (x_max+x_min)/2.0;
		//centroid_y = (y_max+y_min)/2.0;
		//centroid_z = (z_max+z_min)/2.0;

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*cloud_cluster,centroid);

		geometry_msgs::PointStamped local_centroid;
		geometry_msgs::Point32 global_centroid;

		local_centroid.point.x = centroid_x;
		local_centroid.point.y = centroid_y;
		local_centroid.point.z = centroid_z;

		global_centroid = Local_To_Global_Coordinates(local_centroid);
 
		// Object measurements

		object.x_max = x_max;
		object.x_min = x_min;
		object.y_max = y_max;
		object.y_min = y_min;
		object.z_max = z_max;
		object.z_min = z_min;

		// Local coordinates with respect to the velodyne

		//object.centroid_x = centroid_x;
		//object.centroid_y = centroid_y;
		//object.centroid_z = centroid_z;

		object.centroid_x = centroid[0];
		object.centroid_y = centroid[1];
		object.centroid_z = centroid[2];

		// Cloud

		object.cloud = cloud_cluster;

		// Type

		object.type = type;
 
		output_objects->push_back(object);
 
		if (DEBUG)
		{
			 cout << "Cluster "<< number_of_clusters <<" centroid_x "<<centroid_x<<" centroid_y "<<centroid_y <<" centroid_z "<< centroid_z<<endl;
		}

		*number_output_objects = *number_output_objects + 1;
		number_of_clusters++;
	}
}

// Merging different clusters with the same XY position
vector<Merged_Object> merging_z(vector<Object> objects)
{
	// Parameters
	// Objects: Objects to merge
	
	// Returns z merged objects

	vector<Merged_Object> merged_objects;
	merged_objects.clear();

	

}


// Callbacks //

// Callback to store the full path
void route_cb(const sec_msgs::Route::ConstPtr& route_msg)
{
	route = *route_msg;
}

// Callback to allow the car continue (If stop signal is received (Ego-vehicle is stopped in a STOP regulatory element))
void waiting_cb(const std_msgs::Empty msg)
{
	stop = 2;
}

// Callback to extract information of the regeleme and monitorized_lanelets_msg topics
void regelement_cb(const sec_msgs::RegElem::ConstPtr& regelem, const sec_msgs::Route::ConstPtr& monitorized_lanelets_msg)
{
	// Parameters: 
	// regelem: Pointer to the information of the /currentRegElem topic
	// monitorized_lanelets_msg: Pointer to the information of the /monitorizedLanelets topic

	// Global variables

	monitorized_lanelets = *monitorized_lanelets_msg; 
	current_regulatory_element = *regelem;

	// Initialize use cases lanelets

	pedestrian_crossing_lanelets.route.clear();
	merging_lanelets.route.clear();
	route_lanelets.route.clear();

	// TODO: DELETE THESE VARIABLES?
	left_lanelets.route.clear();
	right_lanelets.route.clear();
	route_left_lanelets.route.clear();
	route_right_lanelets.route.clear();
	all_lefts.route.clear();
	all_rights.route.clear();

	// Initialize pedestrian crossing and merging monitors

	merging_monitor = false;
	pedestrian_crossing_monitor = false;

	// Evaluate the current regulatory elements and monitorized lanelets

	// Pedestrian crossing

	if (!strcmp(regelem->type.c_str(),"pedestrian_crossing"))
	{
		if (regelem->distance<30) // It the pedestrian crossing is within 30 m
		{
			// Activate pedestrian crossing monitor

			pedestrian_crossing_monitor = true;

			if (regelem->laneletID != id_lanelet_pedestrian_crossing)
			{
				global_pedestrian_crossing_occupied = 0; // We are in a new pedestrian crossing, so the presence of pedestrians must be reevaluated 
			}

			// Store current regulatory element ID, which corresponds to a pedestrian crossing

			id_lanelet_pedestrian_crossing = regelem->laneletID;

			// Store pedestrian crossing lanelets

			for (int i=0; i<monitorized_lanelets.route.size(); i++)
			{
				//cout<<"\nType: "<<monitorized_lanelets.route[i].type.c_str()<<endl;
				/*if (!strcmp(monitorized_lanelets.route[i].type.c_str()," pedestrian_crossing"))
				{
					pedestrian_crossing_lanelets.route.push_back(monitorized_lanelets.route[i]);
				}*/

				string type = monitorized_lanelets.route[i].type.c_str();
				istringstream iss(type);

				do
				{
					string subs;
					iss >> subs;

					//cout<<"\nSubs pedestrian: "<<subs.c_str()<<endl;
						
					if (!strcmp(subs.c_str(),"pedestrian_crossing"))    
					{	
						pedestrian_crossing_lanelets.route.push_back(monitorized_lanelets.route[i]);
					}
				}while(iss);
			}
		}
		else // Pedestrian crossing far away, turn off pedestrian crossing monitor
		{
			pedestrian_crossing_monitor = false;
		}

		// Obtain data from regelem to build the pedestrian crossing area

		polygon_area[0].x = regelem->A1.latitude;
		polygon_area[0].y = regelem->A1.longitude;
		polygon_area[1].x = regelem->A2.latitude;
		polygon_area[1].y = regelem->A2.longitude;
		polygon_area[2].x = regelem->A3.latitude;
		polygon_area[2].y = regelem->A3.longitude;
		polygon_area[3].x = regelem->A4.latitude;
		polygon_area[3].y = regelem->A4.longitude;
	}
	else // There is not a pedestrian crossing, turn off pedestrian crossing monitor
	{
		pedestrian_crossing_monitor = false;
	}
	
	// Give way or STOP

	if (!strcmp(regelem->type.c_str(),"give way") || !strcmp(regelem->type.c_str(),"give_way") || !strcmp(regelem->type.c_str(),"stop"))
	{
		if (regelem->distance<30) // It the give way/stop is within 30 m
		{
			// Activate merging monitor

			merging_monitor = true;

			int id_lanelet_merging = regelem->laneletID;

			// Store merging lanelets related to regulatory element and activate merging monitor

			//cout<<"\n..............."<<endl;
			for (int i=0; i<monitorized_lanelets.route.size(); i++)
			{
				//cout<<"Type: "<<monitorized_lanelets.route[i].type.c_str()<<"-> ID: "<<monitorized_lanelets.route[i].id<<endl;

				string type = monitorized_lanelets.route[i].type.c_str();
				istringstream iss(type);

				do
				{
					string subs;
					iss >> subs;

					//cout<<"\nSubs merging: "<<subs.c_str()<<endl;

					
						
					if (!strcmp(subs.c_str(),"id"))
					{	
						string subs;
						iss >> subs;

						if(!strcmp(subs.c_str(), to_string(id_lanelet_merging).c_str()))
						{
							merging_lanelets.route.push_back(monitorized_lanelets.route[i]);
						}
					}
				}while(iss);
			}
			//cout<<"..............."<<endl;
		}
		else // Give way or STOP far away, turn off pedestrian monitor
		{
			merging_monitor = false;
		}

	
		// If particularly STOP

		if (!strcmp(regelem->type.c_str(),"stop"))
		{
			// if stop monitor is 0 (inactive) or 1 (active), stop variable is 1 (vehicle cannot continue)
			if (stop != 2)
			{
				stop = 1;
			}
		}
		else // If current regulatory element is not a STOP, turn off the variable
		{
			stop = 0;
		}
	}
	else // There is neither a give way not a STOP, turn off merging monitor
	{
		merging_monitor = false;
	}

	// Create a variable with the monitorized lanelets that are in the route
	// Route:Lanelets in which the ego-vehicle is supposed to be driven (Coloured in blue in RVIZ)

	for (int i=0; i<monitorized_lanelets.route.size(); i++)
	{
		string type = monitorized_lanelets.route[i].type.c_str();
		istringstream iss(type);

		do
		{
			string subs;
			iss >> subs;

			if (!strcmp(subs.c_str(),"route"))
			{
				route_lanelets.route.push_back(monitorized_lanelets.route[i]);
			}
		}while(iss);
	}
	
	// TODO: IMPROVE HOW TO DETERMINE EXACTLY THE CURRENT LEFT AND RIGHT LANELETS 

	// If the current monitorized lanelets are based on a right-defined route

	for (int i=0; i<monitorized_lanelets.route.size(); i++)
	{
		string type = monitorized_lanelets.route[i].type.c_str();
		istringstream iss(type);

		do
		{
			string subs;
			iss >> subs;

			if (!strcmp(subs.c_str(),"left"))
			{
				left_lanelets.route.push_back(monitorized_lanelets.route[i]);
			}

			// If there exist left lanelets, it means that the user has planified the route along the right lanelet of the road. So, the current right lanelet is 				  
                        // represented by "route" type in Monitorized Lanelets

			if(!strcmp(subs.c_str(),"route") || !strcmp(subs.c_str(),"merging split route") || !strcmp(subs.c_str(),"split merging route"))
			{
				route_right_lanelets.route.push_back(monitorized_lanelets.route[i]); 
			}
		 
		}while(iss);
	}

	// If the current monitorized lanelets are based on a left-defined route

	if (left_lanelets.route.size() == 0)
	{
		for (int i=0; i<monitorized_lanelets.route.size(); i++)
		{
			string type = monitorized_lanelets.route[i].type.c_str();
			istringstream iss(type);

			do
			{
				string subs;
				iss >> subs;

				if(!strcmp(subs.c_str(),"route") || !strcmp(subs.c_str(),"merging split route") || !strcmp(subs.c_str(),"split merging route"))
				{
					route_left_lanelets.route.push_back(monitorized_lanelets.route[i]); 
				}

				if (i>0 && !strcmp(subs.c_str(),"lanelet"))
				{
					right_lanelets.route.push_back(monitorized_lanelets.route[i]);
				}
			 
			}while(iss);
		}
	}

	if (left_lanelets.route.size()>0)
	{
		for (int i=0;i<left_lanelets.route.size();i++){
		all_lefts.route.push_back(left_lanelets.route[i]);}

		for (int i=0;i<route_right_lanelets.route.size();i++){
		all_rights.route.push_back(route_right_lanelets.route[i]);}
	}
	else
	{
		for (int i=0;i<route_left_lanelets.route.size();i++){
		all_lefts.route.push_back(route_left_lanelets.route[i]);}

		for (int i=0;i<right_lanelets.route.size();i++){
		all_rights.route.push_back(right_lanelets.route[i]);}
	}
}

void clustering_precision_tracking_monitors_cb(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const nav_msgs::Odometry::ConstPtr& odom_msg)
{
	cout<<"------------------------------------------------"<<endl;
	//ROS_INFO("Time: [%lf]", (double)ros::Time::now().toSec());

	// Auxiliar variables

	std::vector<int> associated_filter;

	// Obtain the movement of the ego-vehicle in X and Y (Global coordinates) and Orientation (Yaw)

	double displacement_x_global = odom_msg->pose.pose.position.x - previous_odom.pose.pose.position.x;
	double displacement_y_global = odom_msg->pose.pose.position.y - previous_odom.pose.pose.position.y;
	double yaw = tf::getYaw(odom_msg->pose.pose.orientation);

	// Obtain displacement of the ego-vehicle and Velocities in Local coordinates

	//cout<<"Global displacement x: "<<displacement_x_global<<endl;
	//cout<<"Global displacement y: "<<displacement_y_global<<endl;

	double displacement_x_local = displacement_x_global*cos(yaw) + displacement_y_global*sin(yaw);
	double displacement_y_local = displacement_x_global*(-sin(yaw)) + displacement_y_global*cos(yaw);

	//cout<<"Local displacement x: "<<displacement_x_local<<endl;
	//cout<<"Local displacement y: "<<displacement_y_local<<endl;

	double time = odom_msg->header.stamp.toSec() - previous_odom.header.stamp.toSec();

	double vel_x_with_yaw = displacement_x_local/time;
	double vel_y_with_yaw = displacement_y_local/time;
	double abs_vel = sqrt(pow(vel_x_with_yaw,2)+pow(vel_y_with_yaw,2));

	double vel_x = displacement_x_global/time;
	double vel_y = displacement_y_global/time;

	//cout<<"My vel x: "<<vel_x<<endl;
	//cout<<"My vel y: "<<vel_y<<endl;
	//cout<<"My yaw: "<<yaw;
	
	// Store previous odometry

	previous_odom = *odom_msg;

	// Store odom in different formats: TODO: Required?

	geodesy::UTMPoint odomUTMmsg;
	odomUTMmsg.band = utm_origin.band;
	odomUTMmsg.zone = utm_origin.zone;
	odomUTMmsg.altitude = 0;
	odomUTMmsg.easting = odom_msg->pose.pose.position.x + utm_origin.easting;
	odomUTMmsg.northing = odom_msg->pose.pose.position.y + utm_origin.northing;
 	geographic_msgs::GeoPoint latLonOdom;
	latLonOdom = geodesy::toMsg(odomUTMmsg);

	try
	{
		// Obtain transforms between frames and store

		listener->lookupTransform("base_link", "ego_vehicle/camera/semantic_segmentation/semantic", lidar_msg->header.stamp, transformBaseLinkBaseCamera);
		listener->lookupTransform("map", "base_link", lidar_msg->header.stamp, transformOdomBaseLink);
		listener->lookupTransform("base_link", "map", lidar_msg->header.stamp, transformBaseLinkOdom);
	}
	catch(tf::TransformException& e)
	{
		
		cout<<e.what();
		return; // Exit the program
	}

	// Auxiliar Point Clouds

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr vlp_cloud_Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaclesInLanelet_Ptr (new pcl::PointCloud<pcl::PointXYZ>); 
	pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaclesInPedestrian_Ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaclesMerging_Ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_xyz_filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_angle_filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cars (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pedestrians (new pcl::PointCloud<pcl::PointXYZRGB>);

	sensor_msgs::PointCloud2 msga;

	// If full cloud is used ...

	pcl::fromROSMsg(*lidar_msg, *vlp_cloud_Ptr); // The content pointed by non_filtered_cloud will contain the content of lidar_msg
	//pcl::fromROSMsg(*lidar_msg, *non_filtered_cloud); // The content pointed by non_filtered_cloud will contain the content of lidar_msg

	// Rotate the point cloud according to /base_link frame

	pcl::toROSMsg(*vlp_cloud_Ptr,msga);

	tfBaseLinkBaseCamera = tf::Transform(transformBaseLinkBaseCamera.getRotation(), transformBaseLinkBaseCamera.getOrigin()); // Transform from camera frame to LiDAR frame

	sensor_msgs::PointCloud2 aux_cloud;

	pcl_ros::transformPointCloud("/base_link",transformBaseLinkBaseCamera,msga,aux_cloud);
	pcl::fromROSMsg(aux_cloud,*non_filtered_cloud);

	// XYZ Filter

	pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud = xyz_filter(non_filtered_cloud);
	*xyz_filtered_cloud = filtered_cloud;
	//xyz_filtered_cloud = non_filtered_cloud;
	
	// Publish as ROS message

	sensor_msgs::PointCloud2 lidar_cloud;
	pcl::toROSMsg(*xyz_filtered_cloud, lidar_cloud);
	lidar_cloud.header.frame_id = "base_link"; // In addition to apply the transform between camera frame and LiDAR frame you have set the point cloud in the corresponding
	// frame, in this case LiDAR (but z-filtered, so base_link)
	lidar_cloud.header.stamp = lidar_msg->header.stamp;

	pub_LiDAR_Pointcloud_Coloured_XYZ_Filtered.publish(lidar_cloud);

	if (DEBUG)
	{
		cerr<<"Point Cloud data: "<<non_filtered_cloud->points.size()<<" points"<<endl;
	}

	// Initialize number of clusters

	number_of_clusters = 0;

	// Angle filter to extract the clusters only from this portion of the point cloud (since with a single camera
	// we do not have a whole coloured point cloud)

	pcl::fromROSMsg(lidar_cloud,*aux_xyz_filtered_cloud); // Now xyz_filtered
	pcl::PointCloud<pcl::PointXYZRGB> aux_filtered_cloud = angle_filter(aux_xyz_filtered_cloud);
	*xyz_angle_filtered_cloud = aux_filtered_cloud;
	//xyz_angle_filtered_cloud = aux_xyz_filtered_cloud;

	// Publish as ROS message

	sensor_msgs::PointCloud2 lidar_cloud2;
	pcl::toROSMsg(*xyz_angle_filtered_cloud, lidar_cloud2);
	lidar_cloud2.header.frame_id = "base_link";
	lidar_cloud2.header.stamp = lidar_msg->header.stamp;

	pub_LiDAR_Pointcloud_Coloured_XYZ_Angle_Filtered.publish(lidar_cloud2);

	// Fill each PCL pointer with the xyz_angle_filtered_cloud to extract each corresponding object (It is possible to add more ...)

	*cloud_cars = *xyz_angle_filtered_cloud;
	*cloud_pedestrians = *xyz_angle_filtered_cloud;

	// Cluster segmentation 

	vector<Object> cars, pedestrians, total_objects;
	cars.clear();
	pedestrians.clear();
	total_objects.clear();

	int number_cars = 0, number_pedestrians = 0;

	// TODO: The estimation of the centroid should be more precise! -> In the object centroid is not consistent during the time, the velocity estimation is impossible

	segmentation_filter(cloud_cars, 10, -1, 10, -1, 155, 132, 2, 5, 2000, &cars, &number_cars, "car");   // Cars (CARLA) RGB: 0, 0, 142
	segmentation_filter(cloud_pedestrians, 230, 210, 30, 10, 70, 50, 1, 5, 200, &pedestrians, &number_pedestrians, "pedestrian");   // Pedestrians (CARLA) RGB: 220, 20, 60

	for (int i=0; i<cars.size(); i++)
	{
		total_objects.push_back(cars[i]);
	}

	for (int i=0; i<pedestrians.size(); i++)
	{
		total_objects.push_back(pedestrians[i]);
	}

	cout<<"\nNumber of cars: "<<number_cars;
	cout<<"\nNumber of pedestrians: "<<number_pedestrians;
	cout<<"\nTotal objects: "<<total_objects.size()<<endl<<endl;

	// Obstacles merging and filter updates //

	if (number_cars != 0 || number_pedestrians != 0)
	{
		// Required merging_z ?

		// Kalman and Precision Trackers

		Points_Kalman point_kalman_aux, point_kalman_prediction;

		pcl::PointXYZ p1, p2;
		int i1, i2, i3;

		// Travel all objects

		for (int i=0; i<total_objects.size(); i++)
		{
			point_kalman_aux.x = total_objects[i].centroid_x;
			point_kalman_aux.y = total_objects[i].centroid_y;
			point_kalman_aux.z = total_objects[i].centroid_z;
			point_kalman_aux.w = total_objects[i].x_max - total_objects[i].x_min;
			point_kalman_aux.h = total_objects[i].y_max - total_objects[i].y_min;
			point_kalman_aux.d = total_objects[i].z_max - total_objects[i].z_min;
			point_kalman_aux.type = total_objects[i].type;

			Eigen::Vector4f centroid;
			pcl::compute3DCentroid(*total_objects[i].cloud,centroid);
			p1.x = centroid[0];
			p1.y = centroid[1];
			p1.z = 0;

			// Associate the object with a filter

			float distance = 0;
			float min_distance = 100000;
			int minKalman = -1; // Number of Kalman filter

			cout<<"Number of Kalman Filters: "<<kfs.size()<<endl;
			cout<<"Number of Precision Trackers: "<<pTrackers.size()<<endl;

			for (int j=0; j<pTrackers.size(); j++)
			{
				cout<<"pTracker ID: "<<pTrackers[i].id<<endl;

				// If pTracker is not too old, compare
				if ((lidar_msg->header.stamp.toSec() - pTrackers[j].time) < TIME_PRECISION_TRACKING)
				{
					// Calculate the distance between the detected centroid and the predicted tracker centroid

					p2.x = pTrackers[j].centroid[0] + (pTrackers[j].estimated_velocity[0]*(lidar_msg->header.stamp.toSec()-pTrackers[j].time));
					p2.y = pTrackers[j].centroid[1] + (pTrackers[j].estimated_velocity[1]*(lidar_msg->header.stamp.toSec()-pTrackers[j].time));
					p2.z=0;

					distance = get_Centroids_Distance(p1,p2);

					/*cout<<"P Tracker centroid x: "<<pTrackers[j].centroid[0]<<endl;
					cout<<"P Tracker centroid y: "<<pTrackers[j].centroid[1]<<endl;
					cout<<"P Tracker velocity x: "<<pTrackers[j].estimated_velocity[0]<<endl;
					cout<<"P Tracker velocity y: "<<pTrackers[j].estimated_velocity[1]<<endl;
					cout<<"Distance: "<<distance<<endl;*/

					if ((distance < min_distance) && (distance < 5))
					{
						min_distance = distance;
						minKalman = j; // Current Kalman filter associated to j-precision tracker

						if (DEBUG)
						{
							ROS_ERROR("Obstacle %d associated to %d precision tracker\n", i, j);
						}
					}
				}
			}

			if (minKalman == -1) // Current Kalman filter is not associated with the obstacle. Create a new Kalman filter and Precision Tracker
			{
				// Init new Kalman Filter and add to Kalman list
				
				kfs.push_back(initKalman(point_kalman_aux.x, point_kalman_aux.y, point_kalman_aux.z, point_kalman_aux.w, point_kalman_aux.h, point_kalman_aux.d, 0.999, 0.999, 0.999, 0.001,0.001, 0.001, 0.01));  
				
				// Add "time" (Counter)
					
				kfsTime.push_back(20);
				min_distance = 0;

				// Kalman filter number

				minKalman = kfs.size() - 1; // Now it is the new one, so it has the highest number

				// Precision Tracking variables

				double horizontal_sensor_resolution, vertical_sensor_resolution;
				Eigen::Vector3f centroid;
				Eigen::Vector4f centroid2;
				pcl::compute3DCentroid(*total_objects[i].cloud,centroid2);

				centroid[0] = centroid2[0];
				centroid[1] = centroid2[1];
				centroid[2] = centroid2[2];

				// Create Precision Tracker (Evaluate using 3D or color if the machine allows to work at correct ratio)

				params.useColor = false;
				params.use3D = false;
				precision_tracking::Tracker tracker(&params);
				tracker.setPrecisionTracker(boost::make_shared<precision_tracking::PrecisionTracker>(&params));
				precision_tracking::getSensorResolution(centroid, &horizontal_sensor_resolution, &vertical_sensor_resolution);

				if (DEBUG)
				{
					ROS_ERROR("Horizontal resolution %f Vertical resolution %f\n", horizontal_sensor_resolution, vertical_sensor_resolution);
				}

				tracker.addPoints(total_objects[i].cloud, lidar_msg->header.stamp.toSec(), horizontal_sensor_resolution, vertical_sensor_resolution, &estimated_velocity);

				// Estimated velocity in absolute value (Relative velocity estimation w.r.t. the ego-vehicle + ego-vehicle velocity)

				estimated_velocity[0] = estimated_velocity[0] + vel_x_with_yaw;
				estimated_velocity[1] = estimated_velocity[1] + vel_y_with_yaw;

				// Precision Trackers

				trackers.push_back(tracker);

				Precision_Trackers pt;
				pt.centroid = centroid;
				pt.estimated_velocity = estimated_velocity;
				pt.time = lidar_msg->header.stamp.toSec();
				pt.size[0] = fabs(total_objects[i].x_max - total_objects[i].x_min);
				pt.size[1] = fabs(total_objects[i].y_max - total_objects[i].y_min);
				pt.size[2] = fabs(total_objects[i].z_max - total_objects[i].z_min);
				pt.id = indexpTrackers;
				pt.type = total_objects[i].type;
				pt.pedestrian_state = 0;
				pTrackers.push_back(pt);

				total_objects[i].id = indexpTrackers; // Current object ID = New precision tracker ID

				associated_filter.push_back(indexpTrackers);
				indexpTrackers++;	
			}
			else // if minKalman != 1, the object is already associated with a Precision Tracker filter 
			{
				// Update its associated Kalman filter (If true the last argument, use these measurements as correction parameters)

				updateKalman(kfs[minKalman], point_kalman_aux.x, point_kalman_aux.y, point_kalman_aux.z, point_kalman_aux.w, point_kalman_aux.h, point_kalman_aux.d, true);

				// Update the Precision Tracker sensor resolution 

				double horizontal_sensor_resolution, vertical_sensor_resolution;
				Eigen::Vector3f centroid;
				Eigen::Vector4f centroid2;

				pcl::compute3DCentroid(*total_objects[i].cloud,centroid2);
				centroid[0] = centroid2[0];
				centroid[1] = centroid2[1];
				centroid[2] = centroid2[2];

				pTrackers[minKalman].previous_centroid = pTrackers[minKalman].centroid;
				pTrackers[minKalman].previous_velocity = pTrackers[minKalman].estimated_velocity;

				precision_tracking::getSensorResolution(centroid, &horizontal_sensor_resolution, &vertical_sensor_resolution);

				trackers[minKalman].addPoints(total_objects[i].cloud, lidar_msg->header.stamp.toSec(), horizontal_sensor_resolution, vertical_sensor_resolution, &estimated_velocity);

				// Estimated velocity in absolute value (Relative velocity estimation w.r.t. the ego-vehicle + ego-vehicle velocity)

				/*cout<<"\nObject x: "<<estimated_velocity[0]<<endl;
				cout<<"Object y: "<<estimated_velocity[1]<<endl;
				cout<<"\nMe x: "<<vel_x_with_yaw<<endl;
				cout<<"Me y: "<<vel_y_with_yaw<<endl;*/

				estimated_velocity[0] = estimated_velocity[0] + vel_x_with_yaw;
				estimated_velocity[1] = estimated_velocity[1] + vel_y_with_yaw;

				double pTrackerVelocity = sqrt(pow(estimated_velocity[0],2)+pow(estimated_velocity[1],2)); // Note that the pTracker stores the local velocity of the object + ego-vehicle velocity

				/*if ((estimated_velocity[0] > 2*pTrackers[minKalman].previous_velocity[0]) || (estimated_velocity[0] < 0.5*pTrackers[minKalman].previous_velocity[0]) || (estimated_velocity[1] > 2*pTrackers[minKalman].previous_velocity[1]) || (estimated_velocity[0] < 0.5*pTrackers[minKalman].previous_velocity[0]))
				{
					estimated_velocity = pTrackers[minKalman].previous_velocity;
				}*/

				/*cout<<"\nPrecision Tracker velocity absolute velocity : "<<setprecision(2)<<pTrackerVelocity<<" km/h"<<endl;

				cout<<"\nPrecision Tracker centroid x: "<<pTrackers[minKalman].centroid[0]<<endl;
				cout<<"\nPrecision Tracker centroid y: "<<pTrackers[minKalman].centroid[1]<<endl;*/

				// Update the associated Precision Tracker

				p1.x = centroid[0]; // Centroid of the current object
				p1.y = centroid[1];
				p1.z = centroid[2];
				p2.x = centroid[0] + (estimated_velocity[0]*10); // Predicted centroid of this object based on the estimated_velocity
				p2.y = centroid[1] + (estimated_velocity[1]*10);
				p2.z = centroid[2] + (estimated_velocity[2]*10);
	
				pTrackers[minKalman].centroid = centroid;
				pTrackers[minKalman].estimated_velocity = estimated_velocity;
				pTrackers[minKalman].time = lidar_msg->header.stamp.toSec();
				pTrackers[minKalman].size[0] = fabs(total_objects[i].x_max - total_objects[i].x_min);
				pTrackers[minKalman].size[1] = fabs(total_objects[i].y_max - total_objects[i].y_min);
				pTrackers[minKalman].size[2] = fabs(total_objects[i].z_max - total_objects[i].z_min);

				if (strcmp(total_objects[i].type.c_str(),"none")) // If the type is not none ...
				{
					pTrackers[minKalman].type = total_objects[i].type;
				}

				associated_filter.push_back(pTrackers[minKalman].id);

				total_objects[i].id = pTrackers[minKalman].id; // Current object ID = Associated precision tracker ID

				total_objects[i].pedestrian_state = pTrackers[minKalman].pedestrian_state; // Associated pedestrian state (-1, 0, 1, 2, 3)

				// Get Kalman Prediction. TODO: Delete? Not using right now

				point_kalman_prediction = getKalmanPrediction(kfs[minKalman]);
			}

			if (KALMAN)
			{
				double r,g,b;
				std::stringstream ss2, ss3;
				r = 255/255.0;
				g = 255/255.0;
				b = 0/255.0;
				ss2 << "Aux Kalman cluster" << i;
				// viewer->addCube(point_kalman_aux.x-point_kalman_aux.w/2,point_kalman_aux.x+point_kalman_aux.w/2, point_kalman_aux.y-point_kalman_aux.h/2,point_kalman_aux.y+point_kalman_aux.h/2, point_kalman_aux.z-point_kalman_aux.d/2, point_kalman_aux.z+point_kalman_aux.d/2,r,g,b,ss2.str());
				r = 255/255.0;
				g = 0/255.0;
				b = 0/255.0;
				ss3 << "Predicted Kalman cluster" << i;
				// viewer->addCube(point_kalman_prediction.x-point_kalman_prediction.w/2,point_kalman_prediction.x+point_kalman_prediction.w/2, point_kalman_prediction.y-point_kalman_prediction.h/2,point_kalman_prediction.y+point_kalman_prediction.h/2, point_kalman_prediction.z-point_kalman_prediction.d/2, point_kalman_prediction.z+point_kalman_prediction.d/2,r,g,b,ss2.str());
			}

			if (DEBUG)
			{
				cout<<"\t.:Kalman cluster:."<<endl;
				cout << "Cluster " << i << endl << " Minimum distance: "<< min_distance << endl << " Kalman filter index: "<< minKalman << endl;
				cout << "Cluster x" << point_kalman_aux.x << " Kalman x " << point_kalman_prediction.x << " Cluster y " << point_kalman_aux.y << " Kalman y " << point_kalman_prediction.y << " Cluster z " << point_kalman_aux.z << " Kalman z " << point_kalman_prediction.z << endl;
				cout << "Cluster w" << point_kalman_aux.w << " Kalman w " << point_kalman_prediction.w << " Cluster h " << point_kalman_aux.h << " Kalman h " << point_kalman_prediction.h << " Cluster d " << point_kalman_aux.d << " Kalman d " << point_kalman_prediction.d << endl;
			}
		}
	}
	/*else // If none obstacle is detected
	{
		pedestrian_crossing_occupied = 0;
		merging_occupied = 0;
		
	}*/

	// End Obstacles merging and filter updates //

	// Decrease Kalman filter life

	for (unsigned int i=0; i<kfsTime.size(); i++)
	{
		kfsTime[i]--; // When a Kalman filter is initialized, this value is set to 20
	}

	// Delete precision tracker if it is not update in the last TIME_PRECISION_TRACKING s (see parameter)

	for (unsigned int j=0; j<pTrackers.size(); j++)
	{
		if ((lidar_msg->header.stamp.toSec()-pTrackers[j].time) > TIME_PRECISION_TRACKING)
		{
			kfsTime.erase(kfsTime.begin()+j); // Delete that element of the array
			kfs.erase(kfs.begin()+j);
			pTrackers.erase(pTrackers.begin()+j);
		}
	}

	// TODO: Store the lanelet for each obstacle ?

	// Monitors //

	Obstacles.obstacles.clear();

	// Auxiliar variables for obstacles

	geometry_msgs::PointStamped point_local, point_global, v1, v2, v3, v4, v1_global, v2_global, v3_global, v4_global; // v-i = Vertice of the BEV bounding box
	geometry_msgs::Point32 point32_global;

	pcl::PointXYZ pointaux;
	
	point_local.header.frame_id = "/base_link";
	point_local.header.stamp = lidar_msg->header.stamp;

	// Auxiliar variables for monitors

	bool pedestrian_detection = false;
	double distance_to_front_car = 5000000;
        std_msgs::Float64 distance_To_Front_Car;
	sec_msgs::Obstacle front_car; 
	string current_type = "none";
	double distance_overtake = 0;

	// Object evaluation For each detected cluster, regardingless if the obstacle is in the current lanelet //

	for (unsigned int i=0; i<total_objects.size(); i++)
	{
		total_objects[i].w = total_objects[i].x_max - total_objects[i].x_min;
		total_objects[i].h = total_objects[i].y_max - total_objects[i].y_min;
		total_objects[i].d = total_objects[i].z_max - total_objects[i].z_min;

		point_local.point.x = total_objects[i].centroid_x;
		point_local.point.y = total_objects[i].centroid_y;
		point_local.point.z = total_objects[i].centroid_z;

		// BEV (Bird's Eye View) of Cluster

		v1 = point_local;
		v1.point.x = total_objects[i].centroid_x + (total_objects[i].w/2);
		v1.point.y = total_objects[i].centroid_y - (total_objects[i].h/2);

		v2 = point_local;
		v2.point.x = total_objects[i].centroid_x + (total_objects[i].w/2);
		v2.point.y = total_objects[i].centroid_y + (total_objects[i].h/2);

		v3 = point_local;
		v3.point.x = total_objects[i].centroid_x - (total_objects[i].w/2);
		v3.point.y = total_objects[i].centroid_y + (total_objects[i].h/2);

		v4 = point_local;
		v4.point.x = total_objects[i].centroid_x - (total_objects[i].w/2);
		v4.point.y = total_objects[i].centroid_y - (total_objects[i].h/2);

		// Transform Local to Global points. TODO: Transform local vertices to global vertices?

		point32_global = Local_To_Global_Coordinates(point_local);

		Area_Point area_point; // Cluster BEV centroid to evaluate
		area_point.x = point32_global.x;
		area_point.y = point32_global.y;

		// Search in monitorized lanelets and store relevant obstacles //

		vector<double> lanelets_id; // Auxuliar vector to avoid repeating the analysis of the obstacles in two lanelets with the same ID (for example, if its type is route or lanelet, 
		// but actually the lanelet is the same)

		for (int k=0; k<monitorized_lanelets.route.size(); k++)
		{
			lanelets_id.push_back(0);
		}

		bool flag_monitorized = true;

		for (int j=0; j<monitorized_lanelets.route.size(); j++)
		{
			sec_msgs::Lanelet lanelet = monitorized_lanelets.route[j];

			lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id); // lane is a pointer to the lanelet, instead of the lanelet

			for (int t=0; t<lanelets_id.size(); t++)
			{
				if (lanelets_id[t] == lanelet.id)
				{
					flag_monitorized = false;
				}
			}

			if ((isInsideLanelet(lane, area_point.x, area_point.y, utm_origin) || pedestrian_crossing_monitor) && flag_monitorized == true) // We evaluate the presence of objects if
			// they are inside the lanelet or if there is a pedestrian crossing close, so the pedestrian crossing area must be evaluated
			{
				if (DEBUG)
				{
					ROS_ERROR("Obstacle %d is inside Lanelet %d", i, j);
				}

				lanelets_id[j] = lanelet.id;

				// Store obstacle since is relevant to monitors/vehicle
	
				Obstacle_in_Lanelet(ObstaclesInLanelet_Ptr, point_local, point32_global, v1, v2, v3, v4, lidar_msg->header.stamp, lanelet);

				// Search the associated tracker and store the relative velocity

				for (int k = 0; k<pTrackers.size(); k++)
				{
					if (pTrackers[k].id == associated_filter[i]) // The index of associated_filter is the same that total_objects index
					{
						current_obstacle.twist.linear.x = pTrackers[k].estimated_velocity[0] - vel_x; // Relative velocity
						current_obstacle.twist.linear.y = pTrackers[k].estimated_velocity[1] - vel_y;
						// current_obstacle.twist.linear.x = pTrackers[k].estimated_velocity[0]; // Absolute velocity
						// current_obstacle.twist.linear.x = pTrackers[k].estimated_velocity[0]; 

						current_obstacle.type = pTrackers[k].type;
						current_type = current_obstacle.type;

						if (!strcmp(current_regulatory_element.type.c_str(),"pedestrian_crossing"))
						{
							Inside_Polygon(polygon_area,Number_of_sides,area_point,pedestrian_detection); // pedestrian_detection = 0 (Safety area occupied)
															              // pedestrian_detection = 1 (Safety area occupied)
						}
					}
				}
			}
		}

		// End Search in monitorized lanelets and store relevant obstacles //
		
		// Lanelets variables

		Point point_reg_left, point_reg_right;
		int node_reg_left, node_reg_right;
		geodesy::UTMPoint reg_point_UTM_msg;
		vector<point_with_id_t> wayL, wayR;

		sec_msgs::Route lanelets;
		sec_msgs::Lanelet lanelet_object;

		visualization_msgs::Marker line_list, point_list;
		
		string left_border, right_border, role;

		// Init markers

		point_list = init_points("map","map_manager_visualization",0,0,0.0,0.0,1.0f,1.0,ros::Time::now(),0.2);
		line_list = init_points("map","map_manager_visualization",1,1,0.0,1.0f,0.0,1.0,ros::Time::now(),0.2);

		// Evaluate the monitors //

		// Pedestrian monitor
		// Stop if a pedestrian is inside the safety area associated to a pedestrian crossing regulatory element
		// TODO: Use velocities to estimate the direction of the pedestrian instead of the previous position
		// TODO: Improve to work with several pedestrians in the same lanelet. Use associated filter to identify the pedestrian (Add a new field in the structure?)

		if (pedestrian_crossing_monitor)
		{
			if (!strcmp(current_type.c_str(),"pedestrian"))
			{
				//cout<<"\n\nPedestrian crossing lanelets: "<<pedestrian_crossing_lanelets.route.size()<<endl;

				for (int j=0; j<pedestrian_crossing_lanelets.route.size(); j++)
				{
					sec_msgs::Lanelet lanelet = pedestrian_crossing_lanelets.route[j];

					lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id);

					cout<<"\nPedestrian detection: "<<pedestrian_detection<<endl;

					if (pedestrian_detection) // Pedestrian detected in the safety area
					{
						int pedestrian_crossing_occupied = total_objects[i].pedestrian_state;

						pointaux.x = point32_global.x;
						pointaux.y = point32_global.y;
						pointaux.z = point32_global.z;
						ObstaclesInPedestrian_Ptr->points.push_back(pointaux);

						//cout<<"\nPedestrian_crossing_occupied 1: "<<pedestrian_crossing_occupied<<endl;

						if (pedestrian_crossing_occupied == 0) // Pedestrian crossing was unoccupied
						{	
							pedestrian_crossing_occupied = 3; 
						}

						if (isInsideLanelet(lane,point32_global.x, point32_global.y,utm_origin)) // Pedestrian detected in the current lanelet or adyacent lanelet
						{
							cout<<"\nPedestrian inside some pedestrian lanelet!"<<endl;
							//cout<<"Current lanelet: "<<lanelet.id<<" Pedestrian crossing lanelet: "<<id_lanelet_pedestrian_crossing<<endl;
							if (lanelet.id == id_lanelet_pedestrian_crossing) // If current lanelet, maximum relevance (1)
							{
								pedestrian_crossing_occupied = 1; 
							}
							else // If pedestrian is not in current lanelet (Left or right from the current one)
							{
								// If pedestrian comes from current lanelet, not relevant (is going away from ego-vehicle)
								if (pedestrian_crossing_occupied == 1 || pedestrian_crossing_occupied == -1)
								{
									pedestrian_crossing_occupied = -1; 
								}
								else // If pedestrian has not been detected in current lanelet, the pedestrian is relevant (it is going to the ego-vehicle lanelet)
								{
									pedestrian_crossing_occupied = 2;
								}
							}
						}

						// If the map manager does not calculate adyacent pedestrian crossing lanelets (due to intersections, etc.)
						// TODO: Improve this casuistic

						//cout<<"\nPedestrian_crossing_occupied 2: "<<pedestrian_crossing_occupied<<endl;

						else if (pedestrian_crossing_occupied == 1 || pedestrian_crossing_occupied == -1) // Pedestrian not detected in a pedestrian crossing lanelet, but it was relevant
						{
							pedestrian_crossing_occupied = -1; 
						}

						//cout<<"\nPedestrian_crossing_occupied 2: "<<pedestrian_crossing_occupied<<endl;

						for (int k = 0; k<pTrackers.size(); k++)
						{	
							if (pTrackers[k].id == total_objects[i].id)
							{
								pTrackers[k].pedestrian_state = pedestrian_crossing_occupied;
							}
						}		
					}
				}
			}
		}

		// Merging monitor (Give way and STOP regulatory elements)
		// Stop if a merging lanelet associated with the regulatory element is occupied
		// TODO: Improve with velocities (If a car is stopped in the merging lanelet it must not be relevant for the monitor)

		if (merging_monitor)
		{
			if (!strcmp(current_type.c_str(),"car"))
			{
				cout<<"\n\nCar and Merging monitor!"<<endl; 
				for (int j=0; j<merging_lanelets.route.size(); j++)
				{
					cout<<"Merging "<<j<<" id "<<merging_lanelets.route[j].id<<endl;
					sec_msgs::Lanelet lanelet = merging_lanelets.route[j];
				
					lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id);

					if (isInsideLanelet(lane,point32_global.x, point32_global.y,utm_origin))
					{
						pointaux.x = point32_global.x;
						pointaux.y = point32_global.y;
						pointaux.z = point32_global.z;
						ObstaclesMerging_Ptr->points.push_back(pointaux);
						merging_occupied = 1;
							
						cout<<"\nEstoy detectando algo"<<endl;
					}
				}
			}
		}

		// ACC monitor
		// TODO: Only store the closest obstacle if it is in front of the ego-vehicle (avoid storing obstacles behing the ego-vehicle)

		if (!strcmp(current_type.c_str(),"car"))
		{
			for (int j=0; j<route_lanelets.route.size(); j++)
			{
				sec_msgs::Lanelet lanelet = route_lanelets.route[j];
				
				lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id);

				float distance = sqrt(pow(point_local.point.x,2)+pow(point_local.point.y,2));

				if ((point_local.point.x > 0) && (distance < distance_to_front_car) && (isInsideLanelet(lane,point32_global.x, point32_global.y,utm_origin)))
				{
					front_car = current_obstacle;
					distance_to_front_car = distance;
				}
			}
		}

		// Overtaking monitor
		// If there is an obstacle in left lanelets, lane change is not possible
		// TODO: Improve monitor with velocities
		// Right now the system is working on a two-lanes map

		for (int j=0; j<left_lanelets.route.size(); j++)
		{
			sec_msgs::Lanelet lanelet = route_lanelets.route[j];
			
			lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id);

			if (isInsideLanelet(lane,point32_global.x, point32_global.y,utm_origin))
			{
				lane_change.data = false;
			}
		}

		// Calculate possible overtake distance
		// TODO: Improve by calculating the possible overtake distance when the car is in the overtaking route
		if (route_lanelets.route.size() > 0)
		{
			// If there is a possible left lanelet in current position of vehicle

			sec_msgs::Lanelet first_lanelet = route_lanelets.route[0];
			lanelet_ptr_t first_lane = loadedMap->lanelet_by_id(first_lanelet.id);

			lanelet_ptr_t lanelet_left_ptr; 
			int left = findLaneletLeft(loadedMap,first_lane,&lanelet_left_ptr,1);

			if (left == 1) // If the left lanelet is of the same direction of the current one
			{
				auto wayL = lanelet_left_ptr->nodes(LEFT);
				auto wayR = lanelet_left_ptr->nodes(RIGHT);

				distance_overtake = DistanceLanelet(wayL,wayR,odomUTMmsg,odomUTMmsg,1,utm_origin); // The 4th argument is not used right now

				vector<lanelet_ptr_t> lanelet_next_ptr;

				int next = findLaneletNext(loadedMap,lanelet_left_ptr,&lanelet_next_ptr);

				while(next)
				{
					next = 0;
					lanelet_ptr_t  lanelet_A;

					for (int k=0; k<lanelet_next_ptr.size(); k++)
					{
						lanelet_ptr_t lanelet_right_ptr;
						int right = findLaneletRight(loadedMap,lanelet_next_ptr[k],&lanelet_right_ptr,1);

						if (right)
						{
							if (isInPath(route,lanelet_right_ptr->id()))
							{
								next = 1;
								lanelet_A = lanelet_next_ptr[k];
								wayL = lanelet_next_ptr[k]->nodes(LEFT);
								wayR = lanelet_next_ptr[k]->nodes(RIGHT);
								distance_overtake = distance_overtake + DistanceLanelet(wayL,wayR,odomUTMmsg,odomUTMmsg,0,utm_origin);
								break;
							}
						}
					}
					if (next == 1)
					{
						next = findLaneletNext(loadedMap,lanelet_A,&lanelet_next_ptr);
					}
				}
			}
			else 
			{
				if (left == 2) // If the left lanelet is of the opposite direction of the current one
				{
					auto wayR = lanelet_left_ptr->nodes(RIGHT);
					auto wayL = lanelet_left_ptr->nodes(LEFT);

					distance_overtake = DistanceLanelet (wayL, wayR, odomUTMmsg, odomUTMmsg, 2, utm_origin);

					vector<lanelet_ptr_t> lanelet_next_ptr;

					int next = findLaneletNext(loadedMap,lanelet_left_ptr,&lanelet_next_ptr);
				
					while (next)
					{
						next = 0;
						lanelet_ptr_t lanelet_A;

						for (int k=0; k<lanelet_next_ptr.size(); k++)
						{
							lanelet_ptr_t lanelet_right_ptr;
							//int right = findLaneletRight(loadedMap,lanelet_next_ptr[k],&lanelet_right_ptr,1);
							int right = findLaneletLeft(loadedMap,lanelet_next_ptr[k],&lanelet_right_ptr,1);

							if (right)
							{
								if (isInPath(route,lanelet_right_ptr->id()))
								{
									next = 1;
									lanelet_A = lanelet_next_ptr[k];
									wayL = lanelet_next_ptr[k]->nodes(LEFT);
									wayR = lanelet_next_ptr[k]->nodes(RIGHT);
									distance_overtake = distance_overtake + DistanceLanelet(wayL,wayR,odomUTMmsg,odomUTMmsg,0,utm_origin);
									break;
								}
							}
						}
						if (next == 1)
						{
							next = findLaneletPrevious(loadedMap,lanelet_A,&lanelet_next_ptr);
						}
					}

				}
			}	
		}

		// End Evaluate the monitors //
	}

	// End Object evaluation For each detected cluster, regardingless if the obstacle is in the current lanelet //

	// Publish the monitors //

	// Pedestrian monitor

	// Evaluate the state of all pedestrian and obtain a global response

	global_pedestrian_crossing_occupied = 0;

	for (int i=0; i<total_objects.size(); i++) // If at least one pedestrian is about to cross ...
	{
		if (!strcmp(total_objects[i].type.c_str(),"pedestrian") && (total_objects[i].pedestrian_state == 1 || total_objects[i].pedestrian_state == 2 || total_objects[i].pedestrian_state == 3))
		{
			global_pedestrian_crossing_occupied = 1;
		}
	}

	if (pedestrian_crossing_monitor)
	{
		cout<<"\nGlobal_pedestrian_crossing_occupied: "<<global_pedestrian_crossing_occupied<<endl;
		if (global_pedestrian_crossing_occupied == 1)
		{
			std_msgs::Bool is_pedestrian;
			is_pedestrian.data = true;
			pub_Detected_Pedestrian.publish(is_pedestrian);
		}
		else
		{
			std_msgs::Bool is_pedestrian;
			is_pedestrian.data = false;
			pub_Detected_Pedestrian.publish(is_pedestrian);
		}
	}

	// Merging monitor
	// stop flag: 0 = Inactive, 1 Active (Car cannot cross the STOP), 2 Merging monitor (Car can cross the stop if merging monitor allows)
	//cout<<"Merging monitor: "<<merging_monitor<<endl;
	//cout<<"\nMerging occupied: "<<merging_occupied<<endl;

	if (merging_monitor)
	{
		cout<<"\nStop flag: "<<stop<<endl;
		std_msgs::Bool is_safe;
		int aux = 0;
	
		if (stop == 1 && abs_vel>0.1) // STOP behaviour is active, the car must stop
		{
			is_safe.data = false;
			aux = 0;
			pub_Safe_Merge.publish(is_safe);
		}
		/*else if (stop == 1 && abs_vel<0.1) // In those situations in which the car is stopped but the RoboGraph dispatch does not send stop == 2 (The car is completely stopped, so ti can continue)
		{
			stop = 2;
		}*/
		else
		{
			if (merging_occupied)
			{
				is_safe.data = false;
				aux = 0;
				pub_Safe_Merge.publish(is_safe);
			}
			else
			{
				is_safe.data = true;
				aux = 1;
				pub_Safe_Merge.publish(is_safe);
			}
		} 
		
		cout<<"Is safe: "<<aux<<endl;
	}

	// ACC monitor

	pub_Front_Car.publish(front_car);

	if (number_cars == 0)
	{
		distance_to_front_car = 0;
	}

	distance_To_Front_Car.data = distance_to_front_car;
	pub_Front_Car_Distance.publish(distance_To_Front_Car);

	// Overtaking monitor

	pub_Safe_Lane_Change.publish(lane_change);

	sec_msgs::Distance distance_overtake_monitor;
	distance_overtake_monitor.distance = distance_overtake;
	distance_overtake_monitor.header.frame_id = "/base_link";
	distance_overtake_monitor.header.stamp = ros::Time::now();
	pub_Distance_Overtake.publish(distance_overtake_monitor);

	// End Publish the monitors //

	// End Monitors //

	// Publish relevant obstacles and velocity visual markers //

	// Publish LiDAR obstacles

	Obstacles.header.frame_id = "/map"; // Global coordinates
	Obstacles.header.stamp = lidar_msg->header.stamp;
	
	pub_LiDAR_Obstacles.publish(Obstacles);

	// Publish velocity visual markers for tracked obstacles (Precision Tracking)

	visualization_msgs::Marker points;
	geometry_msgs::Point p;

	points.header.frame_id = "/base_link"; // map == global coordinates. Base_link == local coordinates. Since the Precision Tracking centroid is in local coordinates, 
	// the frame_id must be "/base_link" 
	points.header.stamp = lidar_msg->header.stamp;
	points.ns = "map_manager_visualization";
	points.action = visualization_msgs::Marker::ADD;
	points.pose.orientation.w = 1.0;

	points.id = 0;
	points.type = visualization_msgs::Marker::SPHERE;
	 
	points.scale.x = 0.25;
	points.scale.y = 0.25;
	points.scale.z = 0.25;
	 
	points.color.r = 1.0f;	     
	points.color.g = 1.0f;
	points.color.b = 0.0f;
	points.color.a = 1.0;

	points.points.clear();

	for (unsigned int i=0; i<pTrackers.size(); i++)
	{
		if (!strcmp(pTrackers[i].type.c_str(),"pedestrian") || !strcmp(pTrackers[i].type.c_str(),"car"))
		{
			points.type = visualization_msgs::Marker::ARROW;
			points.id = pTrackers[i].id;
			points.pose.position.x = pTrackers[i].centroid[0];
			points.pose.position.y = pTrackers[i].centroid[1];
			//points.pose.position.z = pTrackers[i].centroid[2];
			points.pose.position.z = 0; // Projected onto BEV perspective

			points.scale.x = sqrt(pow(pTrackers[i].estimated_velocity[0],2)+pow(pTrackers[i].estimated_velocity[1],2));
			points.scale.y = 0.15;
			points.scale.z = 0.15;

			points.lifetime = ros::Duration(0.05);
			points.color = colours[3];

			double yaw;

			if (pTrackers[i].estimated_velocity[0] == 0 && pTrackers[i].estimated_velocity[1] < 1) // The object is stopped w.r.t. the ego-vehicle current position
			{
				yaw = 0;
			}
			else
			{
				if (pTrackers[i].estimated_velocity[0] != 0) // X velocity
				{
					yaw = atan2(pTrackers[i].estimated_velocity[1],pTrackers[i].estimated_velocity[0]);
				}
				else
				{
					if (pTrackers[i].estimated_velocity[1]>0)
					{
						yaw = 0;
					}
					else
					{
						yaw = M_PI;
					}
				}

				points.pose.orientation=tf::createQuaternionMsgFromRollPitchYaw(0,0,yaw);
				pub_LiDAR_Obstacles_Velocity_Marker.publish(points);
			}

		}
	}

	//total_objects.clear(); // Clear all detected obstacles with the segmentation filter
	
	// End Publish relevant obstacles and velocity visual markers // 

	// End Monitors //	
}

// End Callbacks //

// End Definitions of functions //












