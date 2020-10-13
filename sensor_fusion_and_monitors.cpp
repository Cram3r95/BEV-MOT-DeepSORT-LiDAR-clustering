/***
Created on Thu Dec  8 16:45:21 2019

@author: Carlos Gomez-Huelamo

Code to process the fusion between LiDAR clusters based on PCL algorithms and BEV (Bird's Eye View) 
Object Tracking using the Global Nearest Neighbour (GNN) approach and evaluate the objects according 
to some specified behaviours (ACC, Give Way, STOP, etc.)

Inputs:  BEV Object Tracking, LiDAR pointclud and Monitorized Lanes
Outputs: Evaluated behaviours

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

// BEV Tracking includes
#include ".h"
#include ".h"

// SEC (SmartElderlyCar) includes
#include <sec_msgs/Route.h>
#include <sec_msgs/Lanelet.h>
#include <sec_msgs/RegElem.h>
#include <sec_msgs/Distance.h>
#include <sec_msgs/CarControl.h>
#include <sec_msgs/ObstacleArray.h>
#include "map_manager_base.hpp"

// End Includes //


// Defines //

#define lanelet_filter 1

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

// End Defines //


// Structures //

typedef struct
{
	double x; // x UTM with respect to the map origin
	double y; // y UTM with respect to the map origin
}Area_Point;

typedef struct
{
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
    double a;
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	string type;
	int id;
    double time;
}Object;

typedef struct 
{
    float w;
    float h;
    float d;
	float global_centroid_x; // Global coordinates (w.r.t. "map" frame)
	float global_centroid_y;
    float local_centroid_x; // Local coordinates (w.r.t. "base_link" frame)
    float local_centroid_y; 
	int object_id; // VOT (Visual Object Tracking) assigned ID
    string type;
	double time;
	double yaw;
    int pedestrian_state;
    int stop_state;
    int give_way_state;
}Merged_Object;

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
vector<Merged_Object> merged_objects;
Area_Point polygon_area[] = {0,0,
                             0,0,
                             0,0,
                             0,0};

int Number_of_sides = 4; // Of the area you have defined. Here is a rectangle, so area[] has four rows

// End Global variables


// Declarations of functions // 

// General use functions

geometry_msgs::Point32 Local_To_Global_Coordinates(geometry_msgs::PointStamped );
float get_Centroids_Distance(pcl::PointXYZ , pcl::PointXYZ );
void Inside_Polygon(Area_Point *, int , Area_Point, bool &);
void Obstacle_in_Lanelet(pcl::PointCloud<pcl::PointXYZRGB>::Ptr , geometry_msgs::PointStamped , geometry_msgs::Point32 , geometry_msgs::PointStamped , geometry_msgs::PointStamped , geometry_msgs::PointStamped , geometry_msgs::PointStamped , ros::Time , sec_msgs::Lanelet );

// Point Cloud filters

pcl::PointCloud<pcl::PointXYZRGB> xyz_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr );
pcl::PointCloud<pcl::PointXYZRGB> angle_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr );
void cluster_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr , float , int , int , vector<Object> *, int *);
vector<Merged_Object> merging_z(vector<Object> );

// Cluster functions

vector<Merged_Object> merging_z(vector<Object> );

// Calbacks

void route_cb(const sec_msgs::Route::ConstPtr& );
void waiting_cb(const std_msgs::Empty );
void regelement_cb(const sec_msgs::RegElem::ConstPtr& , const sec_msgs::Route::ConstPtr& );
void sensor_fusion_and_monitors_cb(const sensor_msgs::PointCloud2::ConstPtr& , const t4ac_msgs::BEV_trackers_list::ConstPtr& , const nav_msgs::Odometry::ConstPtr& );

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
	message_filters::Subscriber<nav_msgs::Odometry> ego_vehicle_pose_sub_; // Odometry
	message_filters::Subscriber<t4ac_msgs::BEV_trackers_list> projected_vot_; // Detection and Tracking with camera (CenterNet + Deep Sort)

	regelem_sub_.subscribe(nh, "/currentRegElem", 1);
	regelemLanelet_sub_.subscribe(nh, "/monitorizedLanelets", 1);
	cloud_sub_.subscribe(nh, "/velodyne_coloured", 1); // Colored point cloud (based on semantic segmentation)
	velodyne_cloud_sub_.subscribe(nh, "/velodyne_points", 1);
    ego_vehicle_pose_sub_.subscribe(nh, "t4ac/localization/pose", 1)
	projected_vot_.subscribe(nh, "/t4ac/perception/tracked_obstacles_list", 1);

    waiting_sub = nh.subscribe<std_msgs::Empty>("/waitingAtStop", 1, &waiting_cb);
	route_sub = nh.subscribe<sec_msgs::Route>("/route", 1, &route_cb);

    // End Subscribers //

	// Callbacks //

	// Callback 1: Synchonize monitorized lanelets and current regulatory element (Exact time)

	typedef message_filters::sync_policies::ExactTime<sec_msgs::RegElem, sec_msgs::Route> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync_(MySyncPolicy(10), regelem_sub_, regelemLanelet_sub_);
	sync_.registerCallback(boost::bind(&regelement_cb, _1, _2));

	// Callback 2: Synchronize LiDAR point cloud and camera information (including detection and tracking). Evaluate monitors (Approximate time)

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, t4ac_msgs::BEV_trackers_list, nav_msgs::Odometry> MySyncPolicy2;
	message_filters::Synchronizer<MySyncPolicy2> sync2_(MySyncPolicy2(200), velodyne_cloud_sub_, vision_sub_, ego_vehicle_pose_sub_);
	sync2_.registerCallback(boost::bind(&sensor_fusion_and_monitors_cb, _1, _2, _3));

	// Load map

	string map_frame = "";
	string map_path = ros::package::getPath("sec_map_manager") + "/maps/uah_lanelets_v42.osm";

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
void cluster_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud, float tolerance, int min_cluster, int max_cluster, vector<Object> *output_objects, int *number_output_objects)
{
	// Parameters:
	// filtered_cloud: XYZ and angle filtered LiDAR point cloud that contains the clusters
	// tolerance: Tolerance of clusters
	// min_cluster: Minimum size of a cluster
	// max_cluster: Maximum size of a cluster
	// only_laser_objects: Pointer that points to the array that contains the clusters
	// only_laser_objects_number: Number of objects

	// This function only takes into account the size of the clusters

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

		centroid_x = (x_max+x_min)/2.0;
		centroid_y = (y_max+y_min)/2.0;
		centroid_z = (z_max+z_min)/2.0;

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*cloud_cluster,centroid);

		geometry_msgs::PointStamped local_centroid;
		geometry_msgs::Point32 global_centroid;

		local_centroid.point.x = centroid_x;
		local_centroid.point.y = centroid_y;
		local_centroid.point.z = centroid_z;

		global_centroid = Local_To_Global_Coordinates(local_centroid);

        // Lanelet filter //

        if (lanelet_filter)
        {
            // Lanelets filter // Only use if you are working in real mode or the rosbag has /monitorizedLanelets according to the current map 
			// (since if you modify some node, all ID change in the map)

            for (int j=0; j<monitorized_Lanelets.route.size(); j++)
			{
				sec_msgs::Lanelet lanelet = monitorized_Lanelets.route[j];
				lanelet_ptr_t lane = loadedMap -> lanelet_by_id(lanelet.id);

                if (isInsideLanelet(lane, global_centroid.x, global_centroid.y, utmOrigin))
				{
					// Object measurements

					object.x_max = x_max;
					object.x_min = x_min;
					object.y_max = y_max;
					object.y_min = y_min;
					object.z_max = z_max;
					object.z_min = z_min;

                    object.d = x_max - x_min;
                    object.w = y_max -y_min;
                    object.h = z_max - z_min;

                    object.centroid_x = centroid2[0];
					object.centroid_y = centroid2[1];
					object.centroid_z = centroid2[2];

                    // Global coordinates

					object.centroid_global_x = global_centroid.x;
					object.centroid_global_y = global_centroid.y;
					object.centroid_global_z = global_centroid.z;

                    // Type

					object.type = "none";

					// Cloud

					object.cloud = cloud_cluster;

					output_objects->push_back(object);
					*number_output_objects = *number_output_objects + 1;

					break; // Continue with the next object
                }
            }
        }
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

void sensor_fusion_and_monitors_cb(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const t4ac_msgs::BEV_trackers_list::ConstPtr& bev_trackers_list_msg, const nav_msgs::Odometry::ConstPtr& ego_vehicle_pose_msg);
{
	cout<<"------------------------------------------------"<<endl;
	// ROS_INFO("Time: [%lf]", (double)ros::Time::now().toSec());

    // lidar_msg contains the LiDAR  information = PointCloud
	// bev_trackers_list_msg contains the visual tracked obstacles projected onto the Bird's Eye View space
	// ego_vehicle_pose_msg contains the ego vehicle position

	// Note that if --clock is not published (if we are trying to run a rosbag), the system will not create the transforms

    // Obtain transforms between frames and store

    // Auxiliar variables

    // ACC variables

	double distfrontcar = 5000000;
	std_msgs::Float64 front_car_distance;

    try
	{
		listener->lookupTransform("map", "base_link", lidar_msg->header.stamp, TF_map2base_link);
	}
	catch(tf::TransformException& e)
	{
		cout<<e.what();
		return; 
	}

    // Filter PointCloud //

    // Auxiliar Point Clouds

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr vlp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	sensor_msgs::PointCloud2 msga;

    // If full cloud is used ...

	pcl::fromROSMsg(*lidar_msg, *vlp_cloud); // The content pointed by non_filtered_cloud will contain the content of lidar_msg

    // XYZ filter

    pcl::PointCloud<pcl::PointXYZRGB> aux_xyz = xyz_filter(vlp_cloud);
    *filtered_cloud = aux;

    // Angle filter

    pcl::PointCloud<pcl::PointXYZRGB> aux_angle = angle_filter(filtered_cloud);
    *filtered_cloud = aux_angle;

    // End Filter PointCloud //

    // LiDAR Clustering //

    vector<Object> vehicles, pedestrians, only_laser_objects;
    int number_vehicles = 0, number_pedestrians = 0, number_only_laser_objects = 0;

    cluster_filter(filtered_cloud,2,5,2000,&vehicles,number_vehicles,"vehicles");
    cluster_filter(filtered_cloud,2,5,200,&pedestrians,number_pedestrians,"pedestrians");

    for (int i=0; i<cars.size(); i++)
	{
		only_laser_objects.push_back(vehicles[i]);
	}

	for (int i=0; i<pedestrians.size(); i++)
	{
		only_laser_objects.push_back(pedestrians[i]);
	}

    cout<<"\nNumber of vehicles: "<<number_vehicles;
	cout<<"\nNumber of pedestrians: "<<number_pedestrians;
	cout<<"\nTotal objects: "<<merged_objects.size()<<endl<<endl;

    // End LiDAR Clustering //

    // BEV Projected VOT (Visual Object Tracking)
    
    float diff_lidar_vot = 0;
    int object_id = 0;

    for (int i=0; i<bev_trackers_list_msg->tracked_obstacles_list.size(); i++)
    {
        float max_diff_lidar_vot = 4; // Initialize maximum allowed difference
        int index_most_similar = -1;

        float vot_x = float(bev_trackers_list_msg->tracked_obstacles_list[i].x);
        float vot_y = float(bev_trackers_list_msg->tracked_obstacles_list[i].y);

        geometry_msgs::PointStamped local_centroid;
        geometry_msgs::Point32 global_centroid;

        local_centroid.point.x = vot_x;
        local_centroid.point.y = vot_y;
        local_centroid.point.z = 0;

        global_centroid = Local_To_Global_Coordinates(local_centroid);

        if (only_laser_objects.size() > 0 && (!strcmp(bev_trackers_list_msg->tracked_obstacles_list[i].type.c_str(),"car") || !strcmp(bev_trackers_list_msg->tracked_obstacles_list[i].type.c_str(),"person")))
        {
            double time = bev_trackers_list_msg->header.stamp.toSec();
            object_id = bev_trackers_list_msg->tracked_obstacles_list[i].object_id;

            geometry_msgs::PointStamped local_centroid;
			geometry_msgs::Point32 global_centroid;

			local_centroid.point.x = vot_x;
			local_centroid.point.y = vot_y;
			local_centroid.point.z = 0;

			global_centroid = Local_To_Global_Coordinates(local_centroid);

            for (int j=0; j<only_laser_objects.size(); j++)
            {
                float l_x = float(only_laser_objects[j].centroid_x); 
				float l_y = float(only_laser_objects[j].centroid_y);

                diff_lidar_vot = float(sqrt(pow(vot_x-l_x,2)+pow(vot_y-l_y,2))); 

                if (diff_lidar_vot < max_diff_lidar_vot) // Find the closest cluster
				{
					max_diff_lidar_vot = diff_lidar_vot;
					index_most_similar = j;
				}
            }

            if (max_diff_lidar_vot < 1.5 && index_most_similar != -1)
            // In order to merge both information, the centroid between distance must be less that 1.5 m (VOT projected centroid and closest LiDAR centroid)
            {
                only_laser_objects[index_most_similar].type = bev_trackers_list_msg->tracked_obstacles_list[i].type;
                only_laser_objects[index_most_similar].object_id = bev_trackers_list_msg->tracked_obstacles_list[i].object_id;
				only_laser_objects[index_most_similar].r = bev_trackers_list_msg->tracked_obstacles_list[i].color.r
				only_laser_objects[index_most_similar].g = bev_trackers_list_msg->tracked_obstacles_list[i].color.g
				only_laser_objects[index_most_similar].b = bev_trackers_list_msg->tracked_obstacles_list[i].color.b
				only_laser_objects[index_most_similar].a = bev_trackers_list_msg->tracked_obstacles_list[i].color.a

                int flag = 0;

                // 1. Find out if current merged object was previously stored. Id does, update the object

                for (int k=0; k<merged_objects.size(); k++)
                {
                    if (merged_objects[k].object_id == object_id)
                    {
                        merged_objects[k].global_centroid_x = only_laser_objects[index_most_similar].centroid_global_x;
                        merged_objects[k].global_centroid_y = only_laser_objects[index_most_similar].centroid_global_y;
                        merged_objects[k].local_centroid_x = only_laser_objects[index_most_similar].centroid_x;
                        merged_objects[k].local_centroid_y = only_laser_objects[index_most_similar].centroid_y;

                        merged_object.d = only_laser_objects[index_most_similar].d;
                        merged_object.w = only_laser_objects[index_most_similar].w;
                        merged_object.h = only_laser_objects[index_most_similar].h;

                        flag = 1;
                        break;
                    }
                }

                // 2. If it was not previously stored, then create a new object
	
				if (flag == 0)
				{
                    Merged_Object merged_object;

                    merged_object.global_centroid_x = only_laser_objects[index_most_similar].centroid_global_x;
                    merged_object.global_centroid_y = only_laser_objects[index_most_similar].centroid_global_y;
                    merged_object.local_centroid_x = only_laser_objects[index_most_similar].centroid_x;
                    merged_object.local_centroid_x = only_laser_objects[index_most_similar].centroid_y;

                    merged_object.d = only_laser_objects[index_most_similar].d;
                    merged_object.w = only_laser_objects[index_most_similar].w;
                    merged_object.h = only_laser_objects[index_most_similar].h;

                    merged_object.object_id = only_laser_objects[index_most_similar].object_id;
                    merged_object.type = only_laser_objects[index_most_similar].type;

                    merged_objects.push_back(merged_object);
                }
            }
        }
    }

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
	sec_msgs::Obstacle front_car; 
	string current_type = "none";
    std_msgs::Float64 distance_to_front_car;
    double distance_to_front_car = 5000000;
	double distance_overtake = 0;

	// Object evaluation For each detected cluster, regardingless if the obstacle is in the current lanelet //

	for (unsigned int i=0; i<merged_objects.size(); i++)
	{
		point_local.point.x = merged_objects[i].centroid_x;
		point_local.point.y = merged_objects[i].centroid_y;
		point_local.point.z = merged_objects[i].centroid_z;

		// BEV (Bird's Eye View) of Cluster

		v1 = point_local;
		v1.point.x = merged_objects[i].centroid_x + (merged_objects[i].w/2);
		v1.point.y = merged_objects[i].centroid_y - (merged_objects[i].h/2);

		v2 = point_local;
		v2.point.x = merged_objects[i].centroid_x + (merged_objects[i].w/2);
		v2.point.y = merged_objects[i].centroid_y + (merged_objects[i].h/2);

		v3 = point_local;
		v3.point.x = merged_objects[i].centroid_x - (merged_objects[i].w/2);
		v3.point.y = merged_objects[i].centroid_y + (merged_objects[i].h/2);

		v4 = point_local;
		v4.point.x = merged_objects[i].centroid_x - (merged_objects[i].w/2);
		v4.point.y = merged_objects[i].centroid_y - (merged_objects[i].h/2);

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

            // Evaluate the presence of objects in a certain lanelet or if there is a pedestrian crossing close, so the pedestrian crossing area 
            // must be evaluated   
			if ((isInsideLanelet(lane, area_point.x, area_point.y, utm_origin) || pedestrian_crossing_monitor) && flag_monitorized == true)
			{
				lanelets_id[j] = lanelet.id;

				// Store obstacle since is relevant to monitors/vehicle
	
				//Obstacle_in_Lanelet(ObstaclesInLanelet_Ptr, point_local, point32_global, v1, v2, v3, v4, lidar_msg->header.stamp, lanelet);

                if (!strcmp(current_regulatory_element.type.c_str(),"pedestrian_crossing"))
                {
                    Inside_Polygon(polygon_area,Number_of_sides,area_point,pedestrian_detection); 
                    // pedestrian_detection = 0 (Safety area occupied)
                    // pedestrian_detection = 1 (Safety area occupied)
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
		
		string left_border, right_border, role;

		// Evaluate the monitors //

		// Pedestrian monitor
		// Stop if a pedestrian is inside the safety area associated to a pedestrian crossing regulatory element
		// TODO: Use velocities to estimate the direction of the pedestrian instead of the previous position
		// TODO: Improve to work with several pedestrians in the same lanelet. Use associated filter to identify the pedestrian (Add a new field in the structure?)

		if (pedestrian_crossing_monitor)
		{
			if (!strcmp(current_type.c_str(),"pedestrian"))
			{
				for (int j=0; j<pedestrian_crossing_lanelets.route.size(); j++)
				{
					sec_msgs::Lanelet lanelet = pedestrian_crossing_lanelets.route[j];

					lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id);

					cout<<"\nPedestrian detection: "<<pedestrian_detection<<endl;

					if (pedestrian_detection) // Pedestrian detected in the safety area
					{
						int pedestrian_crossing_occupied = merged_objects[i].pedestrian_state;

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
							if (pTrackers[k].id == merged_objects[i].id)
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
				for (int j=0; j<merging_lanelets.route.size(); j++)
				{
					sec_msgs::Lanelet lanelet = merging_lanelets.route[j];
				
					lanelet_ptr_t lane = loadedMap->lanelet_by_id(lanelet.id);

					if (isInsideLanelet(lane,point32_global.x, point32_global.y,utm_origin))
					{
						pointaux.x = point32_global.x;
						pointaux.y = point32_global.y;
						pointaux.z = point32_global.z;
						ObstaclesMerging_Ptr->points.push_back(pointaux);
						merging_occupied = 1;
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

	for (int i=0; i<merged_objects.size(); i++) // If at least one pedestrian is about to cross ...
	{
		if (!strcmp(merged_objects[i].type.c_str(),"pedestrian") && (merged_objects[i].pedestrian_state == 1 || merged_objects[i].pedestrian_state == 2 || merged_objects[i].pedestrian_state == 3))
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

	if (merging_monitor)
	{
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
/*
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

	for (unsigned int i=0; i<merged_objects.size(); i++)
	{
		if (!strcmp(merged_objects[i].type.c_str(),"pedestrian") || !strcmp(pTrackers[i].type.c_str(),"car"))
		{
			points.type = visualization_msgs::Marker::CUBE;
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
*/
	
	// End Publish relevant obstacles and velocity visual markers // 

	// End Monitors //	
}

// End Callbacks //

// End Definitions of functions //











