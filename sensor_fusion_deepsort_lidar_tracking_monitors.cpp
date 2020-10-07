/***

Carlos G�mez Hu�lamo August 2019

Sensor fusion and monitors

ROS topic (/yolov3_tracking_list/) already includes the detected objects with an identification (ID) and label (semantic information: car or person) that shows the tracking process.

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
#define VER_KALMAN 0
#define VER_KALMAN_BAREA 0
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

#define MIN_MONITOR(x,y) (x < y ? x : y)
#define MAX_MONITOR(x,y) (x > y ? x : y)
#define INSIDE 0
#define OUTSIDE 1

#define Number_of_Tracking_Samples 10

// End Defines //

// Structures //

struct Tracking_Points
{
	float global_centroid_x; // Global coordinates (with respect to the map frame)
	float global_centroid_y;
        float local_centroid_x; // With respect to the velodyne
        float local_centroid_y; 
	int object_id; // VOT (Visual Object Tracking) assigned ID
	int carla_id; // CARLA assigned ID
	double time;
	double average_samples = 0; // Initialize yaw angle
	double yaw_previous;
	std::vector<double> yaw_samples;
};

struct precisiontrackers
{
	int id;
	double time;
	Eigen::Vector3f centroid; 
	Eigen::Vector3f estimated_velocity;
	Eigen::Vector3f size;
	string type;
}precisionTrackers;
	
struct Object
{
	float centroid_x;
	float centroid_y;
	float centroid_z;
	float centroid_global_x;
	float centroid_global_y;
	float centroid_global_z;
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
	bool nuevo_kalman;
	int object_id; // In order to track the object
	int carla_id;
	int lanelet_id;
	double time;
}object;

typedef struct Vertice
{
	pcl::PointXYZ vertice_point[8];
}vertice;

typedef struct{
	double x,y;
}Point_camera;

Point_camera zona_MONITOR[]={ // Each line is a point in the poligon
	0, 0, // p1 Este(x), Norte(Y), Zona = 30, Hemisferio = Norte 
	0, 0, // p2
	0, 0, // p3
	0, 0}; // p4

// End structures

// ROS communication //

// ROS Publishers

ros::Publisher localMapPub;
ros::Publisher cloudlidarPub;
ros::Publisher obstaclelidarPub;

ros::Publisher tracked_objects_marker_pub;
ros::Publisher tracked_objects_vel_marker_pub;
ros::Publisher tracked_merged_objects_marker_pub;
ros::Publisher tracked_merged_objects_vel_marker_pub;

ros::Publisher obstacle_marker_vel_pub;
ros::Publisher obstaclelidarinlaneletPub;
ros::Publisher obstaclepedestrianinlaneletPub;
ros::Publisher obstaclemergingPub;
ros::Publisher pointcloud_pub_;
ros::Publisher pointcloud_only_laser_pub_;
ros::Publisher pointcloud_nubecolor_pub_;
ros::Publisher pubPedestrian;
ros::Publisher pubMerge;
ros::Publisher pubLaneChange;
ros::Publisher frontCarPub;
ros::Publisher pubDistOvertake;
ros::Publisher cmd_vel_pub;

ros::Publisher gps_obstacle_marker_pub;

ros::Publisher odom2_pub;

// ROS Subscribers

ros::Subscriber monitorizedlanes_sub;
ros::Subscriber route_sub; 
ros::Subscriber waiting_sub_; 
ros::Subscriber gps_sub_;
ros::Subscriber lidar_sub_;
//ros::Subscriber yolo_sub_;
//ros::Subscriber carla_sub_;
ros::Subscriber odometry_sub_;
ros::Subscriber our_odom_sub_;

// End ROS communication

// Global variables //

// Transform variables

tf::StampedTransform transformBaseLinkBaseCamera, transformOdomBaseLink, transformBaseLinkOdom;					
tf::Transform tfBaseLinkBaseCamera, tfOdomBaseLink;

tf::TransformListener *listener;
tf::StampedTransform transformMaptoVelodyne, transformVelodynetoMap;

cv::Mat matriz_P = (cv::Mat_<float>(4,4) << 671.5122,    0.0,    664.2965,   0.0,
                                   0.0,      671.5122,  347.0664,   0.0,
                                   0.0,         0.0,       1.0,     0.0,
				   0.0,         0.0,       0.0,     1.0);

cv::Mat ZED_to_lidar_matriz = (cv::Mat_<float>(4,4) <<  1.0,        0.0,    	0.0,     0.0,
                                   			0.0,        1.0,     	0.0,     0.0,
                                   			0.0,        0.0,        1.0,     0.0,
						       -0.35,      -0.06,      0.295,     1.0);

// Visualization variables

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer")); // 3D viewer
pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaculosInLanelet_Ptr (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaculosInPedestrian_Ptr (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ObstaculosMerging_Ptr (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr vlp_cloud_Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new  pcl::PointCloud<pcl::PointXYZRGB>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

namespace rvt = rviz_visual_tools;

namespace rviz_visual_tools
	{
	class RvizVisualToolsDemo 
	{
		private:
		  rvt::RvizVisualToolsPtr visual_tools_;
		  std::string name_;
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

		  void publishLabelHelper(const Eigen::Isometry3d& pose, const std::string& label)
	  	  {
			Eigen::Isometry3d pose_copy = pose;
			pose_copy.translation().x() -= 0.2;
			visual_tools_->publishText(pose_copy, label, rvt::WHITE, rvt::LARGE, false);
	  	  }

		  void Show_WireFrame(geometry_msgs::Point32 location, const std::string label)
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



// Odom in different formats and geographic variables

geographic_msgs::GeoPoint odompoint;
geodesy::UTMPoint odomUTMmsg;
nav_msgs::Odometry odomAnt;

geodesy::UTMPoint utmOrigin;
geodesy::UTMPoint utm_actual;
double lat_origin, lon_origin;
std::shared_ptr< LaneletMap > loadedMap;
geometry_msgs::Pose pose_global;
nav_msgs::OccupancyGrid localMap;

geographic_msgs::GeoPoint geo_origin;
geographic_msgs::GeoPoint geo_actual;
sensor_msgs::NavSatFix gps_data;

nav_msgs::Odometry odom2;
nav_msgs::Odometry odom2_anterior;

// SEC project variables

std::vector<Object> only_laser_objects, merged_objects, output_objects;
std::vector<Tracking_Points> tracking_points, tracking_points_prev, tracking_points_aux, tracking_points_lidar, tracking_points_prev_lidar, tracking_points_aux_lidar;
std::vector<precisiontrackers> pTrackers;
precision_tracking::Params params;
Eigen::Vector3f estimated_velocity;
std::vector<precision_tracking::Tracker> trackers;
int indexpTrackers = 0;
int flag_tracking_points=0;
double persistence_time=15.0000; // Maximum difference in time to not delete an object

int num_only_laser_objects=0, num_outputs_object=0;

sec_msgs::Route pedestrian_crossing;
sec_msgs::Route merging_lanelets;
sec_msgs::Route route_lanelets; // Lanelets on the route

sec_msgs::Route left_lanelets; // Left and Right lanelets associated to route
sec_msgs::Route right_lanelets;
sec_msgs::Route route_left_lanelets;
sec_msgs::Route route_right_lanelets;
sec_msgs::Route all_lefts;
sec_msgs::Route all_rights;

sec_msgs::Route route; // Route received
sec_msgs::Route monitorized_Lanelets;

int id_reg_lanelet; // ID of regulatory element associated with pedestrian monitor

// Monitors variables: 0 inactive, 1 active

bool merging_monitor; // Stop and Give way
bool pedestrian_monitor; // Pedestrian crossing

int pedestrian_actual_occupied=0;
int pedestrian_actual_global=0;
int cont_global=0;
int stop=0; // Flag related to stop monitor: 0 inactive, 1 active (car cannot cross the stop), 2 merging monitor (car can cross the stop if merging monitor allows)

// General purpose variables

cv::Mat imgProjection;
std::vector<std_msgs::ColorRGBA> colours;

// End global variables //

/// Functions and CallBacks ///

// Transform Global to Local points //

geometry_msgs::Point32 Global_To_Local_Coordinates(geometry_msgs::PointStamped point_global)
{
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

// End Transform Global to Local points //

// Transform Local to Global points //

geometry_msgs::Point32 Local_To_Global_Coordinates(geometry_msgs::PointStamped point_local)
{
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

// End Transform Local to Global points //

// Callback to store the full path //

void route_cb(const sec_msgs::Route::ConstPtr& Route_msg)
{	
	route=*Route_msg;
}
 
// End callback to store the full path //

// Callback to store lanelets of interest (monitorized lanelets) and regulatory elements on the route

void regelement_cb(const sec_msgs::RegElem::ConstPtr& regelem, const sec_msgs::Route::ConstPtr& monitorized_Lanelets_msg)
{
	// Store monitorized lanelets

	monitorized_Lanelets=*monitorized_Lanelets_msg;
	pedestrian_crossing.route.clear();
	merging_lanelets.route.clear();

	// Initialize merging monitor

	merging_monitor=false;

	// If there is a pedestrian crossing

	if (!strcmp(regelem->type.c_str(),"pedestrian_crossing"))
	{
		/*** IF PEDESTRIAN CROSSING IS NEAR: TODO. CHANGE THIS MONITOR ACTIVATION FOR BEHAVIOUR FROM THE PETRI NET ****/
		if (regelem->distance<30)
		{
			/*** CHANGE PEDESTRIAN CROSSING FROM PREVIOUS MONITOR, DELETE PEDESTRIAN ACTUAL OCCUPIED **/
			if (regelem->laneletID!=id_reg_lanelet)
			{
				pedestrian_actual_occupied=0;
			}
			/*** STORE CURRENT MONITOR ID ***/
			id_reg_lanelet=regelem->laneletID;
		 
			/*** STORE PEDESTRIAN CROSSING LANELETS ***/
			for (int i=0;i<monitorized_Lanelets.route.size();i++)
			{
				if (!strcmp(monitorized_Lanelets.route[i].type.c_str()," pedestrian_crossing"))
				{
					pedestrian_crossing.route.push_back(monitorized_Lanelets.route[i]);
				}
			}
			/*** ACTIVATE PEDESTRIAN MONITOR ***/
			pedestrian_monitor=true;
		 }
		 else
		 {
			/*** PEDESTRIAN CROSSING FAR AWAY... TURN OFF PEDESTRIAN MONITOR ***/
			pedestrian_monitor=false;
		 }

		//acquire data for square pedestrial crossing
		zona_MONITOR[0].x=regelem->A1.latitude;
        	zona_MONITOR[0].y=regelem->A1.longitude;
		zona_MONITOR[1].x=regelem->A2.latitude;
        	zona_MONITOR[1].y=regelem->A2.longitude;
		zona_MONITOR[2].x=regelem->A3.latitude;
        	zona_MONITOR[2].y=regelem->A3.longitude;
		zona_MONITOR[3].x=regelem->A4.latitude;
        	zona_MONITOR[3].y=regelem->A4.longitude;

        	//cout << "Elemento Regulatorio "<< regelem->type <<endl;
		//for (int i=1;i<5;i++)
		//	cout << "zona Monitor i "<< i << " " << zona_MONITOR[0].x <<endl;

	}
	else
	{
		/*** IF THERE IS NOT A PEDESTRIAN CROSSING, TURN OFF MONITOR ***/
		pedestrian_monitor=false;
	}
	 
	 
	/*** IF THERE IS AN STOP: MONITOR **/
	if (!strcmp(regelem->type.c_str(),"stop"))
	{
		/*** IF STOP MONITOR IS 0 OR 1, STOP=1 (VEHICLE CAN NOT CONTINUE) ***/
		if (stop!=2)
		{
			stop=1;
		}
	}
	else
	{
		/*** IF NOT STOP, STOP=0 (TURN OFF MONITOR) ***/
		stop=0;
	}
	 
	 
	/*** MERGING MONITOR (WHEN GIVE WAY OR STOP SIGNALS)***/
	/*** TODO: CHANGE MONITOR ACTIVATION FOR BEHAVIOUR FROM PETRI NET ***/
	if (!strcmp(regelem->type.c_str(),"give way") || !strcmp(regelem->type.c_str(),"give_way") || !strcmp(regelem->type.c_str(),"stop"))
	{
		/*** IF REGULATORY ELEMENT IS NEAR...**/
		if (regelem->distance<30)
		{
			/*** STORE MERGING LANELETS RELATED TO REGULATORY ELEMENT AND ACTIVATE MERGING MONITOR ***/
			int laneletid=regelem->laneletID;
			for (int i=0;i<monitorized_Lanelets.route.size();i++)
			{
				string type=monitorized_Lanelets.route[i].type.c_str();
				istringstream iss(type);
				do
				{
					string subs;
					iss >> subs;
					if (!strcmp(subs.c_str(),"id"))
					{
						string subs;
						iss >> subs;
						if (!strcmp(subs.c_str(),std::to_string(laneletid).c_str()))
						{
							merging_lanelets.route.push_back(monitorized_Lanelets.route[i]);
						}
					}
				} while (iss);
			}
			merging_monitor=true;
		}
	}
	/*** CREATE A VARIABLE WITH THE MONITORIZED LANELETS THAT ARE IN THE ROUTE ***/
	route_lanelets.route.clear();
	for (int i=0;i<monitorized_Lanelets.route.size();i++)
	{
		string type=monitorized_Lanelets.route[i].type.c_str();
		istringstream iss(type);
		do
		{
			string subs;
			iss >> subs;
			if (!strcmp(subs.c_str(),"route"))
			{
				route_lanelets.route.push_back(monitorized_Lanelets.route[i]);
			}
		 
		} while (iss);
	}
	 
	/*** CREATE A VARIABLE WITH THE MONITORIZED LANELETS THAT HAVE THE FLAT "LEFT". TODO: DELETE THIS FUNCTION? ***/ 
	/*left_lanelets.route.clear();
	for (int i=0;i<monitorized_Lanelets.route.size();i++)
	{
		string type=monitorized_Lanelets.route[i].type.c_str();
		istringstream iss(type);
		do
		{
			string subs;
			iss >> subs;
			if (!strcmp(subs.c_str(),"left"))
			{
				left_lanelets.route.push_back(monitorized_Lanelets.route[i]);
			}
		 
		} while (iss);
	}*/

	left_lanelets.route.clear();
	right_lanelets.route.clear();
	route_left_lanelets.route.clear();
	route_right_lanelets.route.clear();
	all_lefts.route.clear();
	all_rights.route.clear();

	for (int i=0;i<monitorized_Lanelets.route.size();i++)
	{
		string type=monitorized_Lanelets.route[i].type.c_str();
		istringstream iss(type);
		do
		{
			string subs;
			iss >> subs;
			if (!strcmp(subs.c_str(),"left"))
			{
				left_lanelets.route.push_back(monitorized_Lanelets.route[i]);
			}
		// We assume that the user creates the path along the right lanelet of the road. So, for an overtaking, the current right lanelet is represented by "route" type in Monitorized Lanelets.
			if(!strcmp(subs.c_str(),"route") || !strcmp(subs.c_str(),"merging_split_route") || !strcmp(subs.c_str(),"split_merging_route") || !strcmp(subs.c_str(),"merging_route"))
			{
				route_right_lanelets.route.push_back(monitorized_Lanelets.route[i]); 
			}
		 
		} while (iss);
	}

	if (left_lanelets.route.size()==0) // The path was defined along the left lanelet
	{
		route_right_lanelets.route.clear();
		for (int i=0;i<monitorized_Lanelets.route.size();i++)
		{
			string type=monitorized_Lanelets.route[i].type.c_str();
			istringstream iss(type);
			do
			{
				string subs;
				iss >> subs;

				if(!strcmp(subs.c_str(),"route") || !strcmp(subs.c_str(),"merging_split_route") || !strcmp(subs.c_str(),"split_merging_route") || !strcmp(subs.c_str(),"merging_route"))
				{
					route_left_lanelets.route.push_back(monitorized_Lanelets.route[i]); 
				}
		 
				if (i>0 && !strcmp(subs.c_str(),"lanelet"))
				{
					right_lanelets.route.push_back(monitorized_Lanelets.route[i]);
				}
			// We assume that the user creates the path along the right lanelet of the road. So, for an overtaking, the current right lanelet is represented by "route" type in Monitorized Lanelets.	
			} while (iss);
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

	cout<<endl<<"LEFT ID: "<<endl;
	for (int i=0;i<all_lefts.route.size();i++){
	cout<<all_lefts.route[i].id<<" ";}

	cout<<endl<<"RIGHT ID: "<<endl;
	for (int i=0;i<all_rights.route.size();i++){
	cout<<all_rights.route[i].id<<" ";}

	

	/*// THE LAST LANELET IS ALWAYS THE OPPOSITE (LEFT OR RIGHT WITH RESPECT TO OUR CURRENT POSITION). IF ITS TYPE IS NOT LEFT, IS RIGHT (type lanelet by default -> IMPROVE??).
	int size_aux = monitorized_Lanelets.route.size();

	string type=monitorized_Lanelets.route[size_aux-1].type.c_str();
	istringstream iss(type);
	string subs; 
	iss >> subs;

	if (!strcmp(subs.c_str(),"lanelet"))
	{
		right_lanelets.route.push_back(monitorized_Lanelets.route[size_aux-1]);
	}*/

	

	//cout<<"Size all lefts: "<<all_lefts.route.size()<<endl;
	//cout<<"Size all rights: "<<all_rights.route.size()<<endl;
 
 
}
 
void waitingCallBack(const std_msgs::Empty msg)
{
	/*** IF STOP SIGNAL IS RECEIVED (CAR IS STOP IN A REGULATORY ELEMENT "STOP"), FLAG=2 TO ALLOW THE CAR CONTINUE ***/
	stop=2;
}

// Inside polygon function (Useful for pedestrian crossing behaviour)

int InsidePolygon(Point_camera *polygon, int N, Point_camera p)
{
	int counter = 0;
	double xinters; // Intersecciones con una línea paralela al eje X
	Point_camera p1, p2;

	p1 = polygon[0];
	for (int i=1;i<=N;i++){
		p2 = polygon[i%N]; // Al hacer el resto, esto mirando los otros vértices del polígono
		if (p.y > MIN_MONITOR(p1.y,p2.y)){ // Podemos estar por encima del mínimo de ambos, pero como luego no estemos por debajo del máximo de ambos, estamos fuera
			if (p.y <= MAX_MONITOR(p1.y, p2.y)){
				if (p.x <= MAX_MONITOR(p1.x, p2.x)){ // Nos quedamos con la franja izquierda del plano acotado verticalmente
					if (p1.y != p2.y) { // Nos aseguramos que no están en la misma cota vertical
						xinters = (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x; // A p1.y le sumamos a lo que equivale en horizontal la distancia en vertical de p a p1, es como hacer 													   // una regla de 3
						if (p1.x == p2.x || p.x <= xinters){ // Si ambos están alineados en horizontal, con lo cual con las condiciones anteriores directamente estaríamos dentro de la línea, o la cota x es menor o igual que la equivalencia calculada sumada a la cota del primer punto, significa que hay intersección con la línea p1-p2 actual trazando una recta paralela al eje X hacia la izquierda
						counter++;}
					}
				}
			}
		}
			p1 = p2; // Así vamos continuando con los puntos del polígno
	}

	if (counter % 2 == 0)
	{ // Si el número de intersecciones (1 máximo por cada línea, la intersección no traspasa líneas) es par, estamos fuera. Si es impar, estamos dentro (Comprobar con lapiz y papel)
	cout << "Fuera" << endl;
	return(OUTSIDE); // = 1
	}
	else
	{
	cout << "Dentro" << endl;
	return(INSIDE); // = 0
	}

}

// End Inside polygon function (Useful for pedestrian crossing behaviour)

// PointCloud filters //

// XYZ filter

pcl::PointCloud<pcl::PointXYZRGB> xyz_filter (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered) 
{
	/*** FILTER A POINTCLOUD WITH VALUES OVER GIVEN Z (-0.6) **/
	/*** TODO: USE PCL_FUNCTION INSTEAD OF FOR.
	 *   OBTAIN Z VALUE BY PARAMETERS ***/
	pcl::PointCloud<pcl::PointXYZRGB> output_cloud;

	for (int i = 0; i < cloud_filtered->points.size(); i++)
	{
		pcl::PointXYZRGB aux_cloud;
		aux_cloud = cloud_filtered->points[i];

		// Z must be above the sidewalk. If we set cloud_filtered->points[i].z>-1.7, the LiDAR sees the sidewalk and considers it as an obstacle -> Totally Wrong

		if (cloud_filtered->points[i].z > -1.3 && cloud_filtered->points[i].z < 1.0 && cloud_filtered->points[i].x > -5.0 && cloud_filtered->points[i].x < 50 &&cloud_filtered->points[i].y > -10 && cloud_filtered->points[i].y < 10)
		{
			output_cloud.points.push_back(aux_cloud);
		}
	}

	return output_cloud;
}

// End XYZ filter

// Cluster filter

void cluster_filter (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered, float tolerance, int min_cluster, int max_cluster, std::vector<Object> *output_objects, int *num_outputs_objects)
{
	// Object clustering

	// Parameters:
	// cloud_filtered: Cloud where obtain clusters
	// tolerance: tolerance of clusters
	// min_cluster: min size of cluster
	// max_cluster: max size of cluster
	// output_objects: clustered objects
	// num_output_objects: Number of clustered objects
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr z0_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	// Travel all the points and eliminate points further from the car
	/*** TODO: CHECK THIS. MAYBE IS BETTER TO CREATE A SEPARATE FUNCTION AND OBTAIN DISTANCE BY PARAMETER ***/

	for (int i = 0; i < cloud_filtered->points.size(); i++)
	{
		pcl::PointXYZRGB aux_cloud;
		aux_cloud = cloud_filtered->points[i];
		double dist=sqrt(pow(cloud_filtered->points[i].x,2) + pow(cloud_filtered->points[i].y,2));
		if (dist<35)
		{
			output_cloud->points.push_back(aux_cloud);
			// Project cloud to floor (avoid several clusters in Z axis)
			//aux_cloud.z=0;
			z0_cloud->points.push_back(aux_cloud);
		}
	}
 
 	if (DEBUG)
	{
		std::cout << "PointCloud RGB after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //
	}

	// Extract clusters from PointCloud

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (z0_cloud);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (tolerance); //2 0.5
	ec.setMinClusterSize (min_cluster); //50
	ec.setMaxClusterSize (max_cluster); //25000
	ec.setSearchMethod (tree);
	ec.setInputCloud (z0_cloud);
	ec.extract (cluster_indices);

	int j = 0;

	// Store all clusters

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
 
		// For each cluster, obtain dimensions and centroid and store

		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
		{
			cloud_cluster->points.push_back (output_cloud->points[*pit]);
		}
 
		cloud_cluster->width = cloud_cluster->points.size ();
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
 
		centroid_x=(x_max+x_min)/2.0;
		centroid_y=(y_max+y_min)/2.0;
		centroid_z=(z_max+z_min)/2.0;
 
		object.centroid_x = centroid_x;
		object.centroid_y = centroid_y;
		object.centroid_z = centroid_z;
 
		object.x_max=x_max;
		object.x_min=x_min;
		object.y_max=y_max;
		object.y_min=y_min;
		object.z_max=z_max;
		object.z_min=z_min;
 
		object.cloud=cloud_cluster;
 
		output_objects->push_back(object); 
 
		if (DEBUG)
		{
			 std::cout << "Cluster " << j <<  " centroid_x " << centroid_x << " centroid_y " << centroid_y << " centroid_z " << centroid_z << std::endl;
		}

		j++;
	}
}

// End cluster filter

// End PointCloud filters

// Cluster fusion based on LiDAR and camera (tracked objects based on CenterNet + DeepSort + YOLOv3) and Monitors

void tracking_lidar_camera(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const yolov3_centernet_ros::yolo_list::ConstPtr& yolo_msg, const carla_msgs::CarlaObjectLocationList::ConstPtr& carla_msg)
{
	// lidar_msg contains the LiDAR information
	// yolo_msg contains the detected obstacles already tracked and identified by using label (car, person, .), object_ID (corresponding object in the scene) and position information both in camera frame and real world frame
	// carla_msg contains the position of all introduced CARLA objects for a given frame

	// Note that if --clock is not published (if we are trying to run a rosbag), the system will not create the transforms
	try
	{
		listener->waitForTransform("map","base_link",yolo_msg->header.stamp,ros::Duration(3.0));
		listener->lookupTransform ("map", "base_link", yolo_msg->header.stamp, transformOdomBaseLink);
		listener->lookupTransform ("base_link", "map", yolo_msg->header.stamp, transformBaseLinkOdom); 
	}
	catch (tf::TransformException& e)
	{
		std::cout<<e.what();
		return;
	}

	// To represent WireFrames in RVIZ

	//rviz_visual_tools::RvizVisualToolsDemo tracking_visual_tools;

	cv::Mat point3d_velodyne_matrix = (cv::Mat_<float>(1,4) << 0.0, 0.0, 0.0, 1.0);
	cv::Mat pixels_matrix_aux = (cv::Mat_<float>(4,1) << 0.0, 0.0, 0.0, 1.0);

	sec_msgs::Obstacle obstacle_fusion;
	sec_msgs::ObstacleArray obstacle_fusion_array;

	std::vector<Vertice> vertices_visualization_array;
	std::vector<Vertice> vertices_aux;
	Vertice vertice_point;

	// ACC variables

	double distfrontcar = 5000000;
	sec_msgs::Obstacle frontCarObstacle;

	// Auxiliar variables

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr vlp_cloud_Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	
	// Read in the cloud data

	pcl::PCDReader reader;

	// If Full Cloud is used ...

	pcl::fromROSMsg(*lidar_msg, *vlp_cloud_Ptr); // vlp_cloud_Ptr stores the LiDAR point cloud
	*cloud_filtered = *vlp_cloud_Ptr;

	// Filter cloud

	pcl::PointCloud<pcl::PointXYZRGB> cl_filter = xyz_filter(cloud_filtered);
	*cloud_filtered = cl_filter;

	sensor_msgs::PointCloud2 cloud;
	pcl::toROSMsg(*cloud_filtered, cloud);
        //cloud.header.frame_id = "velodyne";
	cloud.header.frame_id = "ego_vehicle/lidar/lidar1";
	cloud.header.stamp = lidar_msg -> header.stamp;

	// Publish only LiDAR cloud

	pointcloud_only_laser_pub_.publish(cloud);

	// Cluster Segmentation

	// Creating the KdTree object for cluster extraction (Cloud filtered)

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree -> setInputCloud (cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
  	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  	ec.setClusterTolerance (1);
  	ec.setMinClusterSize (2);
  	ec.setMaxClusterSize (25000);
  	ec.setSearchMethod (tree);
  	ec.setInputCloud (cloud_filtered); // Clusters will be obtained from this filtered cloud
  	ec.extract (cluster_indices);

	num_only_laser_objects = 0;

	vertices_aux.clear();
	vertices_visualization_array.clear();
	only_laser_objects.clear();
	merged_objects.clear();

	if (flag_tracking_points == 0) // Initialize only in the first iteration
	{
		// Only vision tracking
		tracking_points.clear();
		tracking_points_prev.clear();
		tracking_points_aux.clear();

		// LiDAR tracking after merging with camera data
		tracking_points_lidar.clear();
		tracking_points_prev_lidar.clear();
		tracking_points_aux_lidar.clear();

		flag_tracking_points = 1;
	}

	//visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("/map","/vision_rviz_visual_markers")); // To represent wireframe cuboids 
	//visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("/map","/fusion_rviz_visual_markers"));

	std::stringstream ss, ss2, ss3, ss4, ss5, ss6, ss7;

	// Visualize pointclouds with 3D viewer //

	// Refresh 3D viewer

	if (VIEWER_3D)
	{
		viewer -> spinOnce(0.5);
		viewer -> removeAllPointClouds();
		viewer -> removeAllShapes();
	}

	pcl::PCLPointCloud2::Ptr Input_Cloud (new pcl::PCLPointCloud2 ());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_with_colour (new  pcl::PointCloud<pcl::PointXYZRGB>);
	pcl_conversions::toPCL(*lidar_msg,*Input_Cloud);
	pcl::fromPCLPointCloud2(*Input_Cloud, *cloud_with_colour);

	// Cloud with colour

	double r,g,b;
	r=255;
	g=255;
	b=255;

	ss << "Point Cloud with colour 3D viewer"; // Each PCL must have a different stringstream

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_handler (cloud_with_colour,r,g,b);

	if (VIEWER_3D)
	{
		viewer->addPointCloud<pcl::PointXYZRGB> (cloud_with_colour,color_handler, ss.str());
	}

	// Cloud filtered

	r=0;
	g=0;
	b=255;

	ss2 << "Point Cloud filtered with colour 3D viewer"; // Each PCL must have a different stringstream

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_handler2 (cloud_filtered,r,g,b);

	if (VIEWER_3D)
	{
		viewer->addPointCloud<pcl::PointXYZRGB> (cloud_filtered,color_handler2, ss2.str());
	}

	// End Visualize pointclouds with 3D viewer //

	// Store the LiDAR clusters //

	//cout<<endl<<endl<<"Cluster indices size: "<<cluster_indices.size()<<endl<<endl;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
		
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
		{
			cloud_cluster->points.push_back (cloud_filtered->points[*pit]); 
		}

		// Fill "object" strcture with detected clusters

		object.cloud = cloud_cluster;

		cloud_cluster -> width = cloud_cluster -> points.size();
		cloud_cluster -> height = 1;
		cloud_cluster -> is_dense = true;

		// Initialize point cloud vertices. Set to +/- INFINITY to ensure a proper behaviour for the first cluster 

		float x_min = INFINITY; 
		float y_min = INFINITY;
		float z_min = INFINITY;
		float x_max = -INFINITY;
		float y_max = -INFINITY;
		float z_max = -INFINITY;
 
		float centroid_x=-INFINITY;
		float centroid_y=-INFINITY;
		float centroid_z=-INFINITY;
		float length_x=-INFINITY;
		float width_y=-INFINITY;
		float height_z=-INFINITY;

		for (int i = 0; i < cloud_cluster->points.size(); i++)
		{
			if (cloud_cluster->points[i].x < x_min)		
			{
				x_min = cloud_cluster->points[i].x;
				vertice_point.vertice_point[0].x = cloud_cluster->points[i].x;
				vertice_point.vertice_point[0].y = cloud_cluster->points[i].y;
				vertice_point.vertice_point[0].z = -1.95; //Nivel del suelo
			}
 
			if (cloud_cluster->points[i].y < y_min)		
			{
				y_min = cloud_cluster->points[i].y;
				vertice_point.vertice_point[1].x = cloud_cluster->points[i].x;
				vertice_point.vertice_point[1].y = cloud_cluster->points[i].y;
				vertice_point.vertice_point[1].z = -1.95; //Nivel del suelo
			}
 
			if (cloud_cluster->points[i].z < z_min)	
			{
				z_min = cloud_cluster->points[i].z;		
			}
			if (cloud_cluster->points[i].x > x_max)
			{
				x_max = cloud_cluster->points[i].x;
				vertice_point.vertice_point[2].x = cloud_cluster->points[i].x;
				vertice_point.vertice_point[2].y = cloud_cluster->points[i].y;
				vertice_point.vertice_point[2].z = -1.95; //Nivel del suelo
			}
			if (cloud_cluster->points[i].y > y_max)
			{
				y_max = cloud_cluster->points[i].y;
				vertice_point.vertice_point[3].x = cloud_cluster->points[i].x;
				vertice_point.vertice_point[3].y = cloud_cluster->points[i].y;
				vertice_point.vertice_point[3].z = -1.95; //Nivel del suelo
			}
			if (cloud_cluster->points[i].z > z_max)
			{
				z_max = cloud_cluster->points[i].z;
			}
		}

		centroid_x=(x_max+x_min)/2.0;
		centroid_y=(y_max+y_min)/2.0;
		centroid_z=(z_max+z_min)/2.0;

		// Centroid

		Eigen::Vector4f centroid2;
		pcl::compute3DCentroid(*cloud_cluster,centroid2);

		//time = ros::Time::now().toSec();

		geometry_msgs::PointStamped local_centroid;
		geometry_msgs::Point32 global_centroid;

		local_centroid.point.x = centroid_x;
		local_centroid.point.y = centroid_y;
		local_centroid.point.z = centroid_z;

		global_centroid = Local_To_Global_Coordinates(local_centroid);

		if (LaneletFilter == 1)
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

					object.x_max=x_max;
					object.x_min=x_min;
					object.y_max=y_max;
					object.y_min=y_min;
					object.z_max=z_max;
					object.z_min=z_min;

					length_x = x_max-x_min;
					width_y = y_max-y_min;
					height_z = z_max - z_min;

					// Local coordinates with respect to the velodyne

					//object.centroid_x = centroid_x;
					//object.centroid_y = centroid_y;
					//object.centroid_z = centroid_z;

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

					object.cloud=cloud_cluster;

					only_laser_objects.push_back(object);
					vertices_visualization_array.push_back(vertice_point);
	
					num_only_laser_objects++;

					break; // Continue with the next object
				}
			}

			// End Lanelets filter //
		}
		else 
		{
			if (x_max-x_min<4 && y_max-y_min<2) // To avoid huge group of non-sense points
			{
				// Store all objects, both in and out the monitorizedLanelets

				// Object measurements

				object.x_max=x_max;
				object.x_min=x_min;
				object.y_max=y_max;
				object.y_min=y_min;
				object.z_max=z_max;
				object.z_min=z_min;

				length_x = x_max-x_min;
				width_y = y_max-y_min;
				height_z = z_max - z_min;

				// Local coordinates with respect to the velodyne

				//object.centroid_x = centroid_x;
				//object.centroid_y = centroid_y;
				//object.centroid_z = centroid_z;

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

				// Time

				double time = yolo_msg->header.stamp.toSec();
				object.time = time;

				// Store closest CARLA object with respect to LiDAR

				float ccx, ccy; // CARLA centroid (Local)
				ccx = ccy = 0;
				float diff_lidar_carla = 0;
				int index_most_similar_lidar_carla = 0;
				float max_diff_lidar_carla = 4;

				for (int k=0; k<carla_msg->carlaobjectlocationlist.size(); k++)
				{
					geometry_msgs::PointStamped global_carla_centroid;
					geometry_msgs::Point32 local_carla_centroid;

					global_carla_centroid.point.x = float(carla_msg->carlaobjectlocationlist[k].pos_x); 
					global_carla_centroid.point.y = float(-carla_msg->carlaobjectlocationlist[k].pos_y); // Note that CARLA gives you the Y-axis in the opposite direction
					global_carla_centroid.point.z = float(carla_msg->carlaobjectlocationlist[k].pos_z);

					local_carla_centroid = Global_To_Local_Coordinates(global_carla_centroid);

					ccx = local_carla_centroid.x;
					ccy = local_carla_centroid.y;

					diff_lidar_carla = float(sqrt(pow(local_centroid.point.x-ccx,2)+pow(local_centroid.point.y-ccy,2))); 

					if (diff_lidar_carla < max_diff_lidar_carla) // Find the closest cluster
					{
						max_diff_lidar_carla = diff_lidar_carla;
						index_most_similar_lidar_carla = k;
					}
				}

				//cout<<endl<<"Difference LiDAR CARLA: "<<diff_lidar_carla;

				if (diff_lidar_carla < 2)
				{
					object.carla_id = carla_msg->carlaobjectlocationlist[index_most_similar_lidar_carla].object_id;
				}
				else
				{
					object.carla_id = 0; // LiDAR object not associated with any CARLA object
				}

				only_laser_objects.push_back(object);
				vertices_visualization_array.push_back(vertice_point);

				num_only_laser_objects++;
			}
		}
	}

	// End store the LiDAR clusters //

	// YOLO filter //

	// Variables for YOLO + LiDAR fusion
	 
	float ycx, ycy; // YOLO centroid (Local)
	ycx = ycy = 0;
	float lcx, lcy; // LiDAR centroid (Local)
	lcx = lcy = 0;
	float ccx, ccy; // CARLA centroid (Local)
	ccx = ccy = 0;
	float diff_lidar_vot = 0; // Distance between LiDAR estimation and Visual Object Tracking estimation
	float diff_vot_carla = 0; // Distance between VOT (Visual Object Tracking) estimation and CARLA groundtruth
	int index_most_similar_lidar_vot = 0;
	int index_most_similar_vot_carla = 0;
	int object_id = 0; // VOT reference

	for (int i=0; i<yolo_msg->yolo_list.size(); i++) // YOLO is more restrictive than LiDAR
	{
		float max_diff_lidar_vot = 4; // Initialize maximum allowed difference
		float max_diff_vot_carla = 4;

		ycx = float(yolo_msg->yolo_list[i].tx); // Distance to the object with respect to the camera but with Velodyne frame
		ycy = float(yolo_msg->yolo_list[i].ty);

		if (only_laser_objects.size() > 0 && (!strcmp(yolo_msg->yolo_list[i].type.c_str(),"car") || !strcmp(yolo_msg->yolo_list[i].type.c_str(),"person"))) 
		{
			//cout<<endl<<"Yolo msg list size: "<<yolo_msg->yolo_list.size()<<endl;
			double time = yolo_msg->header.stamp.toSec();
			object_id = yolo_msg->yolo_list[i].object_id;

			geometry_msgs::PointStamped local_centroid;
			geometry_msgs::Point32 global_centroid;

			local_centroid.point.x = ycx;
			local_centroid.point.y = ycy;
			local_centroid.point.z = 0;

			global_centroid = Local_To_Global_Coordinates(local_centroid);

			// Store closest CARLA object with respect to VOT

			for (int k=0; k<carla_msg->carlaobjectlocationlist.size(); k++)
			{
				geometry_msgs::PointStamped global_carla_centroid;
				geometry_msgs::Point32 local_carla_centroid;

				global_carla_centroid.point.x = float(carla_msg->carlaobjectlocationlist[k].pos_x); 
				global_carla_centroid.point.y = float(-carla_msg->carlaobjectlocationlist[k].pos_y); // Note that CARLA gives you the Y-axis in the opposite direction
				global_carla_centroid.point.z = float(carla_msg->carlaobjectlocationlist[k].pos_z);

				local_carla_centroid = Global_To_Local_Coordinates(global_carla_centroid);

				ccx = local_carla_centroid.x;
				ccy = local_carla_centroid.y;

				diff_vot_carla = float(sqrt(pow(ycx-ccx,2)+pow(ycy-ccy,2))); 

				if (diff_vot_carla < max_diff_vot_carla) // Find the closest cluster
				{
					max_diff_vot_carla = diff_vot_carla;
					index_most_similar_vot_carla = k;
				}
			}

			//cout<<endl<<"Difference VOT Carla: "<<diff_vot_carla<<endl;

			// 1. Find out if current detected object was previously stored. If does, update the object

			int flag = 0;

			for (int l=0; l<tracking_points.size(); l++)
			{
				if (tracking_points[l].object_id == object_id)
				{
					tracking_points[l].global_centroid_x = global_centroid.x;
					tracking_points[l].global_centroid_y = global_centroid.y;
					tracking_points[l].local_centroid_x = local_centroid.point.x;
					tracking_points[l].local_centroid_y = local_centroid.point.y;

					tracking_points[l].object_id = object_id;

					if (carla_msg->carlaobjectlocationlist.size() > 0 && diff_vot_carla<3)
					{
						
						tracking_points[l].carla_id = carla_msg->carlaobjectlocationlist[index_most_similar_vot_carla].object_id;
					} 
					else
					{
						tracking_points[l].carla_id = 0; // VOT object not associated with any CARLA object
					}

					//cout<<"Carla Object associated "<<tracking_points[l].carla_id<<endl;

					tracking_points[l].time = time;

					flag = 1;
					break;
				}
			}

			// 2. If it was not previously stored, then create a new object

			if (flag == 0)
			{
				Tracking_Points tracking_point;

				tracking_point.global_centroid_x = global_centroid.x;
				tracking_point.global_centroid_y = global_centroid.y;
				tracking_point.local_centroid_x = local_centroid.point.x;
				tracking_point.local_centroid_y = local_centroid.point.y;

				tracking_point.object_id = object_id;

				if (carla_msg->carlaobjectlocationlist.size() > 0 && diff_vot_carla<3)
				{
					tracking_point.carla_id = carla_msg->carlaobjectlocationlist[index_most_similar_vot_carla].object_id;
				}
				else
				{
					tracking_point.carla_id = 0; // VOT object not associated with any CARLA object
				}

				tracking_point.time = time;
				tracking_point.yaw_samples.clear();

				tracking_points.push_back(tracking_point);
			}

			// Find closest LiDAR object with respecto to VOT

			for (int j=0; j<only_laser_objects.size(); j++)
			{
				lcx = float(only_laser_objects[j].centroid_x); 
				lcy = float(only_laser_objects[j].centroid_y); 

				diff_lidar_vot = float(sqrt(pow(ycx-lcx,2)+pow(ycy-lcy,2))); 

				if (diff_lidar_vot < max_diff_lidar_vot) // Find the closest cluster
				{
					max_diff_lidar_vot = diff_lidar_vot;
					index_most_similar_lidar_vot = j;
				}
			}

			if (max_diff_lidar_vot < 3) // In order to merge both information, the centroid separation have to less that 1.5 (Yolo object and closes LiDAR object)
			{
				only_laser_objects[index_most_similar_lidar_vot].type = yolo_msg->yolo_list[i].type;
				only_laser_objects[index_most_similar_lidar_vot].object_id = yolo_msg->yolo_list[i].object_id;
				only_laser_objects[index_most_similar_lidar_vot].r = yolo_msg->yolo_list[i].color.r;
				only_laser_objects[index_most_similar_lidar_vot].g = yolo_msg->yolo_list[i].color.g;
				only_laser_objects[index_most_similar_lidar_vot].b = yolo_msg->yolo_list[i].color.b;
				only_laser_objects[index_most_similar_lidar_vot].a = yolo_msg->yolo_list[i].color.a;

				// Monitorizing position and velocity in the 3D using LiDAR and CARLA (only ID)

				// 1. Find out if current merged object was previously stored. Id does, update the object

				int flag = 0;

				for (int l=0; l<tracking_points_lidar.size(); l++)
				{
					if (tracking_points_lidar[l].object_id == object_id)
					{
						tracking_points_lidar[l].global_centroid_x = only_laser_objects[index_most_similar_lidar_vot].centroid_global_x;
						tracking_points_lidar[l].global_centroid_y = only_laser_objects[index_most_similar_lidar_vot].centroid_global_y;
						tracking_points_lidar[l].local_centroid_x = only_laser_objects[index_most_similar_lidar_vot].centroid_x;
						tracking_points_lidar[l].local_centroid_y = only_laser_objects[index_most_similar_lidar_vot].centroid_y;
						tracking_points_lidar[l].object_id = only_laser_objects[index_most_similar_lidar_vot].object_id;
						tracking_points_lidar[l].carla_id = only_laser_objects[index_most_similar_lidar_vot].carla_id;
					}
				}

				// 2. If it was not previously stored, then create a new object
	
				if (flag == 0)
				{
					Tracking_Points tracking_point;

					tracking_point.global_centroid_x = only_laser_objects[index_most_similar_lidar_vot].centroid_global_x;
					tracking_point.global_centroid_y = only_laser_objects[index_most_similar_lidar_vot].centroid_global_y;
					tracking_point.local_centroid_x = only_laser_objects[index_most_similar_lidar_vot].centroid_x;
					tracking_point.local_centroid_y = only_laser_objects[index_most_similar_lidar_vot].centroid_y;
					tracking_point.object_id = only_laser_objects[index_most_similar_lidar_vot].object_id;
					tracking_point.carla_id = only_laser_objects[index_most_similar_lidar_vot].carla_id;
					tracking_point.time = only_laser_objects[index_most_similar_lidar_vot].time;

					tracking_points_lidar.push_back(tracking_point);
				}
			}
		}	
	}

	// Update VOT Tracking Points //

	if (yolo_msg->yolo_list.size() > 0 && tracking_points.size() > 0 && tracking_points_prev.size() > 0)
	{
		cout<<endl<<"Update VOT tracking points"<<endl;
		// 1. Store in auxiliar buffer

		for (int n=0; n<tracking_points.size(); n++)
		{
			tracking_points_aux.push_back(tracking_points[n]);
		}

		// 2. Erase: If the difference between last sample time and current time is higher than three seconds, erase this element

		for (int t=0; t<tracking_points.size(); t++)
		{
			double time = yolo_msg->header.stamp.toSec();
			cout<<endl<<"tiempos vot: "<<tracking_points[t].time<<" "<<time;
			if (tracking_points[t].time + persistence_time < time)
			{
				cout<<endl<<"HAGO UN ERASE VOT"<<endl;
				tracking_points_aux.erase(tracking_points_aux.begin()+t);
			}
		}
		
		// 3. Update

		tracking_points.clear();
		
		for (int p=0; p<tracking_points_aux.size(); p++)
		{
			tracking_points.push_back(tracking_points_aux[p]);
		}

		tracking_points_aux.clear();
	}

	// Update Merged Tracking Points

	// TODO: Fix this update

	if (yolo_msg->yolo_list.size() > 0 && tracking_points_lidar.size() > 0 && tracking_points_prev_lidar.size() > 0)
	{
		cout<<endl<<"Update LiDAR tracking points"<<endl;
		// 1. Store in auxiliar buffer

		for (int n=0; n<tracking_points_lidar.size(); n++)
		{
			tracking_points_aux_lidar.push_back(tracking_points_lidar[n]);
		}

		// 2. Erase: If the difference between last sample time and current time is higher than three seconds, erase this element

		for (int t=0; t<tracking_points_lidar.size(); t++)
		{
			double time = yolo_msg->header.stamp.toSec();
			if (tracking_points_lidar[t].time + persistence_time < time)
			{
				cout<<endl<<"tiempos lidar: "<<tracking_points_lidar[t].time<<" "<<time;
				cout<<endl<<"HAGO UN ERASE LIDAR"<<endl;
				tracking_points_aux_lidar.erase(tracking_points_aux_lidar.begin()+t);
			}
		}
		
		// 3. Update

		tracking_points_lidar.clear();
		
		for (int p=0; p<tracking_points_aux_lidar.size(); p++)
		{
			tracking_points_lidar.push_back(tracking_points_aux_lidar[p]);
		}

		tracking_points_aux_lidar.clear();
	}

	// End YOLO filter //

	// Store LiDAR objects identified by YOLO in a separate array

	int k=0;

	for (int i=0; i<only_laser_objects.size(); i++)
	{
		if (!strcmp(only_laser_objects[i].type.c_str(),"car") || !strcmp(only_laser_objects[i].type.c_str(),"person")) // Different to none
		{
			merged_objects.push_back(only_laser_objects[i]);
		}
	}

	cout <<endl<< "Number of original LiDAR objects: " << only_laser_objects.size() << endl;
	cout << "Number of Yolo objects: " << yolo_msg->yolo_list.size() << endl;
	cout << "Number of merged objects: " << merged_objects.size() << endl<<endl;

	// Obstacles visualization //

	// Original LiDAR objects (Green)

	for (int i=0; i<only_laser_objects.size(); i++)
	{
		r=0; 
		g=200; 
		b=0; 
		r=r/255.0;
		g=g/255.0;
		b=b/255.0;

		float xmin = only_laser_objects[i].x_min;
		float xmax = only_laser_objects[i].x_max;
		float ymin = only_laser_objects[i].y_min;
		float ymax = only_laser_objects[i].y_max;
		float zmin = only_laser_objects[i].z_min;
		float zmax = only_laser_objects[i].z_max;

		ss3 <<"Original LiDAR clusters"; // Note that each iteration must have its own streamstring (ss), so it is repeated  in every loop

		//viewer->addCube(xmin,xmax,ymin,ymax,-1.8, 0,r,g,b,ss3.str());
		if (VIEWER_3D)
		{
			viewer->addCube(xmin,xmax,ymin,ymax,zmin,zmax,r,g,b,ss3.str());
		}

		//cout<<"LiDAR centroid (Solo laser): "<<i<<" "<<only_laser_objects[i].centroid_x<<" "<<only_laser_objects[i].centroid_y<<endl;

		ss4 << "Point Cloud Original objects"; // Each PCL must have a different stringstream

		r=0;
		g=255;
		b=0;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud (new  pcl::PointCloud<pcl::PointXYZRGB>);
		object_cloud = only_laser_objects[i].cloud;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_handler (object_cloud,r,g,b); // For ColorHandlerCustom, the r,g,b MUST NOT have by divided by 255

		if (VIEWER_3D)
		{
			viewer->addPointCloud<pcl::PointXYZRGB> (object_cloud,color_handler, ss4.str());
		}
	}

	cout<<endl<<"--------------------VISION START--------------------"<<endl;

	// YOLO 3D objects (Red)

	// TODO: Improve the 3D projection (deepsort code) or train the deepsort in 3D to obtain a better centroid

	for (int i=0; i<yolo_msg->yolo_list.size(); i++)
	{
		if (!strcmp(yolo_msg->yolo_list[i].type.c_str(),"car") || !strcmp(yolo_msg->yolo_list[i].type.c_str(),"person"))
		{
			cout<<endl<<"Yolo centroid and type: "<<i<<" X "<<yolo_msg->yolo_list[i].tx<<" Y "<<yolo_msg->yolo_list[i].ty<<" Type "<<yolo_msg->yolo_list[i].type<<endl;

			r=255;
			g=0;
			b=0;
			r=r/255.0;
			g=g/255.0;
			b=b/255.0;

			float xmin = yolo_msg->yolo_list[i].tx-1;
			float xmax = yolo_msg->yolo_list[i].tx+1;
			float ymin = yolo_msg->yolo_list[i].ty-0.5;
			float ymax = yolo_msg->yolo_list[i].ty+0.5;

			ss5 <<"Original YOLOv3 clusters"; 

			if (VIEWER_3D)
			{
				viewer->addCube(xmin,xmax,ymin,ymax, -1.8, -0.8,r,g,b,ss5.str());
			}

			// Visualize object in RVIZ

			geometry_msgs::Point32 p0,p1, p1_local;
			double t1;

			int object_id = yolo_msg->yolo_list[i].object_id;
			double yaw, yaw_previous;

			visualization_msgs::Marker obstacle_points;

			obstacle_points.header.frame_id = "/map"; // map == global coordinates. Base_link == local coordinates
			obstacle_points.header.stamp = ros::Time::now();
			obstacle_points.ns = "map_manager_visualization";
			obstacle_points.action = visualization_msgs::Marker::ADD;
			obstacle_points.type = visualization_msgs::Marker::SPHERE;
		 	obstacle_points.id = object_id;

			obstacle_points.points.clear();

			obstacle_points.color = yolo_msg->yolo_list[i].color;
			obstacle_points.scale.x = 0.25;
			obstacle_points.scale.y = 0.25;
			obstacle_points.scale.z = 0.25;
			obstacle_points.lifetime = ros::Duration(0.25);

			int flag = 0;

			//cout<<endl<<"Object ID: "<<object_id;

			for (int k=0; k<tracking_points.size(); k++)
			{
				//cout<<endl<<endl<<endl<<"Tracking Object ID and CARLAID: "<<tracking_points[k].object_id<<" "<<tracking_points[k].carla_id<<endl<<endl<<endl;
				if (tracking_points[k].object_id == object_id && tracking_points[k].carla_id != 0)
				{
					flag = 1;
					p1.x = tracking_points[k].global_centroid_x;
					p1.y = tracking_points[k].global_centroid_y;
					p1.z = 0;
					p1_local.x = tracking_points[k].local_centroid_x;
					p1_local.y = tracking_points[k].local_centroid_y;
					p1_local.z = 0;
					t1 = tracking_points[k].time;

					break;
				}
			}

			obstacle_points.pose.position.x = p1.x;
			obstacle_points.pose.position.y = p1.y;
			obstacle_points.pose.position.z = p1.z;

                        tracked_objects_marker_pub.publish(obstacle_points);

			// TODO: Fix this wireframe representation

			/*Eigen::Affine3d pose;
			
			pose.translation() = Eigen::Vector3d(p1.x, p1.y, p1.z);

			auto id_rviz = to_string(object_id);
			tracking_visual_tools.Show_WireFrame(p1,id_rviz);*/

			if (flag == 1)
			{
				float euclidean_distance, diff_y, diff_x;
				int flag_carla_object = 0;

				/*cout<<endl<<"Tracked objects size: "<<tracking_points.size()<<endl;
				cout<<"Tracked object ID and CARLA ID: "<<tracking_points[k].object_id<<" "<<tracking_points[k].carla_id<<endl;
				cout<<"Yolo size: "<<yolo_msg->yolo_list.size()<<endl;
				cout<<"CARLA size: "<<carla_msg->carlaobjectlocationlist.size()<<endl;*/

				int t;

				for (t=0; t<carla_msg->carlaobjectlocationlist.size(); t++)
				{
					//cout<<"CARLA ID: "<<carla_msg->carlaobjectlocationlist[t].object_id<<endl;
					if (int(tracking_points[k].carla_id) == int(carla_msg->carlaobjectlocationlist[t].object_id))
					{
						flag_carla_object = 1;
						break;
					}
				}

				// Store comparison between projected vision and groundtruth of CARLA //
			
				if (flag_carla_object == 1)
				{
					int vot_id = -1; // To know that this .txt line belongs to VOT-CARLA comparison

					geometry_msgs::PointStamped global_centroid;
					geometry_msgs::Point32 local_centroid;
					global_centroid.point.x = float(carla_msg->carlaobjectlocationlist[t].pos_x);
					global_centroid.point.y = float(-carla_msg->carlaobjectlocationlist[t].pos_y); // The Y-axis sign is the opposite in CARLA
					global_centroid.point.z = float(carla_msg->carlaobjectlocationlist[t].pos_z);
					local_centroid = Global_To_Local_Coordinates(global_centroid); 

					diff_x = float(p1_local.x) - float(local_centroid.x);
					diff_y = float(p1_local.y) - float(local_centroid.y); 

					//cout<<endl<<"Puntos a escribir VOT: "<<p1_local.x<<" "<<p1_local.y<<" "<<local_centroid.x<<" "<<local_centroid.y<<endl;

					euclidean_distance = float(sqrt(pow(diff_x,2)+pow(diff_y,2)));

					string sub_path ("/home/robesafe/compartido_con_docker/Nuevos_Ficheros_CGH/tracking_results/a_la_vez/");

					// Invidual object tracking
					/*
					string a ("vision_object_");
					auto b = to_string(int(tracking_points[k].object_id)); // VOT ID, not CARLA ID
					string c (".txt");
					string filename = a+b+c;
					*/
				
					// All tracked objects
					string filename ("tracked_objects.txt");
					string path = sub_path+filename;
			
					ofstream tracking_file;

					tracking_file.open(path, ios::app);

					int carla_size = int(carla_msg->carlaobjectlocationlist.size());

					tracking_file<<vot_id<<" "<<yolo_msg->yolo_list.size()<<" "<<carla_msg->carlaobjectlocationlist.size()<<" "<<tracking_points[k].object_id<<" "<<tracking_points[k].carla_id<<" "<<local_centroid.x<<" "<<local_centroid.y<<" "<<p1_local.x<<" "<<p1_local.y<<" "<<euclidean_distance<<" "<<tracking_points[k].time<<endl;

					tracking_file.close();
				}
			
				// End Store comparison between projected vision and groundtruth of CARLA //

				// Visualize arrow to show tracking

				if (tracking_points_prev.size() > 0)
				{
					double t0;

					int flag_prev = 0;

					for (int j=0; j<tracking_points_prev.size(); j++)
					{
						if (tracking_points_prev[j].object_id == object_id)
						{
							flag_prev = 1;
							p0.x = tracking_points_prev[j].global_centroid_x;
							p0.y = tracking_points_prev[j].global_centroid_y;
							p0.z = 0;
							t0 = tracking_points_prev[j].time;

							if (tracking_points_prev[j].yaw_samples.size() == Number_of_Tracking_Samples)
							{
								yaw_previous = tracking_points_prev[j].average_samples; 
							}
							else
							{
								yaw_previous = tracking_points_prev[j].yaw_previous;
							}

							break;
						}
					}

					if (flag == 1 && flag_prev == 1) // Both tracking_points structures have the same object, so we can add an arrow with the estimated velocity vector
					{
						visualization_msgs::Marker vel_points;

						vel_points.header.frame_id = "/map"; // map == global coordinates. Base_link == local coordinates
						vel_points.header.stamp = ros::Time::now();
						vel_points.ns = "map_manager_visualization";
						vel_points.action = visualization_msgs::Marker::ADD;
						vel_points.type = visualization_msgs::Marker::ARROW;
					 	vel_points.id = object_id;

						vel_points.points.clear();

						float vel_x, vel_y, vel_lin;

						// Global velocities

						vel_x = (p1.x - p0.x)/(t1 - t0); 
						vel_y = (p1.y - p0.y)/(t1 - t0); 

						/*cout<<endl<<endl<<"Points x: "<<p1.x<<" "<<p0.x<<endl;
						cout<<"Points y: "<<p1.y<<" "<<p0.y<<endl;
						cout<<"Times: "<<t1<<" "<<t0<<" "<<t1-t0<<endl;
						cout<<"Velocities: "<<vel_x<<" "<<vel_y<<endl;*/

						if (vel_x == 0 && vel_y == 0)
						{
							yaw = 0;
						}
						else
						{
							if (vel_x != 0)
							{

								yaw=atan2(vel_y, vel_x);
							}
							else
							{
								if (vel_y > 0) 
								{
									yaw = 0;
								}
								else
								{
									yaw = M_PI;
								}
							}
						}

						vel_lin = sqrt(pow(vel_x,2)+pow(vel_y,2)); // m/s

						tracking_points[k].yaw_previous = yaw;

						// Solve this section if you want to smooth the yaw angle with respect to previous samples
						// The problem is that the vision introduces many errors with the 3D projection

						/*if ((tracking_points[k].yaw_samples.size() == Number_of_Tracking_Samples) && (abs(yaw) - abs(yaw_previous) > M_PI/8) || (abs(yaw) - abs(yaw_previous) < -M_PI/8) || isnan(float(yaw)) || isnan(float(vel_lin))) // Carla pedestrians in m/s
						{
							yaw = yaw_previous;
						}*/

						// Solve this section if you want to smooth the yaw angle with respect to previous samples

						// Push yaw samples in the buffer

						if (tracking_points[k].yaw_samples.size() < Number_of_Tracking_Samples) // Previously identified object
						{
							tracking_points[k].yaw_samples.push_back(yaw);
						}
						else
						{
							int size = tracking_points[k].yaw_samples.size();

							for (int t=0; t<tracking_points[k].yaw_samples.size()-1; t++)
							{
								tracking_points[k].yaw_samples[t] = tracking_points[k].yaw_samples[t+1];	
							}
	
							tracking_points[k].yaw_samples[size-1] = yaw;
						}

						// Calculate average yaw angle if buffer is full

						if (tracking_points[k].yaw_samples.size() == Number_of_Tracking_Samples)
						{
							tracking_points[k].average_samples = 0;

							//cout<<"Yaw samples: ";

							for (int l=0; l<tracking_points[k].yaw_samples.size(); l++)
							{
								tracking_points[k].average_samples += tracking_points[k].yaw_samples[l];
								//cout<<tracking_points[k].yaw_samples[l]*180/M_PI<<" ";	
							}

							tracking_points[k].average_samples = tracking_points[k].average_samples/tracking_points[k].yaw_samples.size();

							//cout<<endl<<"Average yaw sample: "<<tracking_points[k].average_samples*180/M_PI;
						}

						if (vel_lin < 30 && vel_lin > 0) // To avoid problems with scale 0 in RVIZ
						{
							vel_points.pose.position.x = p1.x;
							vel_points.pose.position.y = p1.y;
							vel_points.pose.position.z = p1.z;
							//vel_points.color= colours[object_id % colours.size()];
							vel_points.color = yolo_msg->yolo_list[i].color;
							vel_points.scale.x = (sqrt(pow(vel_x,2)+pow(vel_y,2)));
							vel_points.scale.y = 0.15;
							vel_points.scale.z = 0.15;
							vel_points.lifetime = ros::Duration(0.25);
							vel_points.pose.orientation=tf::createQuaternionMsgFromRollPitchYaw(0,0,yaw);

							cout<<endl<<"Object ID: "<<int(yolo_msg->yolo_list[i].object_id)<<" "<<"Vel_lin_km_h: "<<vel_lin*3.6<<" "<<"Yaw_angle_degrees: "<<yaw*180/M_PI;

							tracked_objects_vel_marker_pub.publish(vel_points);	
						}	
					}
				}
			}
		}
	}

	tracking_points_prev.clear();

	for (int j=0; j<tracking_points.size(); j++)
	{
		tracking_points_prev.push_back(tracking_points[j]);
	}

	cout<<endl<<"--------------------VISION END--------------------"<<endl;
	cout<<endl<<"--------------------MERGED LIDAR START--------------------"<<endl;

	// Merged objects (YOLOv3 + DeepSort + Centernet assigned colour)
	
	for (int i=0; i<merged_objects.size(); i++)
	{
		cout<<endl<<"Merged centroid and type: "<<i<<" X "<<merged_objects[i].centroid_x<<" Y "<<merged_objects[i].centroid_y<<" Type "<<merged_objects[i].type<<endl;

		r = merged_objects[i].r; // Over 1 to visualize the cube
		g = merged_objects[i].g; 
		b = merged_objects[i].b; 

		ss6 <<"Merged clusters"; 

		float xmin = merged_objects[i].x_min;
		float xmax = merged_objects[i].x_max;
		float ymin = merged_objects[i].y_min;
		float ymax = merged_objects[i].y_max;
		float zmin = merged_objects[i].z_min;
		float zmax = merged_objects[i].z_max;

		if (VIEWER_3D)
		{
			viewer->addCube(xmin,xmax,ymin,ymax,zmin,zmax,r,g,b,ss6.str() );
		}

		r = merged_objects[i].r*255; // Over 255 to visualize the pointcloud
		g = merged_objects[i].g*255;
		b = merged_objects[i].b*255;

		ss7 << "Point Cloud Merged objects"; // Each PCL must have a different stringstream

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_object_cloud (new  pcl::PointCloud<pcl::PointXYZRGB>);
		merged_object_cloud = merged_objects[i].cloud;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_handler (merged_object_cloud,r,g,b); // For ColorHandlerCustom, the r,g,b MUST NOT have by divided by 255

		if (VIEWER_3D)
		{
			viewer->addPointCloud<pcl::PointXYZRGB> (merged_object_cloud,color_handler, ss7.str());

			pcl::PointXYZRGB text;
			text.x = merged_objects[i].centroid_x;
			text.y = merged_objects[i].centroid_y;
			text.z = merged_objects[i].centroid_y + 0.3;

			stringstream ss_text;
			ss_text << merged_objects[i].type <<" "<<merged_objects[i].object_id;

			viewer->addText3D(ss_text.str(), text, 0.3, 1.0, 1.0, 1.0, ss_text.str()); 
		}

		// Visualize object in RVIZ

		geometry_msgs::Point32 p0,p1, p1_local;
		double t1;

		int object_id = merged_objects[i].object_id;
		double yaw, yaw_previous;

		visualization_msgs::Marker obstacle_points;

		obstacle_points.header.frame_id = "/map"; // map == global coordinates. Base_link == local coordinates
		obstacle_points.header.stamp = ros::Time::now();
		obstacle_points.ns = "map_manager_visualization";
		obstacle_points.action = visualization_msgs::Marker::ADD;
		obstacle_points.type = visualization_msgs::Marker::SPHERE;
	 	obstacle_points.id = object_id;

		obstacle_points.points.clear();

		obstacle_points.color = colours[1]; // Only green;
		obstacle_points.scale.x = 0.25;
		obstacle_points.scale.y = 0.25;
		obstacle_points.scale.z = 0.25;
		obstacle_points.lifetime = ros::Duration(0.25);

		int flag = 0;

		for (int k=0; k<tracking_points_lidar.size(); k++)
		{
			if (tracking_points_lidar[k].object_id == object_id)
			{
				flag = 1;
				p1.x = tracking_points_lidar[k].global_centroid_x;
				p1.y = tracking_points_lidar[k].global_centroid_y;
				p1.z = 0;
				p1_local.x = tracking_points_lidar[k].local_centroid_x;
				p1_local.y = tracking_points_lidar[k].local_centroid_y;
				p1_local.z = 0;
				t1 = tracking_points_lidar[k].time;

				break;
			}
		}

		obstacle_points.pose.position.x = p1.x;
		obstacle_points.pose.position.y = p1.y;
		obstacle_points.pose.position.z = p1.z;

		tracked_merged_objects_marker_pub.publish(obstacle_points);

		if (flag == 1)
		{
			float euclidean_distance, diff_y, diff_x;
			int flag_carla_object = 0;

			int t;

			for (t=0; t<carla_msg->carlaobjectlocationlist.size(); t++)
			{
				cout<<endl<<"Tracking point lidar carla id: "<<tracking_points_lidar[k].carla_id;
				if (int(tracking_points_lidar[k].carla_id) == int(carla_msg->carlaobjectlocationlist[t].object_id))
				{
					flag_carla_object = 1;
					break;
				}
			}

			// Store comparison among merged objects and CARLA groundtruth //

			if (flag_carla_object == 1)
			{
				int merged_id = -2; // To know that this .txt line belongs to Merged objects-CARLA comparison

				geometry_msgs::PointStamped global_centroid;
				geometry_msgs::Point32 local_centroid;
				global_centroid.point.x = float(carla_msg->carlaobjectlocationlist[t].pos_x);
				global_centroid.point.y = float(-carla_msg->carlaobjectlocationlist[t].pos_y); // The Y-axis sign is the opposite in CARLA
				global_centroid.point.z = float(carla_msg->carlaobjectlocationlist[t].pos_z);
				local_centroid = Global_To_Local_Coordinates(global_centroid);

				cout<<endl<<"Puntos a escribir LiDAR: "<<p1_local.x<<" "<<p1_local.y<<" "<<local_centroid.x<<" "<<local_centroid.y<<endl;

				diff_x = float(p1_local.x) - float(local_centroid.x);
				diff_y = float(p1_local.y) - float(local_centroid.y); 

				euclidean_distance = float(sqrt(pow(diff_x,2)+pow(diff_y,2)));

				string sub_path ("/home/robesafe/compartido_con_docker/Nuevos_Ficheros_CGH/tracking_results/a_la_vez/");

				// Invidual object tracking
				/*
				string a ("vision_object_");
				auto b = to_string(int(tracking_points[k].object_id)); // VOT ID, not CARLA ID
				string c (".txt");
				string filename = a+b+c;
				*/
			
				// All tracked objects
				string filename ("tracked_objects.txt");
				string path = sub_path+filename;
		
				ofstream tracking_file;

				tracking_file.open(path, ios::app);

				int carla_size = int(carla_msg->carlaobjectlocationlist.size());

				tracking_file<<merged_id<<" "<<yolo_msg->yolo_list.size()<<" "<<carla_msg->carlaobjectlocationlist.size()<<" "<<tracking_points_lidar[k].object_id<<" "<<tracking_points_lidar[k].carla_id<<" "<<local_centroid.x<<" "<<local_centroid.y<<" "<<p1_local.x<<" "<<p1_local.y<<" "<<euclidean_distance<<" "<<tracking_points_lidar[k].time<<endl;

				tracking_file.close();
			}

			// End store comparison among merged objects and CARLA groundtruth //

			// Visualize arrow to show tracking

			if (tracking_points_prev_lidar.size() > 0)
			{
				double t0;

				int flag_prev = 0;

				for (int j=0; j<tracking_points_prev_lidar.size(); j++)
				{
					if (tracking_points_prev_lidar[j].object_id == object_id)
					{
						flag_prev = 1;
						p0.x = tracking_points_prev_lidar[j].global_centroid_x;
						p0.y = tracking_points_prev_lidar[j].global_centroid_y;
						p0.z = 0;
						t0 = tracking_points_prev_lidar[j].time;
						//yaw_previous = tracking_points_prev[j].average_samples; 
						yaw_previous = tracking_points_prev_lidar[j].yaw_previous;
						break;
					}
				}

				if (flag == 1 && flag_prev == 1) // Both tracking_points structures have the same object, so we can add an arrow with the estimated velocity vector
				{
					visualization_msgs::Marker vel_points;

					vel_points.header.frame_id = "/map"; // map == global coordinates. Base_link == local coordinates
					vel_points.header.stamp = ros::Time::now();
					vel_points.ns = "map_manager_visualization";
					vel_points.action = visualization_msgs::Marker::ADD;
					vel_points.type = visualization_msgs::Marker::ARROW;
				 	vel_points.id = object_id;

					vel_points.points.clear();

					float vel_x, vel_y, vel_lin;

					// Global velocities

					vel_x = (p1.x - p0.x)/(t1 - t0); 
					vel_y = (p1.y - p0.y)/(t1 - t0); 

					cout<<endl<<"Points x: "<<p1.x<<" "<<p0.x<<endl;
					cout<<"Points y: "<<p1.y<<" "<<p0.y<<endl;
					cout<<"Times: "<<t1<<" "<<t0<<" "<<t1-t0<<endl;
					cout<<"Velocities: "<<vel_x<<" "<<vel_y;

					if (vel_x == 0 && vel_y == 0)
					{
						yaw = 0;
					}
					else
					{
						if (vel_x != 0)
						{

							yaw=atan2(vel_y, vel_x);
						}
						else
						{
							if (vel_y > 0) 
							{
								yaw = 0;
							}
							else
							{
								yaw = M_PI;
							}
						}
					}

					vel_lin = sqrt(pow(vel_x,2)+pow(vel_y,2)); // m/s

					if (vel_lin < 30)
					{
						vel_points.pose.position.x = p1.x;
						vel_points.pose.position.y = p1.y;
						vel_points.pose.position.z = p1.z;
						//vel_points.color= colours[object_id % colours.size()];
						vel_points.color = colours[1]; // Only Green
						vel_points.scale.x = (sqrt(pow(vel_x,2)+pow(vel_y,2)));
						vel_points.scale.y = 0.15;
						vel_points.scale.z = 0.15;
						vel_points.lifetime = ros::Duration(0.25);
						vel_points.pose.orientation=tf::createQuaternionMsgFromRollPitchYaw(0,0,yaw);

						cout<<endl<<"Object ID: "<<int(merged_objects[i].object_id)<<" "<<"Vel_lin_km_h: "<<vel_lin*3.6<<" "<<"Yaw_angle_degrees: "<<yaw*180/M_PI;

						tracked_merged_objects_vel_marker_pub.publish(vel_points);	
					}	
				}
			}
		}
	}

	tracking_points_prev_lidar.clear();

	for (int j=0; j<tracking_points_lidar.size(); j++)
	{
		tracking_points_prev_lidar.push_back(tracking_points_lidar[j]);
	}
	cout<<endl<<"--------------------MERGED LIDAR END--------------------"<<endl;			
}

// End Functions and CallBacks ///


void Procesar_carla_cb(const carla_msgs::CarlaObjectLocation::ConstPtr& carla_msg)
{

cout << "Procesar CARLA" << endl;

}

// Main ROS Function //

int main (int argc, char** argv)
{
	// Initialize ROS

	ros::init (argc, argv, "objectdetection_lidar");
	ros::NodeHandle nh;
 
	// Lat and Lon by parameters

    	nh.param<double>("/lat_origin",lat_origin,40.5126566);
    	nh.param<double>("/lon_origin",lon_origin,-3.34460735);

    	// Initialize map origin

    	geographic_msgs::GeoPoint geo_origin;
    	geo_origin.latitude = lat_origin;
    	geo_origin.longitude = lon_origin;
    	geo_origin.altitude = 0;
	utmOrigin = geodesy::UTMPoint(geo_origin);

	// Transform listener

	listener = new tf::TransformListener(ros::Duration(5.0));
 
	// Publishers //

	obstaclelidarPub= nh.advertise<sec_msgs::ObstacleArray>("/obstacles", 1, true);
	frontCarPub= nh.advertise<sec_msgs::Obstacle>("/frontCarCurrentLane", 1, true);
	pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_pointcloud_colored_transformed",1);
	pointcloud_only_laser_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_local_filter_on_floor",1);
	pointcloud_nubecolor_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_pointcloud_nubecolor",1);
	pubPedestrian = nh.advertise<std_msgs::Bool>("/pedestrian",1);
	pubMerge = nh.advertise<std_msgs::Bool>("/safeMerge",1);
	pubLaneChange = nh.advertise<std_msgs::Bool>("/safeLaneChange",1);
	pubDistOvertake = nh.advertise<sec_msgs::Distance>("/distOvertake", 1);

	// Tracking publishers

	tracked_objects_vel_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_objects_marker_vel", 1, true); // Only vision
	tracked_objects_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_objects_marker", 1, true); // Only vision
	tracked_merged_objects_vel_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_merged_objects_marker_vel", 1, true); // Sensory fusion LiDAR and Camera
	tracked_merged_objects_marker_pub = nh.advertise<visualization_msgs::Marker>("/tracked_merged_objects_marker", 1, true); // Sensory fusion LiDAR and Camera
	
	// End Publishers //

 	// Subscribers

 	message_filters::Subscriber<sec_msgs::RegElem> regelem_sub_; // Regulatory elements of current monitorized lanelets
	message_filters::Subscriber<sec_msgs::Route> regelemLanelet_sub_; // Monitorized lanelets
	message_filters::Subscriber<sec_msgs::Distance> regelemDist_sub_; // Distance to regulatory elements
	message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_; // Coloured LiDAR pointcloud
	message_filters::Subscriber<sensor_msgs::PointCloud2> velodyne_cloud_sub_; // LiDAR pointcloud
	message_filters::Subscriber<nav_msgs::Odometry> odom_sub_; // Odometry
	message_filters::Subscriber<yolov3_centernet_ros::yolo_list> yolo_sub_; // Detection and Tracking with camera (CenterNet + Deep Sort + YOLO)
        message_filters::Subscriber<carla_msgs::CarlaObjectLocationList> carla_sub_; 

	regelem_sub_.subscribe(nh, "/currentRegElem", 1);
	regelemLanelet_sub_.subscribe(nh, "/monitorizedLanelets", 1);
	cloud_sub_.subscribe(nh, "/pcl_coloring_real/velodyne_coloured", 1);
	velodyne_cloud_sub_.subscribe(nh, "/velodyne_points", 1);
	odom_sub_.subscribe(nh, "/odom", 1);
	waiting_sub_ = nh.subscribe<std_msgs::Empty>("/waitingAtStop", 1, &waitingCallBack);
	route_sub = nh.subscribe<sec_msgs::Route>("/route", 1, &route_cb);
	yolo_sub_.subscribe(nh, "/yolov3_tracking_list", 1);
        carla_sub_.subscribe(nh, "/carla/hero/location_list", 1); 

	//carla_sub_ = nh.subscribe<carla_msgs::CarlaObjectLocation>("/carla/hero/location", 1, &Procesar_carla_cb); 

        //yolo_sub_ = nh.subscribe<nav_msgs::Path>("/yolov3_tracking_list", 1, &Procesar_yolo_cb);

	// Callbacks

	// Callback 1: Synchonize monitorized lanelets and current regulatory element (Exact time)

	typedef message_filters::sync_policies::ExactTime<sec_msgs::RegElem, sec_msgs::Route> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync_(MySyncPolicy(10), regelem_sub_, regelemLanelet_sub_);
	sync_.registerCallback(boost::bind(&regelement_cb, _1, _2));

	// Callback 2: Synchronize LiDAR pointcloud and camera information (including detection and tracking) (Approximate time)

	// CARLA simulator
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, yolov3_centernet_ros::yolo_list, carla_msgs::CarlaObjectLocationList> MySyncPolicy2;
	message_filters::Synchronizer<MySyncPolicy2> sync2_(MySyncPolicy2(200), velodyne_cloud_sub_, yolo_sub_, carla_sub_);
	sync2_.registerCallback(boost::bind(&tracking_lidar_camera, _1, _2, _3));

	// Real-world
        /*typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, yolov3_centernet_ros::yolo_list> MySyncPolicy2;
	message_filters::Synchronizer<MySyncPolicy2> sync2_(MySyncPolicy2(40), velodyne_cloud_sub_, yolo_sub_);
	sync2_.registerCallback(boost::bind(&tracking_lidar_camera, _1, _2));*/

	// Load map

	string map_frame = "";
	string map_path = ros::package::getPath("sec_map_manager") + "/maps/uah_lanelets_v42.osm"; // Load this path if /map_path argument does not exit

        nh.param<std::string>("/map_path", map_path, map_path);
	loadedMap = std::make_shared<LaneletMap>(map_path);

	// Initialize colours

	std_msgs::ColorRGBA color;

	// 0

	color.a=1.0;
	color.r=1.0;
	color.g=0.0;
	color.b=0.0;
	colours.push_back(color);

	// 1

	color.a=1.0;
	color.r=0.0;
	color.g=1.0;
	color.b=0.0;
	colours.push_back(color);
	
	// 2

	color.a=1.0;
	color.r=0.0;
	color.g=0.0;
	color.b=1.0;
	colours.push_back(color);

	// 3

	color.a=1.0;
	color.r=1.0;
	color.g=1.0;
	color.b=0.0;
	colours.push_back(color);

	// 4

	color.a=1.0;
	color.r=1.0;
	color.g=0.0;
	color.b=1.0;
	colours.push_back(color);

	// 5

	color.a=1.0;
	color.r=0.0;
	color.g=1.0;
	color.b=1.0;
	colours.push_back(color);

	// 6

	color.a=1.0;
	color.r=0.5;
	color.g=0.0;
	color.b=0.0;
	colours.push_back(color);

	// 7

	color.a=1.0;
	color.r=0.0;
	color.g=0.5;
	color.b=0.0;
	colours.push_back(color);

	// 8

	color.a=1.0;
	color.r=0.0;
	color.g=0.0;
	color.b=0.5;
	colours.push_back(color);

	// 9

	color.a=1.0;
	color.r=0.5;
	color.g=0.5;
	color.b=0.0;
	colours.push_back(color);

	// 3D viewer configuration

	if (VIEWER_3D)
	{
		viewer->setBackgroundColor (0.0, 0.0, 0.0);
		viewer->addCoordinateSystem (1.0);
		viewer->initCameraParameters ();
		viewer->setCameraPosition (-10, 0, 5, 0.3, 0, 0.95);
	}
	 
	// ROS Spin

	ros::spin ();
}

// End main ROS Function //

	
	

	

	
	
	
