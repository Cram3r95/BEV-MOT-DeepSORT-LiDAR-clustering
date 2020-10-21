/***
Created on Thu Dec  8 16:45:21 2019

@author: Carlos Gomez-Huelamo

Code to process the fusion between LiDAR clusters and BEV (Bird's Eye View) 
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
#include <visualization_msgs/MarkerArray.h>

// BEV Tracking includes
#include "t4ac_msgs/BEV_detections_list.h"
#include "t4ac_msgs/BEV_trackers_list.h"

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

#define lanelet_filter 0

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
	float global_centroid_x; // Global centroid (with respect to the "map" frame)
	float global_centroid_y;
    float l;
    float w;
    float h;
	double orientation;
	string type;
    double time;
}Object;

typedef struct 
{
    float centroid_x; // Local centroid (with respect to the "base_link" frame)
	float centroid_y;
	float global_centroid_x; // Global centroid (with respect to the "map" frame)
	float global_centroid_y;
    float l;
    float w;
    float h;
	double orientation;
	string type;
	int object_id;
    double time;
    int pedestrian_state;
    int stop_state;
    int give_way_state;
}Tracked_Object;

// End Structures //


// ROS communication // 

// ROS Publishers

ros::Publisher pub_Tracked_Obstacles_Marker;
ros::Publisher pub_LiDAR_Obstacles_Marker;
ros::Publisher pub_VOT_Obstacles_Marker;
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
ros::Subscriber test_1_sub;
ros::Subscriber test_2_sub;
ros::Subscriber test_3_sub;

// End ROS communication //


// Global variables //

// Transform variables

tf::StampedTransform TF_map2base_link;				
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
double persistence_time=2.0; // Maximum difference in time to not delete an object

// Visualization variables

// Namespaces //

using namespace std;

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

vector<std_msgs::ColorRGBA> colours;
vector<Tracked_Object> tracked_objects;
Area_Point polygon_area[] = {0,0,
                             0,0,
                             0,0,
                             0,0};

int Number_of_sides = 4; // Of the area you have defined. Here is a rectangle, so area[] has four rows

// End Global variables


// Declarations of functions // 

// General use functions

geometry_msgs::Point32 Local_To_Global_Coordinates(geometry_msgs::PointStamped );
void Inside_Polygon(Area_Point *, int , Area_Point, bool &);

// Calbacks

void route_cb(const sec_msgs::Route::ConstPtr& );
void waiting_cb(const std_msgs::Empty );
void test_1_cb(const t4ac_msgs::BEV_trackers_list::ConstPtr& );
void test_2_cb(const t4ac_msgs::BEV_detections_list::ConstPtr& );
void test_3_cb(const nav_msgs::Odometry::ConstPtr& );
void regelement_cb(const sec_msgs::RegElem::ConstPtr& , const sec_msgs::Route::ConstPtr& );
void sensor_fusion_and_monitors_cb(const t4ac_msgs::BEV_detections_list::ConstPtr& , const t4ac_msgs::BEV_trackers_list::ConstPtr& , const nav_msgs::Odometry::ConstPtr& );

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

	pub_Detected_Pedestrian = nh.advertise<std_msgs::Bool>("/pedestrian",1);
	pub_Safe_Merge = nh.advertise<std_msgs::Bool>("/safeMerge",1);
	pub_Front_Car = nh.advertise<sec_msgs::Obstacle>("/frontCarCurrentLane", 1, true);
	pub_Front_Car_Distance = nh.advertise<std_msgs::Float64>("/frontCarCurrentLane_distance", 1, true);
	pub_Safe_Lane_Change = nh.advertise<std_msgs::Bool>("/safeLaneChange",1);
	pub_Distance_Overtake = nh.advertise<sec_msgs::Distance>("/distOvertake", 1);

	// Tracking publishers

	pub_Tracked_Obstacles_Marker = nh.advertise<visualization_msgs::MarkerArray>("/t4ac/perception/detection/Tracked_Obstacles", 1, true); // Merged (Tracked obstacles)
	pub_LiDAR_Obstacles_Marker = nh.advertise<visualization_msgs::MarkerArray>("/t4ac/perception/detection/LiDAR_Obstacles", 1, true); // Only LiDAR
    pub_VOT_Obstacles_Marker = nh.advertise<visualization_msgs::MarkerArray>("/t4ac/perception/detection/VOT_BEV_Obstacles", 1, true); // Only vision
	
	// End Publishers //

 	// Subscribers //

 	message_filters::Subscriber<sec_msgs::RegElem> regelem_sub_; // Regulatory elements of current monitorized lanelets
	message_filters::Subscriber<sec_msgs::Route> regelemLanelet_sub_; // Monitorized lanelets
	message_filters::Subscriber<sec_msgs::Distance> regelemDist_sub_; // Distance to regulatory elements
	message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_; // Coloured LiDAR point cloud
	message_filters::Subscriber<sensor_msgs::PointCloud2> velodyne_cloud_sub_; // LiDAR point cloud
	message_filters::Subscriber<nav_msgs::Odometry> ego_vehicle_pose_sub_; // Odometry
	message_filters::Subscriber<t4ac_msgs::BEV_trackers_list> projected_vot_sub_; // Detection and Tracking with camera (CenterNet + Deep Sort)
    message_filters::Subscriber<t4ac_msgs::BEV_detections_list> lidar_detections_sub_; // LiDAR detections

	regelem_sub_.subscribe(nh, "/currentRegElem", 1);
	regelemLanelet_sub_.subscribe(nh, "/monitorizedLanelets", 1);
	velodyne_cloud_sub_.subscribe(nh, "/velodyne_points", 1);
    ego_vehicle_pose_sub_.subscribe(nh, "/localization/pose", 1);
    lidar_detections_sub_.subscribe(nh, "/t4ac/perception/detection/bev_detections", 1);
	projected_vot_sub_.subscribe(nh, "/t4ac/perception/tracked_obstacles", 1);

    /*test_1_sub = nh.subscribe<t4ac_msgs::BEV_trackers_list>("/t4ac/perception/tracked_obstacles", 1, &test_1_cb);
    test_2_sub = nh.subscribe<t4ac_msgs::BEV_detections_list>("/t4ac/perception/detection/bev_detections", 1, &test_2_cb);
    test_3_sub = nh.subscribe<nav_msgs::Odometry>("/localization/pose", 1, &test_3_cb);*/

    waiting_sub = nh.subscribe<std_msgs::Empty>("/waitingAtStop", 1, &waiting_cb);
	route_sub = nh.subscribe<sec_msgs::Route>("/route", 1, &route_cb);

    // End Subscribers //

	// Callbacks //

	// Callback 1: Synchonize monitorized lanelets and current regulatory element (Exact time)

	typedef message_filters::sync_policies::ExactTime<sec_msgs::RegElem, sec_msgs::Route> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync_(MySyncPolicy(10), regelem_sub_, regelemLanelet_sub_);
	sync_.registerCallback(boost::bind(&regelement_cb, _1, _2));

	// Callback 2: Synchronize LiDAR point cloud and camera information (including detection and tracking). Evaluate monitors (Approximate time)

	typedef message_filters::sync_policies::ApproximateTime<t4ac_msgs::BEV_detections_list, t4ac_msgs::BEV_trackers_list, nav_msgs::Odometry> MySyncPolicy2;
	message_filters::Synchronizer<MySyncPolicy2> sync2_(MySyncPolicy2(100), lidar_detections_sub_, projected_vot_sub_, ego_vehicle_pose_sub_);
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

	aux2 = TF_map2base_link * aux;

	point_global.point.x = aux2.getX();
	point_global.point.y = aux2.getY();
	point_global.point.z = aux2.getZ();

	point32_global.x = point_global.point.x;
	point32_global.y = point_global.point.y;
	point32_global.z = point_global.point.z;

	return(point32_global);
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

void test_1_cb(const t4ac_msgs::BEV_trackers_list::ConstPtr& msg)
{
    double time = msg->header.stamp.toSec();
	cout << "VOT projected: " << time << endl;
}

void test_2_cb(const t4ac_msgs::BEV_detections_list::ConstPtr& msg)
{
    double time = msg->header.stamp.toSec();
	cout << "LiDAR detections: " << time << endl;
}

void test_3_cb(const nav_msgs::Odometry::ConstPtr& msg)
{
    double time = msg->header.stamp.toSec();
	cout << "Pose: " << time << endl;
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

void sensor_fusion_and_monitors_cb(const t4ac_msgs::BEV_detections_list::ConstPtr& lidar_detections_msg, const t4ac_msgs::BEV_trackers_list::ConstPtr& bev_trackers_list_msg, const nav_msgs::Odometry::ConstPtr& ego_vehicle_pose_msg)
{
	cout<<"------------------------------------------------"<<endl;
	// ROS_INFO("Time: [%lf]", (double)ros::Time::now().toSec());

    // lidar_detections_msg contains the LiDAR detections
	// bev_trackers_list_msg contains the visual tracked obstacles projected onto the Bird's Eye View space
	// ego_vehicle_pose_msg contains the ego vehicle position

	// Note that if --clock is not published (if we are trying to run a rosbag), the system will not create the transforms

	// Get odom and velocity information

	// Obtain the movement of the ego-vehicle in X and Y (Global coordinates) and Orientation (Yaw)

	double displacement_x_global = ego_vehicle_pose_msg->pose.pose.position.x - previous_odom.pose.pose.position.x;
	double displacement_y_global = ego_vehicle_pose_msg->pose.pose.position.y - previous_odom.pose.pose.position.y;
	double yaw = tf::getYaw(ego_vehicle_pose_msg->pose.pose.orientation);

	// Obtain displacement of the ego-vehicle and Velocities in Local coordinates

	double displacement_x_local = displacement_x_global*cos(yaw) + displacement_y_global*sin(yaw);
	double displacement_y_local = displacement_x_global*(-sin(yaw)) + displacement_y_global*cos(yaw);

	double time = ego_vehicle_pose_msg->header.stamp.toSec() - previous_odom.header.stamp.toSec();

	double vel_x_with_yaw = displacement_x_local/time;
	double vel_y_with_yaw = displacement_y_local/time;
	double abs_vel = sqrt(pow(vel_x_with_yaw,2)+pow(vel_y_with_yaw,2));

	double vel_x = displacement_x_global/time;
	double vel_y = displacement_y_global/time;
	
	// Store odom in different formats: TODO: Required?

	geodesy::UTMPoint odomUTMmsg;
	odomUTMmsg.band = utm_origin.band;
	odomUTMmsg.zone = utm_origin.zone;
	odomUTMmsg.altitude = 0;
	odomUTMmsg.easting = ego_vehicle_pose_msg->pose.pose.position.x + utm_origin.easting;
	odomUTMmsg.northing = ego_vehicle_pose_msg->pose.pose.position.y + utm_origin.northing;
 	geographic_msgs::GeoPoint latLonOdom;
	latLonOdom = geodesy::toMsg(odomUTMmsg);

	// Store previous odometry

	previous_odom = *ego_vehicle_pose_msg;

	vector<Object> laser_objects;
	int number_vehicles = 0;
    int number_pedestrians = 0;

    for (int i = 0; i < lidar_detections_msg->bev_detections_list.size(); i++)
	{
		Object object;
	
		object.centroid_x = -lidar_detections_msg->bev_detections_list[i].y;
        object.centroid_y = -lidar_detections_msg->bev_detections_list[i].x;

		geometry_msgs::PointStamped  centroid;
		geometry_msgs::Point32 global_centroid;

		centroid.point.x = object.centroid_x;
		centroid.point.y = object.centroid_y;
		centroid.point.z = 0;

		global_centroid = Local_To_Global_Coordinates(centroid);

		object.global_centroid_x = global_centroid.x;
		object.global_centroid_y = global_centroid.y;

		object.l = lidar_detections_msg->bev_detections_list[i].l;
        object.w = lidar_detections_msg->bev_detections_list[i].w; 
        object.h = 1.7; // Height of the obstacle. TODO: Take from the 3D object detector

		object.orientation = lidar_detections_msg->bev_detections_list[i].o;

		string type = lidar_detections_msg->bev_detections_list[i].type;

		if (!strcmp(type.c_str(),"1"))
		{
			object.type = "car";
			number_vehicles++;
		}
		else if (!strcmp(type.c_str(),"2"))
		{
			object.type = "pedestrian";
			number_pedestrians++;
		}

		object.time = lidar_detections_msg->header.stamp.toSec();

		laser_objects.push_back(object);
	}

	visualization_msgs::MarkerArray obstacles_array;
	int id = 0;

	for (int i = 0; i < laser_objects.size(); i++)
	{
		visualization_msgs::Marker obstacle_points;

		obstacle_points.header.frame_id = "/base_link"; // map == global coordinates. Base_link == local coordinates
		obstacle_points.header.stamp = lidar_detections_msg->header.stamp;
		obstacle_points.ns = "map_manager_visualization";
		obstacle_points.action = visualization_msgs::Marker::ADD;
		obstacle_points.type = visualization_msgs::Marker::CUBE;
		obstacle_points.id = id;
		id++;
		obstacle_points.points.clear();

		obstacle_points.color = colours[2]; // Only red;
		obstacle_points.scale.x = 1;
		obstacle_points.scale.y = 1;
		obstacle_points.scale.z = 1;
		obstacle_points.lifetime = ros::Duration(0.40);

		/*obstacle_points.pose.position.x = laser_objects[i].global_centroid_x;
		obstacle_points.pose.position.y = laser_objects[i].global_centroid_y;
		obstacle_points.pose.position.z = laser_objects[i].global_centroid_z;*/

        obstacle_points.pose.position.x = laser_objects[i].centroid_x;
		obstacle_points.pose.position.y = laser_objects[i].centroid_y;
		obstacle_points.pose.position.z = 0;

		obstacles_array.markers.push_back(obstacle_points);
	}

	pub_LiDAR_Obstacles_Marker.publish(obstacles_array);

    // BEV Projected VOT (Visual Object Tracking)
    
    float diff_lidar_vot = 0;
    int object_id = 0;

	// Publish

	obstacles_array.markers.clear();

	for (int i = 0; i < bev_trackers_list_msg->bev_trackers_list.size(); i++)
	{
		visualization_msgs::Marker obstacle_points;

		obstacle_points.header.frame_id = "/base_link"; // map == global coordinates. Base_link == local coordinates
		obstacle_points.header.stamp = bev_trackers_list_msg->header.stamp;
		obstacle_points.ns = "map_manager_visualization";
		obstacle_points.action = visualization_msgs::Marker::ADD;
		obstacle_points.type = visualization_msgs::Marker::CUBE;
	 	obstacle_points.id = bev_trackers_list_msg->bev_trackers_list[i].object_id;

		obstacle_points.points.clear();

		obstacle_points.color = colours[0]; // Only red;
		obstacle_points.scale.x = 1;
		obstacle_points.scale.y = 1;
		obstacle_points.scale.z = 1;
		obstacle_points.lifetime = ros::Duration(0.40);

		/*obstacle_points.pose.position.x = laser_objects[i].global_centroid_x;
		obstacle_points.pose.position.y = laser_objects[i].global_centroid_y;
		obstacle_points.pose.position.z = laser_objects[i].global_centroid_z;*/

        obstacle_points.pose.position.x = bev_trackers_list_msg->bev_trackers_list[i].x;
		obstacle_points.pose.position.y = bev_trackers_list_msg->bev_trackers_list[i].y;
		obstacle_points.pose.position.z = 0;

		obstacles_array.markers.push_back(obstacle_points);
	}

	pub_VOT_Obstacles_Marker.publish(obstacles_array);

	double vot_time = bev_trackers_list_msg->header.stamp.toSec();

    for (int i=0; i<bev_trackers_list_msg->bev_trackers_list.size(); i++)
    {
        float max_diff_lidar_vot = 4; // Initialize maximum allowed difference
        int index_most_similar = -1;

        float vot_x = float(bev_trackers_list_msg->bev_trackers_list[i].x);
        float vot_y = float(bev_trackers_list_msg->bev_trackers_list[i].y);

        geometry_msgs::PointStamped  centroid;
        geometry_msgs::Point32 global_centroid;

		cout << "Type: " << bev_trackers_list_msg->bev_trackers_list[i].type << endl;
        if (laser_objects.size() > 0 && (!strcmp(bev_trackers_list_msg->bev_trackers_list[i].type.c_str(),"car") || !strcmp(bev_trackers_list_msg->bev_trackers_list[i].type.c_str(),"person")))
        {
            object_id = bev_trackers_list_msg->bev_trackers_list[i].object_id;
			cout << "VOT id: " << object_id;
            geometry_msgs::PointStamped centroid;
			geometry_msgs::Point32 global_centroid;

			centroid.point.x = vot_x;
			centroid.point.y = vot_y;
			centroid.point.z = 0;

			global_centroid = Local_To_Global_Coordinates(centroid);

            for (int j=0; j<laser_objects.size(); j++)
            {
                float l_x = float(laser_objects[j].centroid_x); 
				float l_y = float(laser_objects[j].centroid_y);

				/*cout << "LiDAR local x: " << l_x << endl;
                cout << "LiDAR local y: " << l_y << endl;
				cout << "VOT local x: " << vot_x << endl;
				cout << "VOT local y: " << vot_y << endl << endl;*/
                diff_lidar_vot = float(sqrt(pow(vot_x-l_x,2)+pow(vot_y-l_y,2))); 

                if (diff_lidar_vot < max_diff_lidar_vot) // Find the closest cluster
				{
					max_diff_lidar_vot = diff_lidar_vot;
					index_most_similar = j;
				}
            }
			//cout << "Max diff lidar vot: " << max_diff_lidar_vot << endl;
            if (max_diff_lidar_vot < 2.0 && index_most_similar != -1)
            // In order to merge both information, the centroid between distance must be less that 1.5 m (VOT projected centroid and closest LiDAR centroid)
            {
                int flag = 0;

                // 1. Find out if current merged object was previously stored. Id does, update the object

                for (int k=0; k<tracked_objects.size(); k++)
                {
					cout << "ID tracked: " << tracked_objects[k].object_id << endl;
                    if (tracked_objects[k].object_id == object_id)
                    {
						tracked_objects[k].centroid_x = laser_objects[index_most_similar].centroid_x;
						tracked_objects[k].centroid_y = laser_objects[index_most_similar].centroid_y;
						tracked_objects[k].global_centroid_x = laser_objects[index_most_similar].global_centroid_x;
						tracked_objects[k].global_centroid_y = laser_objects[index_most_similar].global_centroid_y;

						tracked_objects[k].l = laser_objects[index_most_similar].l;
						tracked_objects[k].w = laser_objects[index_most_similar].w;
						tracked_objects[k].h = laser_objects[index_most_similar].h;
						tracked_objects[k].orientation = laser_objects[index_most_similar].orientation;
	
						tracked_objects[k].time = laser_objects[index_most_similar].time;

						flag = 1;
						break;
                    }
                }

                // 2. If it was not previously stored, then create a new object
	
				if (flag == 0)
				{
					Tracked_Object tracked_object;

					tracked_object.centroid_x = laser_objects[index_most_similar].centroid_x;
					tracked_object.centroid_y = laser_objects[index_most_similar].centroid_y;
					tracked_object.global_centroid_x = laser_objects[index_most_similar].global_centroid_x;
					tracked_object.global_centroid_y = laser_objects[index_most_similar].global_centroid_y;

					tracked_object.l = laser_objects[index_most_similar].l;
					tracked_object.w = laser_objects[index_most_similar].w;
					tracked_object.h = laser_objects[index_most_similar].h;
					tracked_object.orientation = laser_objects[index_most_similar].orientation;

					tracked_object.type = laser_objects[index_most_similar].type;
					tracked_object.object_id = object_id;
					tracked_object.time = laser_objects[index_most_similar].time;

					tracked_objects.push_back(tracked_object);
                }
            }
        }
    }

	cout<<"\nLaser objects: "<<lidar_detections_msg->bev_detections_list.size();
    cout<<"\nVOT objects: "<<bev_trackers_list_msg->bev_trackers_list.size();
    cout<<"\nTracked objects: "<<tracked_objects.size()<<endl<<endl;

	// Update tracked objects

	if (tracked_objects.size() > 0 && bev_trackers_list_msg->bev_trackers_list.size() > 0)
	{
		vector<Tracked_Object> tracked_objects_aux(tracked_objects);

		// 1. Store in auxiliar buffer
		/*
		for (int n=0; n<tracking_objects.size(); n++)
		{
			tracking_points_aux_lidar.push_back(tracking_points_lidar[n]);
		}*/

		// 2. Erase: If the difference between last sample time and current time is higher than three seconds, erase this element

		for (int t=0; t<tracked_objects.size(); t++)
		{
			if (tracked_objects[t].time + persistence_time < vot_time)
			{
				tracked_objects_aux.erase(tracked_objects_aux.begin()+t);
			}
		}
		
		// 3. Update

		tracked_objects.clear();
		
		for (int p=0; p<tracked_objects_aux.size(); p++)
		{
			tracked_objects.push_back(tracked_objects_aux[p]);
		}
	}

	// Publish merged obstacles 

	obstacles_array.markers.clear();

	for (int i = 0; i < tracked_objects.size(); i++)
	{
		visualization_msgs::Marker obstacle_points;

		obstacle_points.header.frame_id = "/base_link"; // map == global coordinates. Base_link == local coordinates
		obstacle_points.header.stamp = bev_trackers_list_msg->header.stamp;
		obstacle_points.ns = "map_manager_visualization";
		obstacle_points.action = visualization_msgs::Marker::ADD;
		obstacle_points.type = visualization_msgs::Marker::CUBE;
	 	obstacle_points.id = tracked_objects[i].object_id;

		obstacle_points.points.clear();

		obstacle_points.color = colours[1]; // Only green;
		obstacle_points.scale.x = 1;
		obstacle_points.scale.y = 1;
		obstacle_points.scale.z = 1;
		obstacle_points.lifetime = ros::Duration(0.40);

        obstacle_points.pose.position.x = tracked_objects[i].centroid_x;
		obstacle_points.pose.position.y = tracked_objects[i].centroid_y;
		obstacle_points.pose.position.z = 0;

		obstacles_array.markers.push_back(obstacle_points);
	}

	pub_Tracked_Obstacles_Marker.publish(obstacles_array);

    // Monitors //
/*
	Obstacles.obstacles.clear();

	// Auxiliar variables for obstacles

	geometry_msgs::PointStamped point_local, point_global, v1, v2, v3, v4, v1_global, v2_global, v3_global, v4_global; // v-i = Vertice of the BEV bounding box
	geometry_msgs::Point32 point32_global;
	
	point_local.header.frame_id = "/base_link";
	point_local.header.stamp = lidar_detections_msg->header.stamp;

	// Auxiliar variables for monitors

	bool pedestrian_detection = false;
	sec_msgs::Obstacle front_car; 
	string current_type = "none";
    std_msgs::Float64 ACC_distance;
    double distance_to_front_car = 5000000;
	double distance_overtake = 0;

	// Object evaluation For each detected cluster, regardingless if the obstacle is in the current lanelet //

	for (unsigned int i=0; i<<tracked_objects.size(); i++)
	{
		point_local.point.x =  tracked_objects[i].centroid_x;
		point_local.point.y =  tracked_objects[i].centroid_y;
		point_local.point.z = 0;

		// BEV (Bird's Eye View) of Cluster

		v1 = point_local;
		v1.point.x =  tracked_objects[i].centroid_x + tracked_objects[i].w/2;
		v1.point.y =  tracked_objects[i].centroid_y - tracked_objects[i].h/2;

		v2 = point_local;
		v2.point.x =  tracked_objects[i].centroid_x + tracked_objects[i].w/2;
		v2.point.y =  tracked_objects[i].centroid_y + tracked_objects[i].h/2;

		v3 = point_local;
		v3.point.x =  tracked_objects[i].centroid_x - tracked_objects[i].w/2;
		v3.point.y =  tracked_objects[i].centroid_y + tracked_objects[i].h/2;

		v4 = point_local;
		v4.point.x =  tracked_objects[i].centroid_x - tracked_objects[i].w/2;
		v4.point.y =  tracked_objects[i].centroid_y - tracked_objects[i].h/2;

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
						int pedestrian_crossing_occupied =  tracked_objects[i].pedestrian_state;

						//pointaux.x = point32_global.x;
						//pointaux.y = point32_global.y;
						//pointaux.z = point32_global.z;
						//ObstaclesInPedestrian_Ptr->points.push_back(pointaux);

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

                         tracked_objects[i].pedestrian_state = pedestrian_crossing_occupied;		
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
						//pointaux.x = point32_global.x;
						//pointaux.y = point32_global.y;
						//pointaux.z = point32_global.z;
						//ObstaclesMerging_Ptr->points.push_back(pointaux);
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

	for (int i=0; i<<tracked_objects.size(); i++) // If at least one pedestrian is about to cross ...
	{
		if (!strcmp(tracked_objects[i].type.c_str(),"pedestrian") && (tracked_objects[i].pedestrian_state == 1 ||  tracked_objects[i].pedestrian_state == 2 ||  tracked_objects[i].pedestrian_state == 3))
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

	if (number_vehicles == 0)
	{
		distance_to_front_car = 0;
	}

	ACC_distance.data = distance_to_front_car;
	pub_Front_Car_Distance.publish(ACC_distance);

	// Overtaking monitor

	pub_Safe_Lane_Change.publish(lane_change);

	sec_msgs::Distance distance_overtake_monitor;
	distance_overtake_monitor.distance = distance_overtake;
	distance_overtake_monitor.header.frame_id = "/base_link";
	distance_overtake_monitor.header.stamp = ros::Time::now();
	pub_Distance_Overtake.publish(distance_overtake_monitor);
*/
	// End Publish the monitors //

	// End Monitors //	
}

// End Callbacks //

// End Definitions of functions //











