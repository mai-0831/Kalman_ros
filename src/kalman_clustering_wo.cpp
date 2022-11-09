#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <fstream>             
#include <sstream>
#include <cmath> //std::pow
#include <Eigen/Eigen>
#include <random>   //std::normal_distribution


const int parameter_num =4;
static std::string str_point_topic = "/points_filter";
static int min_cluster_size = 3;

std::ofstream ofs("/home/itolab-mai/HDD/Kalman_ws/Gain_w_velodyne.csv");
std::ofstream ofs_toleerance("/home/itolab-mai/HDD/Kalman_ws/tolerance_velodyne.csv");
std::ofstream ofs_sse("/home/itolab-mai/HDD/Kalman_ws/sse_velodyne.csv");

class Kalman
{
    private:
        int cycle = 0;
        int candidate_num = 0;
        int pre_cluster_size = 0;


        ros::Subscriber points_sub;
        ros::Publisher centroid_pub;
        ros::Publisher estimate_centroid_pub;
        ros::Publisher gridmap_pub_m;
        ros::Publisher gridmap_pub_e;

        nav_msgs::OccupancyGrid map_measurement;
        nav_msgs::OccupancyGrid map_estimate;
        int map_size = 200;
        double map_resolution = 0.05;
        double offset_x = 5.5;
        double offset_y = 1.5;

        // std::vector<std::array<double,3>> condidate_vv;
        std::vector<std::vector<double>> candidate_vv;

        int observed_num = 2;
        double sensor_accuracy = 0.02;
        double q_noise = 0.0001;
        double delta_t = 0.2;
        double pedestrian_speed = -0.1;

        double cluster_tolerance = 0.5;
        // std::array<double, 1> cluster_tolerance_arr{0.5};
        std::array<double, 10> cluster_tolerance_arr{0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60};

        std::vector<double> Z_vec;  // Loading data of observed value
        std::vector<std::vector<double>> Z_vv;
        Eigen::Matrix<double, parameter_num, 1> Z_matrix;   // Observed value
        Eigen::Matrix<double, parameter_num, 1> Y_matrix;  
        Eigen::Matrix<double, parameter_num,parameter_num> matrix_1;

        Eigen::Matrix<double, parameter_num, parameter_num> Q_matrix;   // Process noise of covariance
        Eigen::Matrix<double, parameter_num, parameter_num> R_matrix;   // Observed noise of covariance
        Eigen::Matrix<double, parameter_num, 1> W_matrix;               // Observed noise
        Eigen::Matrix<double, parameter_num, 1> V_matrix;               // Observed noise

        std::vector<Eigen::Matrix<double, parameter_num, 1>> X_pri_vec;  // Pri Estimate state
        std::vector<Eigen::Matrix<double, parameter_num, 1>> X_post_vec; // Post Estimate state
        std::vector<Eigen::Matrix<double, parameter_num, 1>> X_post_vec_dif; // Post Estimate state
        std::vector<std::vector<Eigen::Matrix<double, parameter_num, 1>>> X_post_vv;

        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> P_pri_vec;  // Pri Estimate uncertainty
        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> P_post_vec; // Post Estimate uncertainty
        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> P_post_vec_dif; // Post Estimate uncertainty
        std::vector<std::vector<Eigen::Matrix<double, parameter_num, parameter_num>>> P_post_vv;

        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> K_gain_vec;
        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> K_gain_vec_dif;
        std::vector<std::vector<Eigen::Matrix<double, parameter_num, parameter_num>>> K_gain_vv;

        Eigen::Matrix<double, parameter_num, parameter_num> matrix_44;
        Eigen::Matrix<double, parameter_num, 1> matrix_41;

        Eigen::Matrix<double, parameter_num, parameter_num> H_;
        Eigen::Matrix<double, parameter_num, parameter_num> H_trans;

        pcl::visualization::PCLVisualizer viewer {"Euclidian Clustering"};
        std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> clusters_vv;  //for viewer
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters_dif;

        std::vector<double> fitness_score_vec;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Kalman();
        void PointCallback(const sensor_msgs::PointCloud2Ptr& points_msg);
        void Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        void Centroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, const std::vector<pcl::PointIndices>& cluster_indices);
        void Filtering();
        void SAD();
        void Visualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        void SSE();
};


Kalman::Kalman()
{
    ros::NodeHandle nh;
    ros::NodeHandle nh_("~");
    points_sub = nh.subscribe(str_point_topic, 1, &Kalman::PointCallback, this);
    gridmap_pub_m = nh.advertise<nav_msgs::OccupancyGrid>("grid_map_m", 1, this);
    gridmap_pub_e = nh.advertise<nav_msgs::OccupancyGrid>("grid_map_e", 1, this);

    std::cout << "Size of K_gain_vv: " << K_gain_vv.size() << std::endl;


    // ofs << "q_noise, sensor_accuracy, delta_t, pedestrian_speed" << std::endl;
    // ofs << q_noise << "," << sensor_accuracy << "," << delta_t << "," << pedestrian_speed << std::endl;
    // ofs << "cycle, cluster_num, cluster_size, centroid_x, centroid_y, estimate_x, estimate_y, estimate_vx, estimate_vy" << std::endl;
    ofs << "Gain1_x, Gain1_y, Gain1_vx, Gain1_vy, Gain2_x, Gain2_y, Gain2_vx, Gain2_vy, Gain3_x, Gain3_y, Gain3_vx, Gain3_vy" << std::endl;
    ofs_toleerance << "tolerance" << std::endl;
    ofs_sse << "SSE" << std::endl;

    std::vector<int8_t> map_initial(map_size*map_size, 0); //要素数20*20, すべての要素の値0で初期化
    map_measurement.info.resolution = map_resolution;
    map_measurement.info.width = map_size;
    map_measurement.info.height = map_size;
    map_measurement.header.frame_id = "/laser";
    map_measurement.info.origin.position.x = -5.5;
    map_measurement.info.origin.position.y = -1.5;
    map_measurement.data = map_initial;
    std::cout << "map_measurement.data[0]: " << map_measurement.data[0] << std::endl;

    map_estimate.info.resolution = map_resolution;
    map_estimate.info.width = map_size;
    map_estimate.info.height = map_size;
    map_estimate.header.frame_id = "/laser";
    map_estimate.info.origin.position.x = -5.5;
    map_estimate.info.origin.position.y = -1.5;
    map_estimate.data = map_initial;
    std::cout << "map_estimate.data[0]: " << map_estimate.data[0] << std::endl;


    // Parameter
    std::cout << "Destinate the parameteres" << std::endl;
    // 1 matrix
    matrix_1 << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
    // std::cout << matrix_1 << std::endl;

    // Process noise of covariance
    Q_matrix << 0.5 * q_noise * delta_t * delta_t, 0, 0, 0,
                0, 0.5 * q_noise * delta_t * delta_t, 0, 0,
                0, 0, q_noise * delta_t, 0,
                0, 0, 0, q_noise * delta_t;
    // std::cout << "Process nose of covariance is: Q\n" << Q_matrix << std::endl;

    // Observed noise of covariance
    R_matrix << std::pow(sensor_accuracy,2), 0, 0, 0,
                0, std::pow(sensor_accuracy,2), 0, 0,
                0, 0, std::pow(sensor_accuracy,2)/delta_t, 0,
                0, 0, 0, std::pow(sensor_accuracy,2)/delta_t;
    // std::cout << "Observed noise of covariance is: R\n" << R_matrix << std::endl; 

    // Matrix transformation formula
    H_ <<   1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    H_trans = H_.transpose();
    // std::cout << "Matrix transformation formula & trans\n" << H_ << "\n\n" << H_trans << std::endl; 

    matrix_44 << 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;
    
    matrix_41 << 0,
                0,
                0,
                0;

    viewer.setBackgroundColor(0.8, 0.8, 0.8);
    viewer.addCoordinateSystem(1.0, "axis");
    viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

}

void Kalman::PointCallback(const sensor_msgs::PointCloud2Ptr& points_msg)
{
    std::cout << "\n\nIn the PointCallback!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*points_msg, cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>(cloud));

    Kalman::Clustering(tmp_cloud);
}

void Kalman::Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
{
    std::cout << "\nIn the Clustering" << std::endl;

    // kd-tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input_cloud);

    // Clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    ece.setMinClusterSize(min_cluster_size);
    ece.setMaxClusterSize(input_cloud->points.size());
    ece.setSearchMethod(tree);
    ece.setInputCloud(input_cloud);


    if (cycle == 0)
    {
        std::cout << "cluster tolerance: " << cluster_tolerance << std::endl;
        ece.setClusterTolerance(cluster_tolerance);
        ece.extract(cluster_indices);
        pre_cluster_size = cluster_indices.size();
        Kalman::Centroid(input_cloud, cluster_indices);
        Kalman::Filtering();
        clusters_dif = clusters_vv[0];
        X_post_vec_dif = X_post_vv[0];
        P_post_vec_dif = P_post_vv[0];
        std::cout << "Size of X_post_vec_dif: " << X_post_vec_dif.size() << std::endl; 
        Kalman::SSE();
        Kalman::Visualization(input_cloud);
    }else{
        for (int n = 0; n < cluster_tolerance_arr.size(); n++)
        {
            std::cout << "---Cycle" << cycle << "--------------------------------------" << std::endl;
            std::cout << "Clustering loop num: " << n << std::endl;
            // cluster_tolerance = distr_cluster(engine_cluster);
            cluster_tolerance = cluster_tolerance_arr[n];
            std::cout << "cluster tolerance: " << cluster_tolerance << std::endl;
            ece.setClusterTolerance(cluster_tolerance); // ここの変数を変更していく
            ece.extract(cluster_indices);
            std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;
            std::cout << "caididate_num: " << candidate_num << std::endl;

            if(cluster_indices.size() == pre_cluster_size)
            {
                Kalman::Centroid(input_cloud, cluster_indices);
                Kalman::Filtering();
                Kalman::SAD();
                candidate_num++;
            }else{
                fitness_score_vec.push_back(100000);
                X_post_vv.emplace_back();
                X_post_vv[candidate_num].push_back(matrix_41);
                P_post_vv.emplace_back();
                P_post_vv[candidate_num].push_back(matrix_44);
                K_gain_vv.emplace_back();
                K_gain_vv[candidate_num].push_back(matrix_44);
                pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                clusters_vv.emplace_back();
                clusters_vv[candidate_num].push_back(dummy_cloud);
            }
            cluster_indices.clear();
            // Z_vec.clear();
        }

        // fitness_socre_vecからX_post_vecとP_post_vecを決定する
        for(size_t i = 0; i < fitness_score_vec.size(); ++i)
        {
            std::cout << fitness_score_vec.at(i) << ", ";
        }
        std::cout << std::endl;
        int min_index = std::distance(fitness_score_vec.begin(), 
                                        std::min_element(fitness_score_vec.begin(), fitness_score_vec.end()));
        std::cout << "Minimum index is: " << min_index << std::endl;
        X_post_vec_dif.clear();
        P_post_vec_dif.clear();
        clusters_dif.clear();
        X_post_vec_dif = X_post_vv[min_index];
        P_post_vec_dif = P_post_vv[min_index];
        K_gain_vec_dif = K_gain_vv[min_index];
        clusters_dif = clusters_vv[min_index];

        std::cout << "Size of X_post_vv: " << X_post_vv.size() << std::endl;
        std::cout << "Size of P_post_vv: " << P_post_vv.size() << std::endl;
        std::cout << "Size of K_gain_vv: " << K_gain_vv.size() << std::endl;
        std::cout << "Size of clusters_vv:  " << clusters_vv.size() << std::endl;
        for(size_t i = 0; i < K_gain_vec_dif.size(); ++i)
        {
            Eigen::Matrix<double, parameter_num, parameter_num> K_gain_tmp;
            K_gain_tmp = K_gain_vec_dif.at(i);
            // std::cout << "K_gain_tmp: " << K_gain_tmp << std::endl;
            ofs << K_gain_tmp(0,0) << "," << K_gain_tmp(1,1) << "," << K_gain_tmp(2,2) << "," << K_gain_tmp(3,3) << ","; 
        }
        ofs << std::endl;


        pre_cluster_size = X_post_vec_dif.size();
        std::cout << "Pre cluster size is: " << pre_cluster_size << std::endl;

        Kalman::SSE();

        std::stringstream ss;
        ss << "Clustering tolerance: " << cluster_tolerance_arr[min_index] << std::endl;
        viewer.removeShape("text", 0);
        viewer.addText (ss.str(),   // Text
                        10, 10,     // Position (x, y)
                        "text",
                        0
                        );
        Kalman::Visualization(input_cloud);

        ofs_toleerance << cluster_tolerance_arr[min_index] << std::endl;

        // K_gain_vec_dif.clear();
        X_post_vv.clear();
        P_post_vv.clear();
        fitness_score_vec.clear();
        candidate_num = 0;
    }
    X_post_vv.clear();
    P_post_vv.clear();
    K_gain_vv.clear();
    input_cloud->clear();
    clusters_vv.clear();
}

void Kalman::Centroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, const std::vector<pcl::PointIndices>& cluster_indices)
{
    std::cout << "\nIn the Centroid" << std::endl;
    // Divinding
    pcl::ExtractIndices<pcl::PointXYZ> ei;
    ei.setInputCloud(input_cloud);
    ei.setNegative(false);

    for(size_t i = 0; i < cluster_indices.size(); i++)
    {
        // Extract
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointIndices::Ptr tmp_clustered_lindilces(new pcl::PointIndices);
        *tmp_clustered_lindilces = cluster_indices[i];
        ei.setIndices(tmp_clustered_lindilces);
        ei.filter(*tmp_clustered_cloud);

        // Centroid
        Eigen::Vector4f xyz_centroid;
        pcl::compute3DCentroid(*tmp_clustered_cloud, xyz_centroid);

        std::cout << "cluster" << i << " centroid (x, y): " << xyz_centroid[0]  << ", " << xyz_centroid[1] << std::endl;

        Z_vec.push_back(xyz_centroid[0]);
        Z_vec.push_back(xyz_centroid[1]);

        std::cout << "size of Z_vec: " << Z_vec.size() << std::endl;
        clusters.push_back(tmp_clustered_cloud);


        // clusters_vv.emplace_back();
        // clusters_vv[candidate_num].push_back(tmp_clustered_cloud);
    }
    clusters_vv.push_back(clusters);
    clusters.clear();
    // Z_vv.emplace_back();
    // Z_vv[candidate_num].push_back(Z_vec);
}

void Kalman::Filtering()
{
    std::cout << "\nIn the Filtering" << std::endl;
    int cluster_size = Z_vec.size()/observed_num;
    std::cout << "Cluster_size: " << cluster_size << std::endl;
    std::cout << "Size of Z_vec: " << Z_vec.size() << std::endl;

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<double> V_dist(0,Q_matrix(0,0));
    std::normal_distribution<double> dV_dist(0, Q_matrix(2,2));

    std::normal_distribution<double> W_dist(0, R_matrix(0,0));
    std::normal_distribution<double> dW_dist(0, R_matrix(2,2));


    int cluster_num = 0;
    while (cluster_num < cluster_size)//cluster_sizeとは限らない
    {
        std::cout << "Cluster_num: " << cluster_num << std::endl;
        if(cycle == 0){
            // Post Estimate State and Initialize
            Eigen::Matrix<double, parameter_num, 1> X_post_ini;
            X_post_ini <<   Z_vec.at(observed_num * cluster_num),
                            Z_vec.at(observed_num * cluster_num + 1),
                            pedestrian_speed,
                            pedestrian_speed;
            X_post_vec.push_back(X_post_ini);
            std::cout << "Initial of estimate state :X\n" << X_post_ini << std::endl;

            // Post Estimate Uncertainty and Initialize
            Eigen::Matrix<double, parameter_num, parameter_num> P_post_ini;
            P_post_ini <<   0.2025, 0, 0, 0,
                            0, 0.2025, 0, 0,
                            0, 0, 0.2, 0,
                            0, 0, 0, 0.2;
            P_post_vec.push_back(P_post_ini);
            std::cout << "Initial of estimate uncertainty :P\n" << P_post_ini << std::endl;
            std::cout << "-----------------------------------------" << std::endl;

        }else{
            // Noise
            W_matrix << W_dist(engine),
                        W_dist(engine),
                        dW_dist(engine),
                        dW_dist(engine);

            V_matrix << V_dist(engine),
                        V_dist(engine),
                        dV_dist(engine),
                        dV_dist(engine);
            // std::cout << "W_matrix:\n" << W_matrix << "\n";
            // std::cout << "V_matrix:\n" << V_matrix << std::endl;

            // Predict
            std::cout << "Size of X_post_vec_dif: " << X_post_vec_dif.size() << std::endl;
            Eigen::Matrix<double, parameter_num, 1> X_post;
            X_post = X_post_vec_dif.at(cluster_num);
            Eigen::Matrix<double, parameter_num, 1> X_pri;
            X_pri <<    X_post(0,0) + X_post(2,0) * delta_t + V_matrix(0,0),
                        X_post(1,0) + X_post(3,0) * delta_t + V_matrix(1,0),
                        X_post(2,0) + V_matrix(2,0),
                        X_post(3,0) + V_matrix(3,0);
            // std::cout << "X_pri:\n" << X_pri << std::endl;
            // X_pri_vec.push_back(X_pri);

            Eigen::Matrix<double, parameter_num, parameter_num> P_post;
            P_post = P_post_vec_dif.at(cluster_num);
            Eigen::Matrix<double, parameter_num, parameter_num> P_pri;
            P_pri = P_post + Q_matrix;
            // std::cout << "P_pri:\n" << X_pri << std::endl;
            // P_pri_vec.puch_back(P_pri);


            std::cout << "In the sort" << std::endl;
            // あとで近傍点探索を実装する
            // Z_vecを並び替える
            // X_post_vecと比較し合わせる
            std::vector<double> distance_vec;
            for(int j = 0; j < cluster_size; j++){
                // distance_vv.emplace_back();
                distance_vec.push_back(std::pow(X_post(0,0) - Z_vec.at(observed_num * j), 2) 
                                        + std::pow(X_post(1,0) - Z_vec.at(observed_num * j+1), 2));
                std::cout << "Distance is: " << distance_vec.at(j) << std::endl;
            }

            int min_distance_index = std::distance(distance_vec.begin(), 
                                std::min_element(distance_vec.begin(), distance_vec.end()));

            Z_matrix << Z_vec.at(observed_num * min_distance_index),
                        Z_vec.at(observed_num * min_distance_index + 1),
                        (Z_vec.at(observed_num * min_distance_index) - X_post(0,0)) / delta_t,
                        (Z_vec.at(observed_num * min_distance_index + 1) - X_post(1,0)) / delta_t;

            Y_matrix  = H_ * Z_matrix + W_matrix;
            
            // Occupancy Map for comparison on measurement clustering
            std::cout << "Y_matrix(0,0): " << Y_matrix(0,0) << std::endl;
            std::cout << "Y_matrix(1,0): " << Y_matrix(1,0) << std::endl;
            int index_x = round((Y_matrix(0,0) + offset_x)/map_measurement.info.resolution);
            int index_y = round((Y_matrix(1,0) + offset_y)/map_measurement.info.resolution);
            // std::cout << "(Y_matrix(1,0) + 10): " << Y_matrix(1,0) + 10 << std::endl;
            std::cout << "index_x: " << index_x << std::endl;
            std::cout << "index_y: " << index_y << std::endl;
            map_measurement.data[index_y * map_measurement.info.height + index_x] = 16;
            std::cout << index_y * map_measurement.info.height + index_x << std::endl;
            std::cout << map_measurement.data[index_y * map_measurement.info.height + index_x] << std::endl;

            map_measurement.data[(index_y + 1) * map_measurement.info.height + index_x] =
            map_measurement.data[(index_y - 1) * map_measurement.info.height + index_x] =
            map_measurement.data[index_y * map_measurement.info.height + index_x + 1] =
            map_measurement.data[index_y * map_measurement.info.height + index_x - 1] = 10;

            map_measurement.data[(index_y + 1) * map_measurement.info.height + index_x + 1] =
            map_measurement.data[(index_y - 1) * map_measurement.info.height + index_x - 1] =
            map_measurement.data[(index_y + 1) * map_measurement.info.height + index_x - 1] = 
            map_measurement.data[(index_y - 1) * map_measurement.info.height + index_x + 1] = 6;

            map_measurement.data[(index_y + 2) * map_measurement.info.height + index_x] =
            map_measurement.data[(index_y - 2) * map_measurement.info.height + index_x] =
            map_measurement.data[index_y * map_measurement.info.height + index_x - 2] = 
            map_measurement.data[index_y * map_measurement.info.height + index_x + 2] = 2;

            map_measurement.data[(index_y + 1) * map_measurement.info.height + index_x + 2] =
            map_measurement.data[(index_y + 2) * map_measurement.info.height + index_x + 1] =
            map_measurement.data[(index_y + 2) * map_measurement.info.height + index_x - 1] =
            map_measurement.data[(index_y + 1) * map_measurement.info.height + index_x - 2] =
            map_measurement.data[(index_y - 1) * map_measurement.info.height + index_x - 2] =
            map_measurement.data[(index_y - 2) * map_measurement.info.height + index_x - 1] =
            map_measurement.data[(index_y - 2) * map_measurement.info.height + index_x + 1] =
            map_measurement.data[(index_y - 1) * map_measurement.info.height + index_x + 2] = 1;

            gridmap_pub_m.publish(map_measurement);


            // Update
            // std::cout << "\nCycle " << cycle << " -------------------------------" << std::endl;
            // std::cout << "Cluster Num " << cluster_num << " --------------------------" << std::endl;

            Eigen::Matrix<double, parameter_num, parameter_num> K_gain;
            K_gain = (P_pri * H_trans) * (H_ * P_pri * H_trans + R_matrix).inverse();
            // std::cout << "K_gain:\n" << K_gain << std::endl;

            K_gain_vec.push_back(K_gain);
            
            X_post = X_pri + K_gain * (Y_matrix - X_pri);
            // std::cout << "X_post:\n" << X_post << std::endl;
            P_post = (matrix_1 - K_gain) * P_pri;
            // std::cout << "P_post:\n" << P_post << std::endl;

            X_post_vec.push_back(X_post);
            P_post_vec.push_back(P_post);

            // Occupancy Map for comparison on estimated clustering
            std::cout << "X_post(0,0): " << X_post(0,0) << std::endl;
            std::cout << "X_post(1,1): " << X_post(1,0) << std::endl;
            std::cout << "(X_post(0,0) + 10) : " << X_post(0,0) + 10 << std::endl;
            int index_x_es = round((X_post(0,0) + offset_x)/map_estimate.info.resolution);
            int index_y_es = round((X_post(1,0) + offset_y)/map_estimate.info.resolution);
            std::cout << "index_x_es: " << index_x_es << std::endl;
            std::cout << "index_y_es: " << index_y_es << std::endl;

            map_estimate.data[index_y_es * map_estimate.info.height + index_x_es] = 16;

            map_estimate.data[(index_y_es + 1) * map_estimate.info.height + index_x_es] =
            map_estimate.data[(index_y_es - 1) * map_estimate.info.height + index_x_es] =
            map_estimate.data[index_y_es * map_estimate.info.height + index_x_es + 1] =
            map_estimate.data[index_y_es * map_estimate.info.height + index_x_es - 1] = 10;

            map_estimate.data[(index_y_es + 1) * map_estimate.info.height + index_x_es + 1] =
            map_estimate.data[(index_y_es - 1) * map_estimate.info.height + index_x_es - 1] =
            map_estimate.data[(index_y_es + 1) * map_estimate.info.height + index_x_es - 1] = 
            map_estimate.data[(index_y_es - 1) * map_estimate.info.height + index_x_es + 1] = 6;

            map_estimate.data[(index_y_es + 2) * map_estimate.info.height + index_x_es] =
            map_estimate.data[(index_y_es - 2) * map_estimate.info.height + index_x_es] =
            map_estimate.data[index_y_es * map_estimate.info.height + index_x_es - 2] = 
            map_estimate.data[index_y_es * map_estimate.info.height + index_x_es + 2] = 2;

            map_estimate.data[(index_y_es + 1) * map_estimate.info.height + index_x_es + 2] =
            map_estimate.data[(index_y_es + 2) * map_estimate.info.height + index_x_es + 1] =
            map_estimate.data[(index_y_es + 2) * map_estimate.info.height + index_x_es - 1] =
            map_estimate.data[(index_y_es + 1) * map_estimate.info.height + index_x_es - 2] =
            map_estimate.data[(index_y_es - 1) * map_estimate.info.height + index_x_es - 2] =
            map_estimate.data[(index_y_es - 2) * map_estimate.info.height + index_x_es - 1] =
            map_estimate.data[(index_y_es - 2) * map_estimate.info.height + index_x_es + 1] =
            map_estimate.data[(index_y_es - 1) * map_estimate.info.height + index_x_es + 2] = 1;

            map_estimate.header.frame_id = "/laser";
            gridmap_pub_e.publish(map_estimate);

            std::cout << "Kalman Gain: \n" << K_gain << std::endl;
            // std::cout << "Estimate state: \n" << X_post << "\n";
            // std::cout << "Estimate uncertainty: \n" << P_post << "\n";
            
            // ofs << cycle << "," << cluster_num << "," << cluster_size << ","
            // << Z_vec.at(observed_num * cluster_num) << "," << Z_vec.at(observed_num * cluster_num + 1) << ","
            // << X_post(0,0) << "," << X_post(1,0) << "," 
            // << X_post(2,0) << "," << X_post(3,0) << ","
            // << W_matrix(0,0) << "," << W_matrix(1,0) << "," << W_matrix(2,0) << "," << W_matrix(3,0) << ","
            // << V_matrix(0,0) << "," << V_matrix(1,0) << "," << V_matrix(2,0) << "," << V_matrix(3,0) << ","
            // << Y_matrix(0,0) << "," << Y_matrix(1,0) << ","
            // << X_pri(0,0) << "," << X_pri(1,0) << ","
            // << K_gain(0,0) << "," << K_gain(1,1) << "," << K_gain(2,2) << "," << K_gain(3,3) << std::endl;
            // std::cout << "-----------------------------------------" << std::endl;
        }
        cluster_num++;
    }

    X_post_vv.push_back(X_post_vec);
    X_post_vec.clear();
    P_post_vv.push_back(P_post_vec);
    P_post_vec.clear();

    std::cout << "Size of K_gain_vec: " << K_gain_vec.size() << std::endl;
    std::cout << "Size of K_gain_vv before push_back: " << K_gain_vv.size() << std::endl;
    K_gain_vv.push_back(K_gain_vec);
    K_gain_vec.clear();
    std::cout << "Size of K_gain_vv after push_back: " << K_gain_vv.size() << std::endl;

    Z_vec.clear();
    cycle++;
}

void Kalman::SAD()
{
    int sum = 0;

    for(unsigned int x = 0; x < map_size; x++){
        for(unsigned int y = 0; y < map_size; y++){
            sum += std::abs(map_measurement.data[y * map_measurement.info.height + x] - map_estimate.data[y * map_estimate.info.height + x]);
        }
    }
    std::cout << "Fitness_score is: " << sum << std::endl;

    std::vector<int8_t> map_initial(map_size*map_size, 0); //すべての要素の値0で初期化
    map_measurement.data = map_initial;
    map_estimate.data = map_initial;

    std::cout << "Size of map_measurement: " << map_measurement.data.size() << std::endl;

    fitness_score_vec.push_back(sum);
}

void Kalman::Visualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
{
    viewer.removeAllPointClouds();
    viewer.addPointCloud(input_cloud, "cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    /*clusters*/
    double rgb[3] = {};
    const int channel = 3;  //RGB   
    const double step = ceil(pow(clusters_dif.size()+2, 1.0/(double)channel));  //exept (000),(111)
    const double max = 1.0;
    // coloring
    for(size_t i=0;i<clusters_dif.size();i++){
            std::string name = "cluster_" + std::to_string(i);
            rgb[0] += 1/step;
            for(int j=0;j<channel-1;j++){
                    if(rgb[j]>max){
                            rgb[j] -= max + 1/step;
                            rgb[j+1] += 1/step;
                    }
            }
            viewer.addPointCloud(clusters_dif[i], name);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, rgb[0], rgb[1], rgb[2], name);
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
    }
    viewer.spinOnce();
    viewer.removeAllShapes();
    clusters_dif.clear();
} 

void Kalman::SSE()
{
    double sse = 0;

    for(size_t i = 0; i < clusters_dif.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr one_cluster(new pcl::PointCloud<pcl::PointXYZ>(*clusters_dif.at(i)));
        Eigen::Vector4f xyz_centroid;
        pcl::compute3DCentroid(*one_cluster, xyz_centroid);
        double cx = xyz_centroid[0];
        double cy = xyz_centroid[1];

        for(size_t j = 0; j < one_cluster->size(); j++)
        {
            sse += std::sqrt(std::pow((one_cluster->points[i].x - cx),2) + std::pow((one_cluster->points[i].y - cy),2));
        }
        std::cout << "SSE of cluster " << i << ": " << sse << std::endl;
    }
    std::cout << "SSE is: " << sse << std::endl;
    ofs_sse << sse << std::endl;
}


int main(int argc, char * argv[])
{
    ros::init(argc, argv, "kalman");
    Kalman kalman;
    ros::spin();
    return 0;
}
