#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

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

static std::string str_point_topic = "/points_scan";
static double max_cluster_tolerance = 0.5;
static double min_cluster_tolerance = 0.1;
static int clustering_loop_num = 10;
static int min_cluster_size = 3;

std::ofstream ofs("/home/itolab-mai/HDD/Kalman_ws/result.csv");

class Kalman
{
    private:
        int cycle = 0;

        ros::Subscriber points_sub;
        ros::Publisher centroid_pub;
        ros::Publisher estimate_centroid_pub;

        // std::vector<std::array<double,3>> condidate_vv;
        std::vector<std::vector<double>> candidate_vv;

        int observed_num = 2;
        double sensor_accuracy = 0.02;
        double q_noise = 0.0001;
        double delta_t = 0.2;
        double pedestrian_speed = -0.1;

        double cluster_tolerance = 0.35;

        std::vector<double> Z_vec;  // Loading data of observed value
        Eigen::Matrix<double, parameter_num, 1> Z_matrix;   // Observed value
        Eigen::Matrix<double, parameter_num, 1> Y_matrix;  
        Eigen::Matrix<double, parameter_num,parameter_num> matrix_1;

        Eigen::Matrix<double, parameter_num, parameter_num> Q_matrix;   // Process noise of covariance
        Eigen::Matrix<double, parameter_num, parameter_num> R_matrix;   // Observed noise of covariance
        Eigen::Matrix<double, parameter_num, 1> W_matrix;               // Observed noise
        Eigen::Matrix<double, parameter_num, 1> V_matrix;               // Observed noise

        std::vector<Eigen::Matrix<double, parameter_num, 1>> X_pri_vec;  // Pri Estimate state
        std::vector<Eigen::Matrix<double, parameter_num, 1>> X_post_vec; // Post Estimate state
        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> P_pri_vec;  // Pri Estimate uncertainty
        std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> P_post_vec; // Post Estimate uncertainty

        // std::vector<Eigen::Matrix<double, parameter_num, parameter_num>> K_gain_vec; // Kalman Gain

        // Eigen::Matrix<double, parameter_num, 1> X_pri;  // Pri Estimate state
        // Eigen::Matrix<double, parameter_num, 1> X_post; // Post Estimate state
        // Eigen::Matrix<double, parameter_num, parameter_num> P_pri;  // Pri Estimate uncertainty
        // Eigen::Matrix<double, parameter_num, parameter_num> P_post; // Post Estimate uncertainty

        // Eigen::Matrix<double, parameter_num, parameter_num> K_gain; // Kalman Gain

        Eigen::Matrix<double, parameter_num, parameter_num> H_;
        Eigen::Matrix<double, parameter_num, parameter_num> H_trans;

        pcl::visualization::PCLVisualizer viewer {"Euclidian Clustering"};
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;  //for viewer

        // std::ofstream ofs("../data/kousa_label0_vetygood_output.csv");

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Kalman();
        std::vector<double> Split(std::string& input, char delimiter);
        void LoadData();
        void Filtering();
        void PointCallback(const sensor_msgs::PointCloud2Ptr& points_msg);
        void Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
        void Centroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, const std::vector<pcl::PointIndices>& cluster_indices);
};


Kalman::Kalman()
{
    ros::NodeHandle nh("~");
    points_sub = nh.subscribe(str_point_topic, 1, &Kalman::PointCallback, this);

    ofs << "q_noise, sensor_accuracy, delta_t, pedestrian_speed" << std::endl;
    ofs << q_noise << "," << sensor_accuracy << "," << delta_t << "," << pedestrian_speed << std::endl;
    ofs << "cycle, cluster_num, cluster_size, centroid_x, centroid_y, estimate_x, estimate_y, estimate_vx, estimate_vy" << std::endl;

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

    viewer.setBackgroundColor(1, 1, 1);
    viewer.addCoordinateSystem(1.0, "axis");
    viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);
}

void Kalman::PointCallback(const sensor_msgs::PointCloud2Ptr& points_msg)
{
    std::cout << "In the PointCallback" << std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*points_msg, cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>(cloud));

    Kalman::Clustering(tmp_cloud);
}

void Kalman::Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
{
    std::cout << "In the Clustering" << std::endl;

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

    // Generate the cluster tolerance with rondom
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> distr(min_cluster_tolerance, max_cluster_tolerance);


    if (cycle == 0)
    {
        ece.setClusterTolerance(cluster_tolerance);
        ece.extract(cluster_indices);
        Kalman::Centroid(input_cloud, cluster_indices);
        // ここでZを決定する
    }else{
        for (int n = 0; n < clustering_loop_num; n++)
        {
            cluster_tolerance = distr(eng);
            ece.setClusterTolerance(cluster_tolerance); // ここの変数を変更していく
            ece.extract(cluster_indices);
            Kalman::Centroid(input_cloud, cluster_indices);            
        }
        // 距離を計算
        // ここでZを決定する
        int max_index = std::distance(candidate_vv[1].begin(), 
                                        std::max_element(candidate_vv[1].begin(), candidate_vv[1].end()));
        Z_vec.push_back(candidate_vv[0][2*max_index]);
        Z_vec.push_back(candidate_vv[0][2*max_index + 1]);
    }
    candidate_vv.clear();
    Kalman::Filtering();
}

void Kalman::Centroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, const std::vector<pcl::PointIndices>& cluster_indices)
{

    // Divinding
    pcl::ExtractIndices<pcl::PointXYZ> ei;
    ei.setInputCloud(input_cloud);
    ei.setNegative(false);

    std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;

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

        std::cout << "centroid x: " << xyz_centroid[0] << std::endl;
        std::cout << "centroid y: " << xyz_centroid[1] << std::endl;

        candidate_vv[0].push_back(xyz_centroid[0]);
        candidate_vv[0].push_back(xyz_centroid[1]);
        // centroid.push_back(xyz_centroid[2]);

        clusters.push_back(tmp_clustered_cloud);
    }

    // Visualization
    viewer.removeAllPointClouds();
    viewer.addPointCloud(input_cloud, "cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    /*clusters*/
    double rgb[3] = {};
    const int channel = 3;  //RGB   
    const double step = ceil(pow(clusters.size()+2, 1.0/(double)channel));  //exept (000),(111)
    const double max = 1.0;
    // coloring
    for(size_t i=0;i<clusters.size();i++){
            std::string name = "cluster_" + std::to_string(i);
            rgb[0] += 1/step;
            for(int j=0;j<channel-1;j++){
                    if(rgb[j]>max){
                            rgb[j] -= max + 1/step;
                            rgb[j+1] += 1/step;
                    }
            }
            viewer.addPointCloud(clusters[i], name);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, rgb[0], rgb[1], rgb[2], name);
            viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
    }
    /*表示の更新*/
    viewer.spinOnce();
    clusters.clear();
}

void Kalman::Filtering()
{
    int cluster_size = Z_vec.size()/observed_num;
    // int cluster_size = 2;
    std::cout << "cluster_size: " << cluster_size << std::endl;

    // int cycle = 1;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<double> V_dist(0,Q_matrix(0,0));
    std::normal_distribution<double> dV_dist(0, Q_matrix(2,2));

    std::normal_distribution<double> W_dist(0, R_matrix(0,0));
    std::normal_distribution<double> dW_dist(0, R_matrix(2,2));

    // Sort
    // if(cycle != 0)
    // {
    //     std::cout << "In the sort" << std::endl;
    //     std::vector<double> distance_vec;
    //     for(int i = 0; i < cluster_size; i++)
    //     {
    //         Eigen::Matrix<double, parameter_num, 1> X_post_tmp;
    //         X_post_tmp = X_post_vec.at(i); 
    //         // distance_vec.push_back(std::pow(X_post_tmp(0,0) - Z_vec.at(observed_num * i), 2) + std::pow(X_post_tmp(1,0) - Z_vec.at(observed_num * i+1), 2));
    //         distance_vec.push_back(std::pow(Z_matrix(0,0) - Z_vec.at(observed_num * i), 2) + std::pow(Z_matrix(1,0) - Z_vec.at(observed_num * i+1), 2));
    //         std::cout << "distance: " << distance_vec.at(i) << std::endl;
    //     }
    //     if(distance_vec.at(0) < distance_vec.at(1))
    //     {
    //         std::cout << "Swappppp" << std::endl;
    //         std::vector<double> Z_tmp{Z_vec.at(2), Z_vec.at(3), Z_vec.at(0), Z_vec.at(1)};
    //         Z_vec = Z_tmp;
    //     }        
    // }

    int cluster_num = 0;
    while (cluster_num < cluster_size)
    {
        std::cout << "cluster_num: " << cluster_num << std::endl;
        std::cout << "cycle: " << cycle << std::endl;
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

            // Predict
            Eigen::Matrix<double, parameter_num, 1> X_post;
            X_post = X_post_vec.at(cluster_num);
            Eigen::Matrix<double, parameter_num, 1> X_pri;
            X_pri <<    X_post(0,0) + X_post(2,0) * delta_t + V_matrix(0,0),
                        X_post(1,0) + X_post(3,0) * delta_t + V_matrix(1,0),
                        X_post(2,0) + V_matrix(2,0),
                        X_post(3,0) + V_matrix(3,0);
            // X_pri_vec.push_back(X_pri);

            Eigen::Matrix<double, parameter_num, parameter_num> P_post;
            P_post = P_post_vec.at(cluster_num);
            Eigen::Matrix<double, parameter_num, parameter_num> P_pri;
            P_pri = P_post + Q_matrix;
            // P_pri_vec.puch_back(P_pri);

            // Observation
            if(cluster_size == 2)
            {
                Z_matrix << Z_vec.at(observed_num * cluster_num),
                            Z_vec.at(observed_num * cluster_num + 1),
                            (Z_vec.at(observed_num * cluster_num) - X_post(0,0)) / delta_t,
                            (Z_vec.at(observed_num * cluster_num + 1) - X_post(1,0)) / delta_t;
            }else{
                Z_matrix << X_pri(0,0),
                            X_pri(1,0),
                            X_pri(2,0),
                            X_pri(3,0);
            }
            Y_matrix  = H_ * Z_matrix + W_matrix;
            

            // Update
            std::cout << "\nCycle " << cycle << " -------------------------------" << std::endl;
            std::cout << "Cluster Num " << cluster_num << " --------------------------" << std::endl;
            Eigen::Matrix<double, parameter_num, parameter_num> K_gain;
            K_gain = (P_pri * H_trans) * (H_ * P_pri * H_trans + R_matrix).inverse();
            
            X_post = X_pri + K_gain * (Y_matrix - X_pri);
            P_post = (matrix_1 - K_gain) * P_pri;

            X_post_vec.at(cluster_num) = X_post;
            P_post_vec.at(cluster_num) = P_post;
            // X_post_vec.push_back(X_post);
            // P_post_vec.push_back(P_post);

            std::cout << "Kalman Gain: \n" << K_gain << "\n";
            std::cout << "Estimate state: \n" << X_post << "\n";
            std::cout << "Estimate uncertainty: \n" << P_post << "\n";
            
            ofs << cycle << "," << cluster_num << "," << cluster_size << ","
            << Z_vec.at(observed_num * cluster_num) << "," << Z_vec.at(observed_num * cluster_num + 1) << ","
            << X_post(0,0) << "," << X_post(1,0) << "," 
            << X_post(2,0) << "," << X_post(3,0) << ","
            << W_matrix(0,0) << "," << W_matrix(1,0) << "," << W_matrix(2,0) << "," << W_matrix(3,0) << ","
            << V_matrix(0,0) << "," << V_matrix(1,0) << "," << V_matrix(2,0) << "," << V_matrix(3,0) << ","
            << Y_matrix(0,0) << "," << Y_matrix(1,0) << ","
            << X_pri(0,0) << "," << X_pri(1,0) << ","
            << K_gain(0,0) << "," << K_gain(1,1) << "," << K_gain(2,2) << "," << K_gain(3,3) << std::endl;
            std::cout << "-----------------------------------------" << std::endl;
        }
        cluster_num++;
    }
    Z_vec.clear();
    cycle++;
}



int main(int argc, char * argv[])
{
    ros::init(argc, argv, "kalman");
    Kalman kalman;
    ros::spin();
    return 0;
}
