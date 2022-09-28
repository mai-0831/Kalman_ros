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
#include <string>
#include <fstream>             
#include <sstream>
#include <cmath> //std::pow
#include <Eigen/Eigen>
#include <random>   //std::normal_distribution


const int parameter_num =4;

static std::string str_point_topic = "/points_scan";
static double cluster_tolerance = 0.3;
static int min_cluster_size = 3;

std::ofstream ofs("/home/itolab-mai/HDD/Kalman_ws/kousa_label0_vetygood_output.csv");

class Kalman
{
    private:
        int cycle = 0;

        ros::Subscriber points_sub;
        ros::Publisher centroid_pub;
        ros::Publisher estimate_centroid_pub;

        int observed_num = 2;
        double sensor_accuracy = 0.25;
        double q_noise = 0.1;
        double delta_t = 0.1;
        double u_noise = 0;

        std::string str_inputfile = "../data/kousa_label0_verygood.csv";
        std::string str_outpoutfile = "../data/kousa_label0_vetygood_output.csv";

        std::vector<double> Z_vec;  // Loading data of observed value
        Eigen::Matrix<double, parameter_num, 1> Z_matrix;   // Observed value
        Eigen::Matrix<double, parameter_num, 1> Y_matrix;  
        Eigen::Matrix<double, parameter_num,parameter_num> matrix_1;

        Eigen::Matrix<double, parameter_num, parameter_num> Q_matrix;   // Process noise of covariance
        Eigen::Matrix<double, parameter_num, parameter_num> R_matrix;   // Observed noise of covariance
        Eigen::Matrix<double, parameter_num, 1> W_matrix;               // Observed noise

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
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

        // std::ofstream ofs("../data/kousa_label0_vetygood_output.csv");

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Kalman();
        std::vector<double> Split(std::string& input, char delimiter);
        void LoadData();
        void Filtering();
        void PointCallback(const sensor_msgs::PointCloud2Ptr& points_msg);
        void Clustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
};


Kalman::Kalman()
{
    ros::NodeHandle nh("~");
    points_sub = nh.subscribe(str_point_topic, 1, &Kalman::PointCallback, this);
    // Kalman::LoadData();

    // ofs(str_outpoutfile);
    ofs << "cycle, cluster_num, centroid_x, centroid_y, estimate_x, estimate_y, estimate_vx, estimate_vy" << std::endl;

    // Parameter
    std::cout << "Destinate the parameteres" << std::endl;
    // 1 matrix
    matrix_1 << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
    // std::cout << matrix_1 << std::endl;

    // Process noise of covariance
    Q_matrix << std::pow(q_noise,2), 0, 0, 0,
                0, std::pow(q_noise,2), 0, 0,
                0, 0, std::pow(q_noise,2)/delta_t, 0,
                0, 0, 0, std::pow(q_noise,2)/delta_t;
    std::cout << "Process nose of covariance is: Q\n" << Q_matrix << std::endl;

    // Observed noise of covariance
    R_matrix << std::pow(sensor_accuracy,2), 0, 0, 0,
                0, std::pow(sensor_accuracy,2), 0, 0,
                0, 0, std::pow(sensor_accuracy,2)/delta_t, 0,
                0, 0, 0, std::pow(sensor_accuracy,2)/delta_t;
    std::cout << "Observed noise of covariance is: R\n" << R_matrix << std::endl; 

    // Matrix transformation formula
    H_ <<   1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
    H_trans = H_.transpose();
    std::cout << "Matrix transformation formula & trans\n" << H_ << "\n\n" << H_trans << std::endl; 

    // // Post Estimate State and Initialize
    // X_post <<   Z_vec.at(0),
    //             Z_vec.at(1),
    //             0.786,
    //             0.786;
    // std::cout << "Initial of estimate state :X\n" << X_post << std::endl;

    // // Post Estimate Uncertainty and Initialize
    // P_post <<   0.205, 0, 0, 0,
    //             0, 0.2025, 0, 0,
    //             0, 0, 0.025, 0,
    //             0, 0, 0, 0.025; 
    // std::cout << "Initial of estimate uncertainty :P\n" << P_post << std::endl;

    // Kalman::Filtering

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
    ece.setClusterTolerance(cluster_tolerance);
    ece.setMinClusterSize(min_cluster_size);
    ece.setMaxClusterSize(input_cloud->points.size());
    ece.setSearchMethod(tree);
    ece.setInputCloud(input_cloud);
    ece.extract(cluster_indices);

    // Divinding
    pcl::ExtractIndices<pcl::PointXYZ> ei;
    ei.setInputCloud(input_cloud);
    ei.setNegative(false);

    // std::vector<double> centroid;

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

        Z_vec.push_back(xyz_centroid[0]);
        Z_vec.push_back(xyz_centroid[1]);
        // centroid.push_back(xyz_centroid[2]);

        clusters.push_back(tmp_clustered_cloud);
    }

    /*cloud*/
    viewer.removeAllPointClouds();
    viewer.addPointCloud(input_cloud, "cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    /*clusters*/
    double rgb[3] = {};
    const int channel = 3;  //RGB   
    const double step = ceil(pow(clusters.size()+2, 1.0/(double)channel));  //exept (000),(111)
    const double max = 1.0;
    /*クラスタをいい感じに色分け*/
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

    Kalman::Filtering();
}


std::vector<double> Kalman::Split(std::string& input, char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    double tmp_double;
    std::vector<double> result;
    while(getline(stream, field, delimiter))
    {
        tmp_double = std::stod(field);
        result.push_back(tmp_double);
    }
    // std::cout << "Load data " << result.size() << std::endl;
    return result;
}


void Kalman::LoadData()
{
    std::ifstream ifs(str_inputfile);
    std::string line;

    std::cout <<"Load " << str_inputfile << std::endl;

    while(std::getline(ifs, line))
    {
        std::vector<double> tmp_vec{Kalman::Split(line, ',')};
        Z_vec.insert(Z_vec.end(), tmp_vec.begin(), tmp_vec.end());
        // std::cout << Z_vec.at(0) << Z_vec.at(1) << Z_vec.at(2) << Z_vec.at(3) << std::endl; 
    }

    std::cout << "Load data size is " << Z_vec.size() << std::endl;
}



void Kalman::Filtering(void)
{
    // int cluster_size = Z_vec.size()/observed_num;
    int cluster_size = 2;
    std::cout << "cluster_size: " << cluster_size << std::endl;

    // int cycle = 1;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<double> V_dist(0,Q_matrix(0,0));
    std::normal_distribution<double> dV_dist(0, Q_matrix(2,0));

    std::normal_distribution<double> W_dist(0, R_matrix(0,0));
    std::normal_distribution<double> dW_dist(0, R_matrix(2,0));

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
                        0.786,
                        0.786;
            X_post_vec.push_back(X_post_ini);
            // std::cout << "Initial of estimate state :X\n" << X_post_ini << std::endl;

            // Post Estimate Uncertainty and Initialize
            Eigen::Matrix<double, parameter_num, parameter_num> P_post_ini;
            P_post_ini <<   0.205, 0, 0, 0,
                            0, 0.2025, 0, 0,
                            0, 0, 0.025, 0,
                            0, 0, 0, 0.025;
            P_post_vec.push_back(P_post_ini);
            // std::cout << "Initial of estimate uncertainty :P\n" << P_post_ini << std::endl;

        }else{
            // Noise
            W_matrix << W_dist(engine),
                        W_dist(engine),
                        dW_dist(engine),
                        dW_dist(engine);

            // Predict
            Eigen::Matrix<double, parameter_num, 1> X_post;
            X_post = X_post_vec.at(cluster_num);
            Eigen::Matrix<double, parameter_num, 1> X_pri;
            X_pri <<    X_post(0,0) + X_post(2,0) * delta_t + V_dist(engine),
                        X_post(1,0) + X_post(3,0) * delta_t + V_dist(engine),
                        X_post(2,0) + dV_dist(engine),
                        X_post(3,0) + dV_dist(engine);
            // X_pri_vec.push_back(X_pri);

            Eigen::Matrix<double, parameter_num, parameter_num> P_post;
            P_post = P_post_vec.at(cluster_num);
            Eigen::Matrix<double, parameter_num, parameter_num> P_pri;
            P_pri = P_post + Q_matrix;
            // P_pri_vec.puch_back(P_pri);

            // Observation
            Z_matrix << Z_vec.at(observed_num * cluster_num),
                        Z_vec.at(observed_num * cluster_num + 1),
                        (Z_vec.at(observed_num * cluster_num) - X_post(0,0)) / delta_t,
                        (Z_vec.at(observed_num * cluster_num + 1) - X_post(1,0)) / delta_t;
            
            Y_matrix  = H_ * Z_matrix + W_matrix;
            

            //Update
            std::cout << "\nCycle " << cycle << " -------------------------------" << std::endl;
            std::cout << "Cluster Num " << cluster_num << " --------------------------" << std::endl;
            Eigen::Matrix<double, parameter_num, parameter_num> K_gain;
            K_gain = (P_pri * H_trans) * (H_ * P_pri * H_trans + R_matrix).transpose();
            X_post = X_pri + K_gain * (Y_matrix - X_pri);
            P_post = (matrix_1 - K_gain) * P_pri;

            X_post_vec.at(cluster_num) = X_post;
            P_post_vec.at(cluster_num) = P_post;
            // X_post_vec.push_back(X_post);
            // P_post_vec.push_back(P_post);

            std::cout << "Kalman Gain: \n" << K_gain << "\n";
            std::cout << "Estimate state: \n" << X_post << "\n";
            std::cout << "Estimate uncertainty: \n" << P_post << "\n";
            
            ofs << cycle << "," << cluster_num << "," 
            << Z_vec.at(observed_num * cluster_num) << "," << Z_vec.at(observed_num * cluster_num + 1) << ","
            << X_post(0,0) << "," << X_post(1,0) << "," 
            << X_post(2,0) << "," << X_post(3,0) << std::endl;
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
