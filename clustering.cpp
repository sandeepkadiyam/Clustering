#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
#include <vector>
#include <../pcl-1.8/pcl/io/pcd_io.h>
#include <../pcl-1.8/pcl/point_types.h>
#include <../pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <../pcl-1.8/pcl/console/parse.h>
#include <../pcl-1.8/pcl/common/common.h>
#include <../pcl-1.8/pcl/filters/extract_indices.h>
#include <../pcl-1.8/pcl/filters/voxel_grid.h>
#include <../pcl-1.8/pcl/filters/crop_box.h>
#include <../pcl-1.8/pcl/kdtree/kdtree.h>
#include <../pcl-1.8/pcl/segmentation/sac_segmentation.h>
#include <../pcl-1.8/pcl/segmentation/extract_clusters.h>
#include <../pcl-1.8/pcl/common/transforms.h>

using namespace std;

struct Color
{

	float r, g, b;

	Color(float setR, float setG, float setB)
		: r(setR), g(setG), b(setB)
	{}
};

struct Box
{
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
};

Box BoundingBox(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    pcl::PointXYZI minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

void renderBox(pcl::visualization::PCLVisualizer::Ptr& viewer, Box box, int id, Color color, float opacity)
{
	if(opacity > 1.0)
		opacity = 1.0;
	if(opacity < 0.0)
		opacity = 0.0;
	
	std::string cube = "box"+std::to_string(id);
    
    viewer->addCube(box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max, color.r, color.g, color.b, cube);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, cube);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cube);
    
    std::string cubeFill = "boxFill"+std::to_string(id);
    
    viewer->addCube(box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max, color.r, color.g, color.b, cubeFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, cubeFill); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, cubeFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity*0.3, cubeFill);
}


void renderPointCloud(pcl::visualization::PCLVisualizer::Ptr& viewer, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, std::string name)
{

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> point_cloud_color_handler(cloud, "intensity");
  	viewer->addPointCloud<pcl::PointXYZI> (cloud, point_cloud_color_handler, name);
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, name);

}

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> Clustering(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    // Time clustering process.
    auto startTime = std::chrono::steady_clock::now();

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clusters;

    // performing euclidean clustering to group detected obstacles.
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for(pcl::PointIndices getIndices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZI>);

        for (int index : getIndices.indices) {
            cloudCluster->points.push_back(cloud->points[index]);
        }

        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        clusters.push_back(cloudCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


int main() {

    // Loading the PCD from the given file.

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI> ("../cloud.pcd", *cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file cloud.pcd \n");
        return (-1);
    }

    std::cerr << "Loaded " << cloud->points.size() << " data points from cloud.pcd" << std::endl;

    // Visualizing the point cloud.

    std::cout << "starting enviroment" << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    

    renderPointCloud(viewer, cloud, "Frst view");  // Renders the input point cloud.


    // Performing Eucledian Clustering.

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = Clustering(cloud, 0.65, 150, 2000);

    int cluster_Id = 0;

    for(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : cloudClusters)
    {
        std::cout << "cluster size " << cluster->points.size()  << std::endl;;
        renderPointCloud(viewer, cluster, "obstCloud"+std::to_string(cluster_Id));  // Renders all the clusters found.
        Box box = BoundingBox(cluster);
        renderBox(viewer, box, cluster_Id, Color(1,0,0), 1.0);  // Renders a bounding box around the clusters found.
        ++cluster_Id;
    }

    // Starting the pcl visualizer.

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce ();
    } 

    return 0;

}
