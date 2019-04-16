#include <iostream>
// #include <algorithm>
// #include <thread>

#include <stdio.h>
#include <sys/types.h>
// #include <dirent.h>
// #include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <tsdf.cuh>

using namespace std;
using namespace cv;

vector<float> LoadMatrixFromFile(std::string filename, int M, int N);
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);
bool invert_matrix(const float m[16], float invOut[16]);
void ReadDepth(std::string filename, int H, int W, float * depth);

int main(int argc, char** argv)
{
    if (argc != 3) {
        cerr << endl << "Usage: ./demo path_to_frames setting_file" << endl;
        return 1;
    }

    //Camera params
    float fx_, fy_, cx_, cy_;
    float maxdepth_;
    int width_, height_;
    float depthfactor_;

    bool offlineRecon;
    float v_g_o_x, v_g_o_y, v_g_o_z;

    string strSettingsFile = argv[2];
    cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);

    fx_ = fSettings["Camera.fx"];
    fy_ = fSettings["Camera.fy"];
    cx_ = fSettings["Camera.cx"];
    cy_ = fSettings["Camera.cy"];
    width_ = fSettings["Camera.width"];
    height_ = fSettings["Camera.height"];
    depthfactor_ = fSettings["DepthMapFactor"];
    maxdepth_ = fSettings["MaxDepth"];

    v_g_o_x = fSettings["Voxel.Origin.x"];
    v_g_o_y = fSettings["Voxel.Origin.y"];
    v_g_o_z = fSettings["Voxel.Origin.z"];

    float v_size = fSettings["Voxel.Size.Online"];

    float v_trunc_margin = fSettings["Voxel.TruncMargin.Online"];

    int v_g_d_x = fSettings["Voxel.Dim.x"];
    int v_g_d_y = fSettings["Voxel.Dim.y"];
    int v_g_d_z = fSettings["Voxel.Dim.z"];

    ark::GpuTsdfGenerator *mpGpuTsdfGenerator = new ark::GpuTsdfGenerator(width_,height_,fx_,fy_,cx_,cy_, maxdepth_,
                                                           v_g_o_x,v_g_o_y,v_g_o_z,v_size,
                                                           v_trunc_margin,v_g_d_x,v_g_d_y,v_g_d_z);

    string folderPath = argv[1];

    //folder path
    struct stat info;

    string rgbPath = folderPath +"RGB/";
    string depthPath = folderPath +"depth/";
    string tcwPath = folderPath +"tcw/";

    int tframe = 150;
    int empty = 0;


    float base2world[4 * 4];
    float cam2base[4 * 4];
    float cam2world[4 * 4];

    // Read base frame camera pose
    std::string base2world_file = tcwPath + "frame-000" + to_string(tframe) + ".pose" + ".txt";
    std::vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
    std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

    float base2world_inv[16] = {0};
    invert_matrix(base2world, base2world_inv);


    float depth_im[height_ * width_];

    while (tframe < 160) {

        if(empty == 1)
            break;

        cout<<"frameLoad frame = "<<tframe<<endl;
        // Mat rgbBig = cv::imread(rgbPath + to_string(tframe) + ".png",cv::IMREAD_COLOR);
        Mat rgbBig = cv::imread(rgbPath + "frame-000" + to_string(tframe) + ".color" + ".png",cv::IMREAD_COLOR);

        if(rgbBig.rows == 0){
            empty ++;
            tframe ++ ;
            cout<< "load "<<tframe<<" fail"<<endl;
            continue;
        }


        // Mat depth255 = cv::imread(depthPath + to_string(tframe) + ".png", -1);
        // depth255.convertTo(depth255, CV_32FC1);
        ReadDepth(depthPath + "frame-000" + to_string(tframe) + ".depth" + ".png", height_, width_, depth_im);

        // Mat tcw;
        string cam2world_file = tcwPath + "frame-000" + to_string(tframe) + ".pose" + ".txt";
        std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
        std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

        multiply_matrix(base2world_inv, cam2world, cam2base);

        for(int i = 0; i < 16; i ++)
          std::cout<<depth_im[i]<<" ";

        std::cout<<std::endl;

        mpGpuTsdfGenerator->processFrame(depth_im, (unsigned char *)rgbBig.datastart, cam2base);

        // cv::cvtColor(frame.imRGB, frame.imRGB, cv::COLOR_BGR2RGB);

        // pointCloudGenerator->OnKeyFrameAvailable(frame);

        empty = 0;

        tframe ++ ;
    }

    cout<<"processFrame"<<endl;

    mpGpuTsdfGenerator->SavePLY("model_online.ply");

    mpGpuTsdfGenerator->Shutdown();

    return 0;
}

vector<float> LoadMatrixFromFile(std::string filename, int M, int N) {
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++) {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}


// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]) {
  mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
  mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
  mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
  mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

  mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
  mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
  mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
  mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

  mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
  mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
  mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
  mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

  mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
  mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
  mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
  mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}


// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16]) {
  float inv[16], det;
  int i;
  inv[0] = m[5]  * m[10] * m[15] -
           m[5]  * m[11] * m[14] -
           m[9]  * m[6]  * m[15] +
           m[9]  * m[7]  * m[14] +
           m[13] * m[6]  * m[11] -
           m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] +
           m[4]  * m[11] * m[14] +
           m[8]  * m[6]  * m[15] -
           m[8]  * m[7]  * m[14] -
           m[12] * m[6]  * m[11] +
           m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] -
           m[4]  * m[11] * m[13] -
           m[8]  * m[5] * m[15] +
           m[8]  * m[7] * m[13] +
           m[12] * m[5] * m[11] -
           m[12] * m[7] * m[9];

  inv[12] = -m[4]  * m[9] * m[14] +
            m[4]  * m[10] * m[13] +
            m[8]  * m[5] * m[14] -
            m[8]  * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

  inv[1] = -m[1]  * m[10] * m[15] +
           m[1]  * m[11] * m[14] +
           m[9]  * m[2] * m[15] -
           m[9]  * m[3] * m[14] -
           m[13] * m[2] * m[11] +
           m[13] * m[3] * m[10];

  inv[5] = m[0]  * m[10] * m[15] -
           m[0]  * m[11] * m[14] -
           m[8]  * m[2] * m[15] +
           m[8]  * m[3] * m[14] +
           m[12] * m[2] * m[11] -
           m[12] * m[3] * m[10];

  inv[9] = -m[0]  * m[9] * m[15] +
           m[0]  * m[11] * m[13] +
           m[8]  * m[1] * m[15] -
           m[8]  * m[3] * m[13] -
           m[12] * m[1] * m[11] +
           m[12] * m[3] * m[9];

  inv[13] = m[0]  * m[9] * m[14] -
            m[0]  * m[10] * m[13] -
            m[8]  * m[1] * m[14] +
            m[8]  * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

  inv[2] = m[1]  * m[6] * m[15] -
           m[1]  * m[7] * m[14] -
           m[5]  * m[2] * m[15] +
           m[5]  * m[3] * m[14] +
           m[13] * m[2] * m[7] -
           m[13] * m[3] * m[6];

  inv[6] = -m[0]  * m[6] * m[15] +
           m[0]  * m[7] * m[14] +
           m[4]  * m[2] * m[15] -
           m[4]  * m[3] * m[14] -
           m[12] * m[2] * m[7] +
           m[12] * m[3] * m[6];

  inv[10] = m[0]  * m[5] * m[15] -
            m[0]  * m[7] * m[13] -
            m[4]  * m[1] * m[15] +
            m[4]  * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

  inv[14] = -m[0]  * m[5] * m[14] +
            m[0]  * m[6] * m[13] +
            m[4]  * m[1] * m[14] -
            m[4]  * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] +
           m[1] * m[7] * m[10] +
           m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] +
           m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] -
           m[0] * m[7] * m[10] -
           m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] -
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float * depth) {
  cv::Mat depth_mat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  if (depth_mat.empty()) {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f;
      if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
        depth[r * W + c] = 0;
    }
}
