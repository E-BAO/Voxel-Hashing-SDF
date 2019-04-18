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

    int tframe = 100;
    int empty = 0;


    float base2world[4 * 4];
    float cam2base[4 * 4];
    float cam2world[4 * 4];

    // Read base frame camera pose
    std::string base2world_file = tcwPath + to_string(tframe) + ".txt";
    std::vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
    std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

    float base2world_inv[16] = {0};
    invert_matrix(base2world, base2world_inv);


    float depth_im[height_ * width_];

    while (tframe < 150) {

        if(empty == 1)
            break;

        cout<<"frameLoad frame = "<<tframe<<endl;
        // Mat rgbBig = cv::imread(rgbPath + to_string(tframe) + ".png",cv::IMREAD_COLOR);
        cv::Mat rgbBig = cv::imread(rgbPath + to_string(tframe) + ".jpg",cv::IMREAD_COLOR);

        if(rgbBig.rows == 0){
            empty ++;
            tframe ++ ;
            cout<< "load "<<tframe<<" fail"<<endl;
            continue;
        }
        std::cout<<"--99-"<<std::endl;

        cv::Mat imRGB;

        cv::resize(rgbBig, imRGB, cv::Size(640,480));
        std::cout<<"--77-"<<std::endl;

        // rgbBig.release();
 
        cv::Mat depth255 = cv::imread(depthPath + std::to_string(tframe) + ".png",-1);
        cv::Mat imDepth;
        depth255.convertTo(imDepth, CV_32FC1);
        imDepth *= 0.001;

        std::cout<<"main depth = ";
        for(int k = 0; k < 5; k ++){
            printf("%lf ", imDepth.at<float>(k * 100,k * 100));
        }
        std::cout<<std::endl;

        std::cout<<"--55-"<<std::endl;

        // Mat tcw;
        // string cam2world_file = tcwPath + to_string(tframe) + ".txt";
        // std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
        // std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);
        // invert_matrix(cam2world, base2world_inv);


        float tcwArr[4][4];
        std::ifstream tcwFile;
        tcwFile.open(tcwPath + std::to_string(tframe) + ".txt");
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                tcwFile >> tcwArr[i][k];
            }
        }
        cv::Mat tcw(4, 4, CV_32FC1, tcwArr);    
        cv::Mat Twc = tcw;//.inv();

        // multiply_matrix(base2world_inv, cam2world, cam2base);


        float cam2base[16];
        for(int r=0;r<3;++r)
            for(int c=0;c<4;++c)
                cam2base[r*4+c] = Twc.at<float>(r,c);
        cam2base[12] = 0.0f;
        cam2base[13] = 0.0f;
        cam2base[14] = 0.0f;
        cam2base[15] = 1.0f;

        mpGpuTsdfGenerator->processFrame((float *)imDepth.datastart, (unsigned char *)imRGB.datastart, cam2base);

        empty = 0;

        tframe ++ ;
    }

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
