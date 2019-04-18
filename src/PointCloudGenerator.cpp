//
// Created by lucas on 1/28/18.
//

#include <chrono>
#include <string>
#include <mutex>
#include <Utils.h>
#include <MathUtils.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/fast_bilateral.h>
//#include <opencv2/ximgproc.hpp>
#include "PointCloudGenerator.h"

int frame_id = 0;

namespace ark {

    PointCloudGenerator::PointCloudGenerator(std::string strSettingsFile) {
        cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);

        fx_ = fSettings["Camera.fx"];
        fy_ = fSettings["Camera.fy"];
        cx_ = fSettings["Camera.cx"];
        cy_ = fSettings["Camera.cy"];
        width_ = fSettings["Camera.width"];
        height_ = fSettings["Camera.height"];
        depthfactor_ = fSettings["DepthMapFactor"];
        maxdepth_ = fSettings["MaxDepth"];

        float v_g_o_x = fSettings["Voxel.Origin.x"];
        float v_g_o_y = fSettings["Voxel.Origin.y"];
        float v_g_o_z = fSettings["Voxel.Origin.z"];

        float v_size = fSettings["Voxel.Size"];

        float v_trunc_margin = fSettings["Voxel.TruncMargin"];

        int v_g_d_x = fSettings["Voxel.Dim.x"];
        int v_g_d_y = fSettings["Voxel.Dim.y"];
        int v_g_d_z = fSettings["Voxel.Dim.z"];

        mpGpuTsdfGenerator = new GpuTsdfGenerator(width_,height_,fx_,fy_,cx_,cy_, maxdepth_,
                                                           v_g_o_x,v_g_o_y,v_g_o_z,v_size,
                                                           v_trunc_margin,v_g_d_x,v_g_d_y,v_g_d_z);

        mKeyFrame.frameId = -1;
        mbRequestStop = false;
    }

    void PointCloudGenerator::Start() {
        mptRun = new std::thread(&PointCloudGenerator::Run, this);
    }


    void PointCloudGenerator::RequestStop() {
        std::cout<<"PointCloudGenerator stop"<<std::endl;
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        mbRequestStop = true;
    }

    bool PointCloudGenerator::IsRunning() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        return mbRequestStop;
    }

    void PointCloudGenerator::Run() {
        ark::RGBDFrame currentKeyFrame;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mRequestStopMutex);
                if (mbRequestStop)
                    break;
            }


            {
                std::unique_lock<std::mutex> lock(mKeyFrameMutex);
                if (currentKeyFrame.frameId == mKeyFrame.frameId)
                    continue;
                mKeyFrame.imDepth.copyTo(currentKeyFrame.imDepth);
                mKeyFrame.imRGB.copyTo(currentKeyFrame.imRGB);
                mKeyFrame.mTcw.copyTo(currentKeyFrame.mTcw);
                currentKeyFrame.frameId = mKeyFrame.frameId;
            }

            cv::Mat Twc = mKeyFrame.mTcw.inv();

            Reproject(currentKeyFrame.imRGB, currentKeyFrame.imDepth, Twc);
        }
    }

    void PointCloudGenerator::Reproject(const cv::Mat &imRGB, const cv::Mat &imD, const cv::Mat &Twc) {
//        cv::Mat imD_guided;
//        cv::Mat imGray(imD.rows, imD.cols, imD.depth());
//
//        for(int i=0;i<imD.rows;++i)
//            for(int j=0;j<imD.cols;++j)
//                imGray.at<float>(i,j) = (imRGB.at<cv::Vec3b>(i,j)[0]+imRGB.at<cv::Vec3b>(i,j)[1]+imRGB.at<cv::Vec3b>(i,j)[2])/3.0/255;
//
//        cv::ximgproc::jointBilateralFilter(imGray, imD, imD_guided, -1, 3, 9);
//
//        int boundary = 0;
//
//        for(int i=0;i<imD_guided.rows;++i)atorage fs;
//        fs.open("t_wc_"+std::to_string(frame_id)+".xml",cv::FileStorage::WRITE);
//        fs<<"pose"<<Twc;
//        fs.release();
//
//        cv::imwrite(std::to_string(frame_id)+".png", imRGB);
//
//        fs.open("depth_"+std::to_string(frame_id)+".xml",cv::FileStorage::WRITE);
//        fs<<"depth"<<imD;
//        fs.release();
//
//        frame_id++;

        float cam2base[16];
        for(int r=0;r<3;++r)
            for(int c=0;c<4;++c)
                cam2base[r*4+c] = Twc.at<float>(r,c);
        cam2base[12] = 0.0f;
        cam2base[13] = 0.0f;
        cam2base[14] = 0.0f;
        cam2base[15] = 1.0f;

        mpGpuTsdfGenerator->processFrame((float *)imD.datastart, (unsigned char *)imRGB.datastart, cam2base);
        //std::cout << "TSDF processed" << std::endl;
    }

    void PointCloudGenerator::Render(){
        mpGpuTsdfGenerator->render();
    }

    void PointCloudGenerator::SavePly(std::string filename) {
        mpGpuTsdfGenerator->SavePLY(filename);
    }

    void PointCloudGenerator::OnKeyFrameAvailable(const RGBDFrame &keyFrame) {
        if (mMapRGBDFrame.find(keyFrame.frameId) != mMapRGBDFrame.end())
            return;
        std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        keyFrame.mTcw.copyTo(mKeyFrame.mTcw);
        keyFrame.imRGB.copyTo(mKeyFrame.imRGB);
        keyFrame.imDepth.copyTo(mKeyFrame.imDepth);

        mKeyFrame.frameId = keyFrame.frameId;
        mMapRGBDFrame[keyFrame.frameId] = ark::RGBDFrame();
    }

    void PointCloudGenerator::OnFrameAvailable(const RGBDFrame &frame) {
        std::cout << "OnFrameAvailable" << frame.frameId << std::endl;
    }

    void PointCloudGenerator::OnLoopClosureDetected() {
        std::cout << "LoopClosureDetected" << std::endl;
    }
}

