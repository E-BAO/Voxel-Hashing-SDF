//
// Created by yiwen on 2/2/19.
//

#include <chrono>
#include <mutex>
#include <Utils.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

//#include <MathUtils.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/filters/fast_bilateral.h>
// #include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include "SaveFrame.h"

namespace ark {

    void createFolder(struct stat &info, std::string folderPath){
        if(stat( folderPath.c_str(), &info ) != 0 ) {
            std::cout<< "Error:"<< folderPath<<" doesn't exist!" << std::endl;
            exit(1);

            if (-1 == mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
            {
                std::cout<< "Error creating directory "<< folderPath<<" !" << std::endl;
                exit(1);
            }
            std::cout << folderPath << " is created" << folderPath << std::endl;
        }else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows
            std::cout<<folderPath<<" is a directory"<<std::endl;
        else
            std::cout<<folderPath<<" is no directory"<<std::endl;
    }

    SaveFrame::SaveFrame(std::string folderPath) {


        struct stat info;

        createFolder(info, folderPath);

        rgbPath = folderPath +"RGB/";
        depthPath = folderPath +"depth/";
        tcwPath = folderPath +"tcw/";

        createFolder(info, rgbPath);
        createFolder(info, depthPath);
        createFolder(info, tcwPath);

        mKeyFrame.frameId = -1;
        mbRequestStop = false;
    }

    void SaveFrame::Start() {
        mptRun = new std::thread(&SaveFrame::Run, this);
    }


    void SaveFrame::RequestStop() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        mbRequestStop = true;
    }

    bool SaveFrame::IsRunning() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        return mbRequestStop;
    }

    void SaveFrame::Run() {
//        ark::RGBDFrame currentKeyFrame;
//        while (true) {
//            {
//                std::unique_lock<std::mutex> lock(mRequestStopMutex);
//                if (mbRequestStop)
//                    break;
//            }
//
//
//            {
//                std::unique_lock<std::mutex> lock(mKeyFrameMutex);
//                if (currentKeyFrame.frameId == mKeyFrame.frameId)
//                    continue;
//                mKeyFrame.imDepth.copyTo(currentKeyFrame.imDepth);
//                mKeyFrame.imRGB.copyTo(currentKeyFrame.imRGB);
//                mKeyFrame.mTcw.copyTo(currentKeyFrame.mTcw);
//                currentKeyFrame.frameId = mKeyFrame.frameId;
//            }
//
//            cv::Mat Twc = mKeyFrame.mTcw.inv();
//
////            Reproject(currentKeyFrame.imRGB, currentKeyFrame.imDepth, Twc);
//        }
    }

    void SaveFrame::OnKeyFrameAvailable(const RGBDFrame &keyFrame) {
        if (mMapRGBDFrame.find(keyFrame.frameId) != mMapRGBDFrame.end())
            return;
        std::cout << "OnKeyFrameAvailable" << keyFrame.frameId << std::endl;
        // std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        keyFrame.mTcw.copyTo(mKeyFrame.mTcw);
        keyFrame.imRGB.copyTo(mKeyFrame.imRGB);
        keyFrame.imDepth.copyTo(mKeyFrame.imDepth);

        mKeyFrame.frameId = keyFrame.frameId;
        mMapRGBDFrame[keyFrame.frameId] = ark::RGBDFrame();
    }

    void SaveFrame::OnFrameAvailable(const RGBDFrame &frame) {
        std::cout << "OnFrameAvailable" << frame.frameId << std::endl;
    }

    void SaveFrame::OnLoopClosureDetected() {
        std::cout << "LoopClosureDetected" << std::endl;
    }

    void SaveFrame::frameWrite(const RGBDFrame &frame){
        if (mMapRGBDFrame.find(frame.frameId) != mMapRGBDFrame.end())
            return;

        std::cout<<"frameWrite frame = "<<frame.frameId<<std::endl;
        if(frame.frameId > 300)
            return;

//        std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        cv::imwrite(rgbPath + std::to_string(frame.frameId) + ".png", frame.imRGB);

        cv::Mat depth255;
        //cv::normalize(frame.imDepth, depth255, 0, 1000, cv::NORM_MINMAX, CV_16UC1); ////cast to 16

        
        frame.imDepth.convertTo(depth255, CV_16UC1, 1000);
        cv::imwrite(depthPath + std::to_string(frame.frameId) + ".png", depth255);

        cv::FileStorage fs(tcwPath + std::to_string(frame.frameId)+".xml",cv::FileStorage::WRITE);
        fs << "tcw" << frame.mTcw ;
        //fs << "depth" << frame.imDepth ;
        fs.release();

        /*
        cv::FileStorage fs2(depth_to_tcw_Path + std::to_string(frame.frameId)+".xml",cv::FileStorage::WRITE);
        fs2 << "depth" << depth255;
        // fs << "rgb" << frame.imRGB;
        fs2.release();
        */

        mMapRGBDFrame[frame.frameId] = ark::RGBDFrame();

    }

    RGBDFrame SaveFrame::frameLoad(int frameId){
        std::cout<<"frameLoad frame ==================== "<<frameId<<std::endl;
//        if(frameId > 300)
//            return;
        // std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        RGBDFrame frame;

        frame.frameId = frameId;


        cv::Mat rgbBig = cv::imread(rgbPath + std::to_string(frame.frameId) + ".jpg",cv::IMREAD_COLOR);

        if(rgbBig.rows == 0){
            frame.frameId = -1;
            return frame;
        }

        cv::resize(rgbBig, frame.imRGB, cv::Size(640,480));

        
        cv::Mat depth255 = cv::imread(depthPath + std::to_string(frame.frameId) + ".png",-1);
        //std::cout << "type: " << depth255.type() << std::endl ;
        //if(frame.frameId == 1)
        //	std::cout << "depth255 = "<< std::endl << " "  << depth255 << std::endl << std::endl;

        depth255.convertTo(frame.imDepth, CV_32FC1);
        frame.imDepth *= 0.001;
        //cv::normalize(depth255, frame.imDepth, 0.2, 10, cv::NORM_MINMAX, CV_32F);
        


        //TCW FROM XML
        /*
        cv::FileStorage fs2(tcwPath + std::to_string(frame.frameId)+".xml", cv::FileStorage::READ);
        fs2["tcw"] >> frame.mTcw;
        
        //fs2["depth"] >> frame.imDepth;
        fs2.release();
        */



        //TCW FROM TEXT
        float tcwArr[4][4];
        std::ifstream tcwFile;
        tcwFile.open(tcwPath + std::to_string(frame.frameId) + ".txt");
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                tcwFile >> tcwArr[i][k];
            }
        }
        cv::Mat tcw(4, 4, CV_32FC1, tcwArr);    
        frame.mTcw = tcw.inv();




        //std::cout << "debugging frame#: " << frame.frameId << std::endl; 
        //std::cout << tcw << std::endl;
        //std::cout << frame.imRGB.rows << std::endl;
        //std::cout << frame.imDepth << std::endl;
        //std::cout << frame.imDepth.rows << std::endl;

        return frame;
    }

}

