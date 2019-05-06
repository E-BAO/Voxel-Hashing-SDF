//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_POINTCLOUDGENERATOR_H
#define OPENARK_POINTCLOUDGENERATOR_H

#include "Utils.h"

#include <tsdf.cuh>

#include <mutex>
#include <thread>
#include <map>
#include <string>
#include <iostream>
#include <sstream>



#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>

namespace ark{

    class PointCloudGenerator{
    public:
        PointCloudGenerator(std::string strSettingsFile);

        void Start();

        void RequestStop();

        bool IsRunning();

        void OnKeyFrameAvailable(const RGBDFrame &keyFrame);

        void OnFrameAvailable(const RGBDFrame &frame);

        void OnLoopClosureDetected();

        void Run();

        void SavePly(std::string filename);

        void Render();

        void Reproject(const cv::Mat &imRGB,const cv::Mat &imD, const cv::Mat &Twc);

        void PushFrame(const RGBDFrame &frame);
    private:


        //Main Loop thread
        std::thread *mptRun;

        //TSDF Generator
        GpuTsdfGenerator *mpGpuTsdfGenerator;

        //RGBDFrame Map
        std::map<int, ark::RGBDFrame> mMapRGBDFrame;

        //Current KeyFrame
        std::mutex mKeyFrameMutex;
        ark::RGBDFrame mKeyFrame;

        //Current Frame
        std::mutex mFrameMutex;
        ark::RGBDFrame mFrame;

        //Request Stop Status
        std::mutex mRequestStopMutex;
        bool mbRequestStop;

        //Camera params
        float fx_, fy_, cx_, cy_;
        float maxdepth_;
        int width_, height_;
        float depthfactor_;
    };
}

#endif //OPENARK_POINTCLOUDGENERATOR_H
