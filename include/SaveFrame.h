#ifndef OPENARK_SAVEFRAME_H
#define OPENARK_SAVEFRAME_H

#include "Utils.h"

#include <mutex>
#include <thread>
#include <map>
#include <string>

#include <opencv2/opencv.hpp>


namespace ark{

    class SaveFrame{
    public:
        SaveFrame(std::string folderPath);

        void Start();

        void RequestStop();

        bool IsRunning();

        void OnKeyFrameAvailable(const RGBDFrame &keyFrame);

        void OnFrameAvailable(const RGBDFrame &frame);

        void OnLoopClosureDetected();

        void Run();

        void frameWrite(const RGBDFrame &frame);

        ark::RGBDFrame frameLoad(int frameId);

//        RGBDFrame loadFrame();

//        void Render();

    private:

//        void Reproject(const cv::Mat &imRGB,const cv::Mat &imD, const cv::Mat &Twc);

        //Main Loop thread
        std::thread *mptRun;
        std::string folderPath;
        std::string rgbPath;
        std::string depthPath;
        std::string tcwPath;
        std::string depth_to_tcw_Path;

//        //TSDF Generator
//        GpuTsdfGenerator *mpGpuTsdfGenerator;

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

//        //Camera params
//        float fx_, fy_, cx_, cy_;
//        float maxdepth_;
//        int width_, height_;
//        float depthfactor_;
    };
}




#endif  //#define OPENARK_SAVEFRAME_H
