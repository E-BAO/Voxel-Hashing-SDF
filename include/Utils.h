//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_UTILS_H
#define OPENARK_UTILS_H

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>

namespace ark{

    typedef pcl::PointXYZRGB PointType;

    class RGBDFrame {
    public:
        cv::Mat mTcw;
        cv::Mat imRGB;
        cv::Mat imDepth;
        int frameId;
        RGBDFrame(){
            mTcw = cv::Mat::eye(4,4,CV_32FC1);
            frameId = -1;
        }
        RGBDFrame(const RGBDFrame& frame)
        {
            frame.mTcw.copyTo(mTcw);
            frame.imRGB.copyTo(imRGB);
            frame.imDepth.copyTo(imDepth);
            frameId = frame.frameId;
        }
    };
}


#endif //OPENARK_UTILS_H