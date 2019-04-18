//
// Created by lucas on 1/30/18.
//

#ifndef MODEL_ACQUISITION_MATHUTILS_H
#define MODEL_ACQUISITION_MATHUTILS_H

#include <opencv2/opencv.hpp>

namespace ark{

    cv::Vec3f rotate(const cv::Matx33f Rwc, const cv::Vec3f p){
        return cv::Vec3f(Rwc(0,0)*p[0] + Rwc(0,1)*p[1] + Rwc(0,2)*p[2],
                         Rwc(1,0)*p[0] + Rwc(1,1)*p[1] + Rwc(1,2)*p[2],
                         Rwc(2,0)*p[0] + Rwc(2,1)*p[1] + Rwc(2,2)*p[2]);
    }

    cv::Vec3f translate(const cv::Vec3f twc, const cv::Vec3f p){
        return cv::Vec3f(twc[0]+p[0], twc[1]+p[1], twc[2]+p[2]);
    }

    cv::Vec3f transform(const cv::Matx33f Rwc, const cv::Vec3f twc, const cv::Vec3f p){
        return translate(twc, rotate(Rwc, p));
    }


}

#endif //MODEL_ACQUISITION_MATHUTILS_H
