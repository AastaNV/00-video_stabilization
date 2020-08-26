/*
* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/HarrisKeypointDetector.h>
#include <vpi/algo/KLTBoundingBoxTracker.h>
#include <vpi/algo/ImageFormatConverter.h>
#include <vpi/algo/PerspectiveImageWarp.h>

#define ONE_MBYTE (1024*1024)

#define CHECK_STATUS(STMT)                                                    \
    do                                                                        \
    {                                                                         \
        VPIStatus status = (STMT);                                            \
        if (status != VPI_SUCCESS)                                            \
        {                                                                     \
            std::cerr << "[ERROR] " << vpiStatusGetName(status) << std::endl; \
            exit(0);                                                          \
        }                                                                     \
    } while (0);

class myClock
{
public:
    inline void tic() { clock_gettime(CLOCK_REALTIME, &t1); };

    void toc(std::string prefix)
    {
        clock_gettime(CLOCK_REALTIME, &t2);
        double t_us = ((double)(t2.tv_sec - t1.tv_sec)) * 1000000.0 + ((double)(t2.tv_nsec - t1.tv_nsec) / 1000.0);
        std::cout << prefix << t_us;
    }

private:
    timespec t1, t2;
};

static void printMemInfo()
{
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf(" GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", used_db/ONE_MBYTE, free_db/ONE_MBYTE, total_db/ONE_MBYTE);
}

static void MatrixMultiply(VPIPerspectiveTransform &r, const VPIPerspectiveTransform &a,
                           const VPIPerspectiveTransform &b)
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            r[i][j] = a[i][0] * b[0][j];
            for (int k = 1; k < 3; ++k)
            {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}


int main(int argc, char *argv[])
{
    cv::VideoCapture video;
    if( !video.open("/opt/nvidia/vpi/samples/assets/dashcam.mp4") )
    {
        std::cerr << "[ERROR] Can't open input video" << std::endl;
        exit(0);
    }

    size_t img_w = video.get(cv::CAP_PROP_FRAME_WIDTH);
    size_t img_h = video.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::Mat ori_t0, ori_t1;
    cv::Mat img_t0, img_t1;
    cv::Mat out_t1;

    VPIStream pva     = NULL;
    VPIStream stream  = NULL;
    VPIPayload harris = NULL;
    VPIPayload klt    = NULL;
    VPIPayload warp   = NULL;

    VPIImage image_t0  = NULL;
    VPIImage image_t1  = NULL;
    VPIImage warpBGR = NULL;
    VPIImage warpIn  = NULL;
    VPIImage warpOut = NULL;

    VPIArray keypoints = NULL;
    VPIArray scores    = NULL;
    VPIArray inputBoxList  = NULL;
    VPIArray inputPredList = NULL;
    VPIArray outputBoxList   = NULL;
    VPIArray outputEstimList = NULL;

    CHECK_STATUS(vpiStreamCreate(VPI_DEVICE_TYPE_PVA, &pva));
    CHECK_STATUS(vpiStreamCreate(VPI_DEVICE_TYPE_CUDA, &stream));
    CHECK_STATUS(vpiCreateHarrisKeypointDetector(stream, img_w, img_h, &harris));
    CHECK_STATUS(vpiCreateKLTBoundingBoxTracker (stream, img_w, img_h, VPI_IMAGE_TYPE_S16, &klt));
    CHECK_STATUS(vpiCreatePerspectiveImageWarp  (pva, &warp));

    CHECK_STATUS(vpiImageCreate(img_w, img_h, VPI_IMAGE_TYPE_NV12, 0, &warpIn));
    CHECK_STATUS(vpiImageCreate(img_w, img_h, VPI_IMAGE_TYPE_NV12, 0, &warpOut));

    CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_KEYPOINT, 0, &keypoints));
    CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_U32, 0, &scores));
    CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &outputBoxList));
    CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &outputEstimList));   


    for( size_t frame_num=0; ; frame_num++ )
    {
        myClock clock;


        // read input
        if( !video.read(ori_t1) )
        {
            std::cerr << "[ERROR] Can't read video frame (EOF?)" << std::endl;
            exit(0);
        }
        cvtColor(ori_t1, img_t1, cv::COLOR_BGR2GRAY);
        img_t1.convertTo(img_t1, CV_16SC1);


        // cvMat to vpi_image
        {
            VPIImageData imgData;
            memset(&imgData, 0, sizeof(imgData));
            imgData.type                = VPI_IMAGE_TYPE_S16;
            imgData.numPlanes           = 1;
            imgData.planes[0].width     = img_t1.cols;
            imgData.planes[0].height    = img_t1.rows;
            imgData.planes[0].rowStride = img_t1.step[0];
            imgData.planes[0].data      = img_t1.data;

            CHECK_STATUS(vpiImageWrapHostMem(&imgData, 0, &image_t1));
        }

        if( frame_num == 0 ) {
            std::swap(image_t1, image_t0);
            continue;
        }


        // harris
        clock.tic();
        {
            VPIHarrisKeypointDetectorParams params;
            params.gradientSize   = 5;
            params.blockSize      = 5;
            params.strengthThresh = 40;
            params.sensitivity    = 0.01;
            params.minNMSDistance = 64; // must be 8 for PVA backend

            CHECK_STATUS(vpiSubmitHarrisKeypointDetector(harris, image_t0, keypoints, scores, &params));
            CHECK_STATUS(vpiStreamSync(stream));
        }
        clock.toc(">> Harris time: ");


        // corner -> bbox
        clock.tic();
        VPIArrayData outKeypointsData;
        CHECK_STATUS(vpiArrayLock(keypoints, VPI_LOCK_READ, &outKeypointsData));
        VPIKeypoint *kpts = (VPIKeypoint *)outKeypointsData.data;

        std::vector<VPIKLTTrackedBoundingBox> bboxes;
        std::vector<VPIHomographyTransform2D> preds;

        for( size_t i=0; i<outKeypointsData.size; i++ )
        {
            VPIKLTTrackedBoundingBox track = {};
            // scale
            track.bbox.xform.mat3[0][0] = 1;
            track.bbox.xform.mat3[1][1] = 1;
            // position
            track.bbox.xform.mat3[0][2] = float(kpts[i].x) - 15.5f;
            track.bbox.xform.mat3[1][2] = float(kpts[i].y) - 15.5f;
            // must be 1
            track.bbox.xform.mat3[2][2] = 1;

            track.bbox.width     = 32.f;
            track.bbox.height    = 32.f;
            track.trackingStatus = 0; // valid tracking
            track.templateStatus = 1; // must update
            bboxes.push_back(track);

            // Identity predicted transform.
            VPIHomographyTransform2D xform = {};
            xform.mat3[0][0]               = 1;
            xform.mat3[1][1]               = 1;
            xform.mat3[2][2]               = 1;
            preds.push_back(xform);
        }

        VPIArrayData data = {};
        data.type         = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
        data.capacity     = bboxes.capacity();
        data.size         = bboxes.capacity();
        data.data         = &bboxes[0];
        CHECK_STATUS(vpiArrayWrapHostMem(&data, 0, &inputBoxList));

        data.type = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;
        data.data = &preds[0];
        CHECK_STATUS(vpiArrayWrapHostMem(&data, 0, &inputPredList));
        clock.toc(", CPU wraper time: ");


        // KLT
        clock.tic();
        {
            VPIKLTBoundingBoxTrackerParams params = {};
            params.numberOfIterationsScaling      = 20;
            params.nccThresholdUpdate             = 0.8f;
            params.nccThresholdKill               = 0.6f;
            params.nccThresholdStop               = 1.0f;
            params.maxScaleChange                 = 0.2f;
            params.maxTranslationChange           = 5.0f;
            params.trackingType                   = VPI_KLT_INVERSE_COMPOSITIONAL;

            CHECK_STATUS(vpiSubmitKLTBoundingBoxTracker(klt, image_t0, inputBoxList, inputPredList,
                                                             image_t1,outputBoxList, outputEstimList, &params));
            CHECK_STATUS(vpiStreamSync(stream));
        }
        clock.toc(", KLT time: ");


        // Calculate global motion
        clock.tic();
        VPIPerspectiveTransform transform;
        {
            VPIArrayData updatedBBoxData;
            VPIArrayData estimData;
            CHECK_STATUS(vpiArrayLock(outputBoxList, VPI_LOCK_READ, &updatedBBoxData));
            CHECK_STATUS(vpiArrayLock(outputEstimList, VPI_LOCK_READ, &estimData));

            auto *updated_bbox = reinterpret_cast<VPIKLTTrackedBoundingBox *>(updatedBBoxData.data);
            auto *estim        = reinterpret_cast<VPIHomographyTransform2D *>(estimData.data);

            float global_mx = 0;
            float global_my = 0;
            for( size_t i=0; i<outKeypointsData.size; i++ )
            {
                float px, py;  // previous pos
                float cx, cy;  // current pos

                px = kpts[i].x;
                py = kpts[i].y;
                cx = ( updated_bbox[i].bbox.xform.mat3[0][2]+estim[i].mat3[0][2] ) + 
                     (updated_bbox[i].bbox.width*estim[i].mat3[0][0]*estim[i].mat3[0][0] )/2;
                cy = ( updated_bbox[i].bbox.xform.mat3[1][2]+estim[i].mat3[1][2] ) +
                     ( updated_bbox[i].bbox.height*estim[i].mat3[1][1]*estim[i].mat3[1][1] )/2;

                //[TODO] Add foreground removal here
                global_mx += (cx-px);
                global_my += (cy-py);
            }
            global_mx /= outKeypointsData.size;
            global_my /= outKeypointsData.size;

            // move image's center to origin of coordinate system
            VPIPerspectiveTransform t1 = {{1, 0, -float(img_w)/2.0f}, {0, 1, -float(img_h)/2.0f}, {0, 0, 1}};
            // [TODO] apply IIR filter here
            VPIPerspectiveTransform mv = {{1, 0, 0}, {0, 1, 0}, {global_mx, global_my, 1}};
            // move image's center back to where it was.
            VPIPerspectiveTransform t2 = {{1, 0,  float(img_w)/2.0f}, {0, 1,  float(img_h)/2.0f}, {0, 0, 1}};

            VPIPerspectiveTransform tmp;
            MatrixMultiply(tmp, mv, t1);
            MatrixMultiply(transform, t2, t1);

            CHECK_STATUS(vpiArrayUnlock(outputBoxList));
            CHECK_STATUS(vpiArrayUnlock(outputEstimList));
        }

        CHECK_STATUS(vpiArrayUnlock(keypoints));
        clock.toc(", motion estimation time: ");


        // warp
        {
            VPIImageData imgData;
            memset(&imgData, 0, sizeof(imgData));
            imgData.type                = VPI_IMAGE_TYPE_BGR8;
            imgData.numPlanes           = 1;
            imgData.planes[0].width     = ori_t1.cols;
            imgData.planes[0].height    = ori_t1.rows;
            imgData.planes[0].rowStride = ori_t1.step[0];
            imgData.planes[0].data      = ori_t1.data;

            CHECK_STATUS(vpiImageWrapHostMem(&imgData, 0, &warpBGR));

            clock.tic();
            CHECK_STATUS(vpiSubmitImageFormatConverter(stream, warpBGR, warpIn, VPI_CONVERSION_CAST, 1, 0));
            CHECK_STATUS(vpiStreamSync(stream));

            CHECK_STATUS(vpiSubmitPerspectiveImageWarp(warp, warpIn, transform, warpOut, VPI_INTERP_LINEAR,
                                                       VPI_BOUNDARY_COND_ZERO, 0));
            CHECK_STATUS(vpiStreamSync(pva));

            CHECK_STATUS(vpiSubmitImageFormatConverter(stream, warpOut, warpBGR, VPI_CONVERSION_CAST, 1, 0));
            CHECK_STATUS(vpiStreamSync(stream));
            clock.toc(", warping time: ");

            CHECK_STATUS(vpiImageLock(warpBGR, VPI_LOCK_READ, &imgData));
            out_t1 = cv::Mat(imgData.planes[0].height, imgData.planes[0].width, CV_8UC3, imgData.planes[0].data, imgData.planes[0].rowStride);
            CHECK_STATUS(vpiImageLock(warpBGR, VPI_LOCK_READ, &imgData));
       }

        /*
        cv::Mat display;
        cv::hconcat(ori_t1, out_t1, display);
        cv::imshow("orignal | warped", display);
        cv::waitKey(10);
        */

        std::swap(img_t0, img_t1);
        std::swap(ori_t0, ori_t1);
        std::swap(image_t0, image_t1);

        vpiArrayDestroy(inputBoxList);
        vpiArrayDestroy(inputPredList);
        vpiImageDestroy(warpBGR);
        vpiImageDestroy(image_t1);

        printMemInfo();
    }

    // Clean up
    if( stream!=NULL )
    {
        vpiStreamSync(stream);
    }
    if( pva!=NULL )
    {
        vpiStreamSync(pva);
    }

    vpiArrayDestroy(keypoints);
    vpiArrayDestroy(scores);
    vpiArrayDestroy(inputBoxList);
    vpiArrayDestroy(inputPredList);
    vpiArrayDestroy(outputBoxList);
    vpiArrayDestroy(outputEstimList);

    vpiImageDestroy(image_t0);
    vpiImageDestroy(image_t1);
    vpiImageDestroy(warpBGR);
    vpiImageDestroy(warpIn);
    vpiImageDestroy(warpOut);

    vpiPayloadDestroy(harris);
    vpiPayloadDestroy(klt);
    vpiPayloadDestroy(warp);
    vpiStreamDestroy(stream);
    vpiStreamDestroy(pva);

    std::cout << "All goods!" << std::endl;
    return 0;
}
