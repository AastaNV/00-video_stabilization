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
#include <cuda_runtime.h>

#include <opencv2/highgui.hpp>
#include "opencv2/cudaimgproc.hpp"

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/HarrisKeypointDetector.h>
#include <vpi/algo/KLTBoundingBoxTracker.h>
#include <vpi/algo/ImageFormatConverter.h>
#include <vpi/algo/PerspectiveImageWarp.h>
#include <cudaUtility.cuh>


int main(int argc, char *argv[])
{
    const char* gst = "filesrc location=/opt/nvidia/vpi/samples/assets/dashcam.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! video/x-raw "
                      "! videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture video(gst, cv::CAP_GSTREAMER);
    if( !video.isOpened() )
    {
        std::cerr << "[ERROR] Can't open input video" << std::endl;
        exit(0);
    }

    size_t img_w  = video.get(cv::CAP_PROP_FRAME_WIDTH);
    size_t img_h  = video.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::Mat ori, img, out;
    cv::cuda::GpuMat img_gpu, img_gpu_pre, ori_gpu;

    void *kpts_buf;
    void *input_box_buf, *input_pred_buf;
    void *output_box_buf, *output_esti_buf;

    VPIStream pva     = NULL;
    VPIStream stream  = NULL;
    VPIPayload harris = NULL;
    VPIPayload klt    = NULL;
    VPIPayload warp   = NULL;

    VPIImage image    = NULL;
    VPIImage imagePre = NULL;
    VPIImage warpBGR  = NULL;
    VPIImage warpNV12 = NULL;
    VPIImage warpOut  = NULL;

    VPIArray keypoints = NULL;
    VPIArray scores    = NULL;
    VPIArray inputBoxList  = NULL;
    VPIArray inputPredList = NULL;
    VPIArray outputBoxList   = NULL;
    VPIArray outputEstimList = NULL;

    // prepare CUDA stream
    cudaStream_t cuda_stream;
    CHECK(cudaStreamCreate(&cuda_stream));
    CHECK_STATUS(vpiStreamWrapCuda(cuda_stream, &stream));
    CHECK_STATUS(vpiStreamCreate(VPI_DEVICE_TYPE_PVA, &pva));

    CHECK_STATUS(vpiCreateHarrisKeypointDetector(stream, img_w, img_h, &harris));
    CHECK_STATUS(vpiCreateKLTBoundingBoxTracker (stream, img_w, img_h, VPI_IMAGE_TYPE_S16, &klt));
    CHECK_STATUS(vpiCreatePerspectiveImageWarp  (pva, &warp));

    CHECK_STATUS(vpiImageCreate(img_w, img_h, VPI_IMAGE_TYPE_NV12, 0, &warpNV12));
    CHECK_STATUS(vpiImageCreate(img_w, img_h, VPI_IMAGE_TYPE_NV12, 0, &warpOut));

    // prepare VPI Array
    CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_U32, 0, &scores));
    {
        VPIArrayData kpData;
        kpData.capacity = 8192;
        kpData.size     = 0;
        kpData.stride   = sizeof(VPIKeypoint);
        kpData.type     = VPI_ARRAY_TYPE_KEYPOINT;

        cudaMalloc( (void**)&kpts_buf, kpData.stride*kpData.capacity);
        kpData.data = kpts_buf;
        CHECK_STATUS(vpiArrayWrapCudaDeviceMem(&kpData, 0, &keypoints));


        kpData.capacity = 128;
        kpData.size     = 0;
        kpData.stride   = sizeof(VPIKLTTrackedBoundingBox);
        kpData.type     = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;

        cudaMalloc( (void**)&input_box_buf, kpData.stride*kpData.capacity);
        kpData.data = input_box_buf;
        CHECK_STATUS(vpiArrayWrapCudaDeviceMem(&kpData, 0, &inputBoxList));

        kpData.capacity = 128;
        kpData.size     = 0;
        kpData.stride   = sizeof(VPIHomographyTransform2D);
        kpData.type     = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;

        cudaMalloc( (void**)&input_pred_buf, kpData.stride*kpData.capacity);
        kpData.data = input_pred_buf;
        CHECK_STATUS(vpiArrayWrapCudaDeviceMem(&kpData, 0, &inputPredList));

        kpData.capacity = 128;
        kpData.size     = 0;
        kpData.stride   = sizeof(VPIKLTTrackedBoundingBox);
        kpData.type     = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;

        cudaMalloc( (void**)&output_box_buf, kpData.stride*kpData.capacity);
        kpData.data = output_box_buf;
        CHECK_STATUS(vpiArrayWrapCudaDeviceMem(&kpData, 0, &outputBoxList));

        kpData.capacity = 128;
        kpData.size     = 0;
        kpData.stride   = sizeof(VPIHomographyTransform2D);
        kpData.type     = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;

        cudaMalloc( (void**)&output_esti_buf, kpData.stride*kpData.capacity);
        kpData.data = output_esti_buf;
        CHECK_STATUS(vpiArrayWrapCudaDeviceMem(&kpData, 0, &outputEstimList));
    }


    for( size_t frame_num=0; ; frame_num++ )
    {
        myClock clock;


        // Read input from OpenCV
        if( !video.read(ori) )
        {
            std::cerr << "[ERROR] Can't read video frame (EOF?)" << std::endl;
            exit(0);
        }

        cvtColor(ori, img, cv::COLOR_BGR2GRAY);
        img.convertTo(img, CV_16SC1);
        img_gpu.upload(img);
        ori_gpu.upload(ori);


        // Convert image to VPI
        {
            VPIImageData imgData;
            memset(&imgData, 0, sizeof(imgData));
            imgData.type                = VPI_IMAGE_TYPE_S16;
            imgData.numPlanes           = 1;
            imgData.planes[0].width     = img_gpu.cols;
            imgData.planes[0].height    = img_gpu.rows;
            imgData.planes[0].rowStride = img_gpu.step;
            imgData.planes[0].data      = img_gpu.data;

            CHECK_STATUS(vpiImageWrapCudaDeviceMem(&imgData, VPI_ARRAY_ONLY_CUDA, &image));

            memset(&imgData, 0, sizeof(imgData));
            imgData.type                = VPI_IMAGE_TYPE_BGR8;
            imgData.numPlanes           = 1;
            imgData.planes[0].width     = ori_gpu.cols;
            imgData.planes[0].height    = ori_gpu.rows;
            imgData.planes[0].rowStride = ori_gpu.step;
            imgData.planes[0].data      = ori_gpu.data;
            CHECK_STATUS(vpiImageWrapCudaDeviceMem(&imgData, VPI_ARRAY_ONLY_CUDA, &warpBGR));
        }

        if( frame_num == 0 ) {
            std::swap(image, imagePre);
            continue;
        }


        // Harris
        clock.total_tic();
        clock.tic();
        {
            VPIHarrisKeypointDetectorParams params;
            params.gradientSize   = 5;
            params.blockSize      = 5;
            params.strengthThresh = 40;
            params.sensitivity    = 0.01;
            params.minNMSDistance = 64; // must be 8 for PVA backend


           CHECK_STATUS(vpiSubmitHarrisKeypointDetector(harris, imagePre, keypoints, scores, &params));
           CHECK_STATUS(vpiStreamSync(stream));
        }
        clock.toc(">> Harris time: ");


        // Convert feature to bbox
        clock.tic();
        {
            uint32_t size;
            vpiArrayGetSize(keypoints, &size);
            vpiArraySetSize(inputBoxList, size);
            vpiArraySetSize(inputPredList, size);

            cuda_feature2bbox(cuda_stream, kpts_buf, input_box_buf, input_pred_buf, size);
        }
        clock.toc(", CUDA wraper time: ");


        // KLT
        clock.tic();
        {
            VPIKLTBoundingBoxTrackerParams params = {};
            params.numberOfIterationsScaling      = 20;
            params.nccThresholdUpdate             = 0.6f;
            params.nccThresholdKill               = 0.2f;
            params.nccThresholdStop               = 0.8f;
            params.maxScaleChange                 = 5.0f;
            params.maxTranslationChange           = 100.0f;
            params.trackingType                   = VPI_KLT_INVERSE_COMPOSITIONAL;
            CHECK_STATUS(vpiSubmitKLTBoundingBoxTracker(klt, imagePre, inputBoxList, inputPredList,
                                                        image, outputBoxList, outputEstimList, &params));
            CHECK_STATUS(vpiStreamSync(stream));
        }
        clock.toc(", KLT time: ");


        // calculate global motion
        clock.tic();
        VPIPerspectiveTransform transform;
        {
            VPIArrayData outKeypointsData;
            VPIArrayData updatedBBoxData;
            VPIArrayData estimData;

            CHECK_STATUS(vpiArrayLock(keypoints, VPI_LOCK_READ, &outKeypointsData));
            CHECK_STATUS(vpiArrayLock(outputBoxList, VPI_LOCK_READ, &updatedBBoxData));
            CHECK_STATUS(vpiArrayLock(outputEstimList, VPI_LOCK_READ, &estimData));

            auto *kpts         = reinterpret_cast<VPIKeypoint *>(outKeypointsData.data);
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
            CHECK_STATUS(vpiArrayUnlock(keypoints));
        }
        clock.toc(", motion estimation time: ");


        // warp
        clock.tic();
        {
            CHECK_STATUS(vpiSubmitImageFormatConverter(stream, warpBGR, warpNV12, VPI_CONVERSION_CAST, 1, 0));
            CHECK_STATUS(vpiStreamSync(stream));

            CHECK_STATUS(vpiSubmitPerspectiveImageWarp(warp, warpNV12, transform, warpOut, VPI_INTERP_LINEAR,
                                                       VPI_BOUNDARY_COND_ZERO, 0));
            CHECK_STATUS(vpiStreamSync(pva));

            CHECK_STATUS(vpiSubmitImageFormatConverter(stream, warpOut, warpBGR, VPI_CONVERSION_CAST, 1, 0));
            CHECK_STATUS(vpiStreamSync(stream));
        }
        clock.toc(", warping time: ");
        clock.total_toc(", total time: ");
        printMemInfo();

        ori_gpu.download(out);

/*
        cv::Mat display;
        cv::hconcat(ori, out, display);
        cv::imshow("orignal | warped", display);
        cv::waitKey(10);
*/

        std::swap(image, imagePre);
        std::swap(img_gpu, img_gpu_pre);
        vpiImageDestroy(warpBGR);
        vpiImageDestroy(image);
    }


    // Clean up
    if( stream!=NULL )
    {
        vpiStreamSync(stream);
    }

    vpiArrayDestroy(keypoints);
    vpiArrayDestroy(scores);
    vpiArrayDestroy(inputBoxList);
    vpiArrayDestroy(inputPredList);
    vpiArrayDestroy(outputBoxList);
    vpiArrayDestroy(outputEstimList);

    vpiImageDestroy(image);
    vpiImageDestroy(imagePre);
    vpiImageDestroy(warpBGR);
    vpiImageDestroy(warpNV12);
    vpiImageDestroy(warpOut);

    vpiPayloadDestroy(harris);
    vpiPayloadDestroy(klt);
    vpiPayloadDestroy(warp);

    cudaFree(kpts_buf);
    cudaFree(input_box_buf);
    cudaFree(input_pred_buf);
    cudaFree(output_box_buf);
    cudaFree(output_esti_buf);

    vpiStreamDestroy(stream);
    vpiStreamDestroy(pva);
    cudaStreamDestroy(cuda_stream);
    return 0;
}
