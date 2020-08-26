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

#ifndef __CUDA_UTILITY_CUH_
#define __CUDA_UTILITY_CUH_

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vpi/Array.h>

#define ONE_MBYTE (1024*1024)
#define THREAD 32

#define CHECK(status)                                                   \
    do                                                                  \
    {                                                                   \
        auto ret = (status);                                            \
        if (ret != 0)                                                   \
        {                                                               \
            std::cerr << "[ERROR] Cuda failure: " << ret << std::endl;  \
            exit(0);                                                    \
        }                                                               \
    } while (0)

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
    inline void total_tic() { clock_gettime(CLOCK_REALTIME, &total_t1); };

    void toc(std::string prefix)
    {
        clock_gettime(CLOCK_REALTIME, &t2);
        double t_us = ((double)(t2.tv_sec - t1.tv_sec)) * 1000000.0 + ((double)(t2.tv_nsec - t1.tv_nsec) / 1000.0);
        std::cout << prefix << t_us;
    }

    void total_toc(std::string prefix)
    {
        clock_gettime(CLOCK_REALTIME, &total_t2);
        double t_us = ((double)(total_t2.tv_sec - total_t1.tv_sec)) * 1000000.0 + ((double)(total_t2.tv_nsec - total_t1.tv_nsec) / 1000.0);
        std::cout << prefix << t_us;
    }

private:
    timespec t1, t2;
    timespec total_t1, total_t2;
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

int cuda_feature2bbox(cudaStream_t &stream, void *kpts, void *input_box, void *input_pred, uint32_t size);

#endif
