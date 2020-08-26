# VPI sample for video stabilization

### Environment
* Jetson Xavier or XavierNX
* JetPack 4.4 Product Release (GA)
* VPI-0.3

### Build
```
$ git clone https://github.com/AastaNV/00-video_stabilization.git
$ cd 00-video_stabilization
$ cmake .
$ make
$ ./vpi_sample_stabilization
```

### Pipeline
#### Step 1:  mount image from *cv::Mat* into *VPIImage*
- Use S16 gray scale image format.

#### Step 2: Harris key-point detector
- Use CUDA to allow larger minNMSDistance (PVA only support minNMSDistance=8)
- Apply algorithm on S16 image format

#### Step 3: wrap *VPI_ARRAY_TYPE_KEYPOINT* into *VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX*
- Wrap bbox by making a fake bounding box center with keypoint with w=h=32

#### Step 4: KLT tracker
- Use CUDA
- Apply algorithm on S16 image format

#### Step 5: calculate transform matrix
- Use global translation as approximate
        - [TODO] User should add foreground removal and IIR filter matrix on their own
- Matrix need to apply on the center of the image. Transform will look like this:
```
         | 1  0  2/w |     | 1  1  mx |     | 1  0  -2/w |
     T = | 0  1  2/h |  *  | 0  1  my |  *  | 0  1  -2/h |
         | 0  0   1  |     | 0  0   1 |     | 0  0    1  |
```
- [TODO] User should add foreground removal and IIR filter on their own.

#### Step 6: warping
- Use PVA
 - Apply algorithm on NV12 image format
- Flow should look like this
> (OpenCV BGR) 
-> [ImageFormatConverter](http://) -> (VPI NV12) 
-> [PerspectiveImageWarp](http://)  -> (VPI NV12) 
-> [ImageFormatConverter](http://) -> (OpenCV BGR)
