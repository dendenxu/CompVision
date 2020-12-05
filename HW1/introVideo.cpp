#include "include.hpp"

void resizeVideo(VideoCapture& src, VideoWriter& dst, Size size, bool preserveRatio)
{
    Mat frame, newFrame;
    while (1) {
        src >> frame;
        if (frame.empty()) break;  // check if at end
        resizeImage(frame, newFrame, size, preserveRatio);
        dst << newFrame;
    };
}
void resizeImage(Mat& src, Mat& dst, Size size, bool preserveRatio)
{
    // ? should we use dst.create here?
    dst = Mat::zeros(size, src.type());
    Point origin;
    Size imageSize;

    if (!preserveRatio) {
        imageSize = size;
        origin = Point(0, 0);
    } else {
        OUTPUTINFO << "Getting ratio: " << size.aspectRatio() << " and " << src.cols / src.rows << endl;
        if (size.aspectRatio() < src.cols / src.rows) {  // width/height
            imageSize.width = size.width;
            imageSize.height = src.rows * size.width / src.cols;
            origin.x = 0;
            origin.y = (size.height - imageSize.height) / 2;
        } else {
            imageSize.height = size.height;
            imageSize.width = src.cols * size.height / src.rows;
            origin.x = (size.width - imageSize.width) / 2;
            origin.y = 0;
        }
    }

    Rect roiRect(origin, imageSize);
    Mat roiMat = dst(roiRect);
    resize(src, roiMat, imageSize);  // resizing into the region of interest
}

int introVideo(int argc, char* argv[])
{
    string windowName = "Preview Intro Video";
    //namedWindow(windowName, WINDOW_AUTOSIZE);
    //string imageName = "lena.png";
    //Mat image = imread(imageName);
    //Mat resized;
    //resizeImage(image, resized, Size(900, 600));
    //imshow(windowName, resized);
    //waitKey();
    VideoCapture inputVideo("Megamind.avi");
    int codec = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));  // Get Codec Type- Int form
    Size size = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),     // Acquire input size
                     (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));
    Size newSize = Size(1920, 1080);
    double fps = inputVideo.get(CAP_PROP_FPS);
    VideoWriter outputVideo("Enlarged.avi", codec, fps, newSize);  // Open the output video
    resizeVideo(inputVideo, outputVideo, newSize);

    return 0;
}