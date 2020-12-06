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

void blendFrames(const Mat& f1, const Mat& f2, VideoWriter& video, int count)
{
    // ! manually blending the two images
    double step = 1.0 / count;
    double weight = 0;
    Mat frame;
    for (int i = 0; i < count; i++, weight += step) {
        OUTPUTINFO << "Blending at index: " << i << " with a step of " << step << endl;
        frame = f1 * weight + f2 * (1 - weight);
        video << frame;
    }
}

int introVideo(int argc, char* argv[])
{
    VideoCapture inputVideo("Megamind.avi");
    int codec = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));  // Get Codec Type- Int form
    Size size = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),     // Acquire input size
                     (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));
    Size newSize = Size(1920, 1080);
    double fps = inputVideo.get(CAP_PROP_FPS);
    VideoWriter outputVideo("Enlarged.avi", codec, fps, newSize);  // Open the output video
    resizeVideo(inputVideo, outputVideo, newSize);

    string previewWindow = "Preview Intro Video";
    namedWindow(previewWindow, WINDOW_AUTOSIZE);
    string imageName = "lena.png";
    Mat image = imread(imageName);
    Mat resized;
    resizeImage(image, resized, newSize);
    Mat grey, sobel;
    cvtColor(image, grey, COLOR_BGR2GRAY);  // Getting gray scale image
    Sobel(grey, sobel, CV_32F, 1, 0);       // Gray scale sobel
    double minVal, maxVal;
    minMaxLoc(sobel, &minVal, &maxVal);
    Mat draw;
    double scale = 255.0 / (maxVal - minVal), delta = -minVal * scale;
    sobel.convertTo(draw, CV_8U, scale, delta);

    OUTPUTINFO << "Getting " << draw.channels() << " channels in gray sobel" << endl;
    vector<Mat> chan3;
    for (int i = 0; i < 3; i++) {
        chan3.push_back(draw);
    }
    merge(chan3, draw);  // 3 chan gray scale sobel
    OUTPUTINFO << "Getting " << draw.channels() << " channels in new sobel" << endl;

    Mat resizedSobel;
    resizeImage(draw, resizedSobel, newSize);

    imshow(previewWindow, resized);

    string sobelWindow = "Sobel Image";
    imshow(sobelWindow, resizedSobel);

    blendFrames(resized, resizedSobel, outputVideo, 0.5 * fps);

    waitKey();

    return 0;
}