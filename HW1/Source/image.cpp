#include "include.hpp"

int ImageShow(int argc, char* argv[])
{
    if (argc < 2) {
        OUTPUTERROR << "Not enough parameters" << endl;
    }
    string imgName = argv[1];
    Mat img = imread(samples::findFile(imgName));
    if (img.empty()) {
        OUTPUTERROR << "Cannot find image file: " << imgName << endl;
    }

    Mat grey;
    cvtColor(img, grey, COLOR_BGR2GRAY);
    Mat sobel;
    Sobel(grey, sobel, CV_32F, 1, 0);
    double minVal, maxVal;
    minMaxLoc(sobel, &minVal, &maxVal);
    Mat draw;
    double scale = 255.0 / (maxVal - minVal), delta = -minVal * scale;
    sobel.convertTo(draw, CV_8U, scale, delta);
    string imgNameNoExt = imgName.substr(0, imgName.find_last_of('.'));
    string WIN_GREY = "Original Grey: " + imgNameNoExt;
    string WIN_SOBELGREY = "SOBEL Grey: " + imgNameNoExt;
    namedWindow(WIN_GREY, WINDOW_AUTOSIZE);
    namedWindow(WIN_SOBELGREY, WINDOW_AUTOSIZE);
    moveWindow(WIN_GREY, 400, 100);
    moveWindow(WIN_SOBELGREY, grey.cols + 400, 100);
    imshow(WIN_GREY, grey);
    imshow(WIN_SOBELGREY, draw);
    waitKey();
}

void imageFilterHelp(char* progName)
{
    cout << endl
         << "This program shows how to filter images with mask: the write it yourself and the"
         << "filter2d way. " << endl
         << "Usage:" << endl
         << progName << " [image_path -- default lena.jpg] [G -- grayscale] " << endl
         << endl;
}

int ImageFilter(int argc, char* argv[])
{
    const char* filename = (argc >= 2) ? argv[1] : "lena.png";
    Mat src, dst0, dst1;
    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    else
        src = imread(samples::findFile(filename), IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Can't open image [" << filename << "]" << endl;
        return EXIT_FAILURE;
    }
    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Input", src);
    double t = (double)getTickCount();
    sharpen(src, dst0);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Hand written function time passed in seconds: " << t << endl;
    imshow("Output", dst0);
    waitKey();
    Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
                  -1, 5, -1,
                  0, -1, 0);
    t = (double)getTickCount();
    filter2D(src, dst1, src.depth(), kernel);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Built-in filter2D time passed in seconds:     " << t << endl;
    imshow("Output", dst1);
    waitKey();
    return EXIT_SUCCESS;
}

void sharpen(const Mat& img, Mat& result)
{
    CV_Assert(img.depth() == CV_8U);  // accept only uchar images
    const int nChannels = img.channels();
    result.create(img.size(), img.type());
    for (int j = 1; j < img.rows - 1; ++j) {
        const uchar* previous = img.ptr<uchar>(j - 1);
        const uchar* current = img.ptr<uchar>(j);
        const uchar* next = img.ptr<uchar>(j + 1);
        uchar* output = result.ptr<uchar>(j);
        for (int i = nChannels; i < nChannels * (img.cols - 1); ++i) {
            *output++ =
                // ! Using saturate_cast to clip to bounds
                saturate_cast<uchar>(5 * current[i] - current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
        }
    }
    result.row(0).setTo(Scalar(0));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));
}