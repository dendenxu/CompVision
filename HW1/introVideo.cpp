#include "include.hpp"

void resizeVideo(VideoCapture& src, VideoWriter& dst, Size size, bool preserveRatio, bool rewind)
{
    Mat frame, newFrame;
    if (rewind) src.set(CAP_PROP_POS_FRAMES, 0);  // last frame
    while (1) {
        src >> frame;
        if (frame.empty()) break;  // check if at end
        resizeImage(frame, newFrame, size, preserveRatio);
        dst << newFrame;
    };
    //return newFrame; // ! return last frame
}
void resizeImage(const Mat& src, Mat& dst, Size size, bool preserveRatio)
{
    // ? should we use dst.create here?
    dst = Mat::zeros(size, src.type());
    Point origin;
    Size imageSize;

    if (!preserveRatio) {
        imageSize = size;
        origin = Point(0, 0);
    } else {
        OUTPUTINFO << "Getting ratio: " << size.width << "/" << size.height << "=" << (double)size.width / size.height
                   << " and " << src.cols << "/" << src.rows << "=" << (double)src.cols / src.rows
                   << endl;
        if (size.aspectRatio() < (double)src.cols / src.rows) {  // width/height
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

void crossDissolve(const Mat& f1, const Mat& f2, VideoWriter& video, int count)
{
    // ! manually blending the two images
    double step = 1.0 / count;
    double weight = 1;
    Mat frame;
    // ! from 0 to 1, including 1 here
    OUTPUTINFO << "Blending"
               << " with a step of " << step
               << " for " << f1.cols << ", " << f1.rows
               << " and for " << f2.cols << ", " << f2.rows
               << endl;

    for (int i = 0; i <= count; i++, weight -= step) {
        OUTPUTINFO << "Blending at index: " << i
                   << " current weight is " << weight
                   << " and " << (1 - weight)
                   << endl;
        frame = f1 * weight + f2 * (1 - weight);
        video << frame;
    }
}

void staticImage(const Mat& image, VideoWriter& video, int count)
{
    for (int i = 0; i < count; i++) {
        video << image;
    }
}

Mat getLastFrame(VideoCapture& video)
{
    Mat last;
    video.set(CAP_PROP_POS_FRAMES, video.get(CAP_PROP_FRAME_COUNT) - 1);  // Last but one frame
    video >> last;                                                        // Get last frame
    return last;
}

Mat getFirstFrame(VideoCapture& video, bool skipBlack)
{
    Mat first;
    video.set(CAP_PROP_POS_FRAMES, 0);  // last frame

    if (skipBlack) {
        double minVal, maxVal = 0;
        while (first.empty() || maxVal == 0) {
            video >> first;
            minMaxLoc(first, &minVal, &maxVal);
            OUTPUTINFO << "Getting maxVal: " << maxVal << endl;
        }

    } else {
        video >> first;
    }
    return first;
}

string toLowerString(const string& str)
{
    string result;
    std::transform(str.begin(), str.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
}

int IntroVideo(int argc, char* argv[])
{
    // TODO: WRITE HELP MESSAGE
    if (argc < 2) {
        OUTPUTERROR << "Not enough parameters" << endl;
    }
    string path = argv[1];
    Size maxSize(0, 0);
    vector<VideoCapture> videos;
    vector<Mat> images;
    for (const auto& entry : directory_iterator(path)) {
        auto ext = entry.path().extension();  // Find extension
        OUTPUTINFO << entry.path() << " with extension: " << ext << endl;
        if (toLowerCase(ext.string()) == ".avi") {
            videos.emplace_back(entry.path().string());
            if (videos.back().isOpened()) {
                int curWidth = (int)videos.back().get(CAP_PROP_FRAME_WIDTH);
                int curHeight = (int)videos.back().get(CAP_PROP_FRAME_HEIGHT);
                if (curHeight > maxSize.height) {
                    maxSize.height = curHeight;
                }
                if (curWidth > maxSize.width) {
                    maxSize.width = curWidth;
                }
            } else {
                OUTPUTERROR << "Cannot open video: " << entry.path() << endl;
            }
        } else {
            // MARK: RECOGNIZING EVERYTHING ELSE AS IMAGES!
            //images.emplace_back(entry.path().string());
            Mat image;
            image = imread(entry.path().string());
            images.push_back(image);
            if (!images.back().empty()) {
                int curWidth = images.back().cols;
                int curHeight = images.back().rows;
                if (curHeight > maxSize.height) {
                    maxSize.height = curHeight;
                }
                if (curWidth > maxSize.width) {
                    maxSize.width = curWidth;
                }
            } else {
                OUTPUTERROR << "Cannot open image: " << entry.path() << endl;
            }
        }
    }

    OUTPUTINFO << "Getting size " << maxSize << endl;

    int codec = VideoWriter::fourcc('H', '2', '6', '4');
    double fps = 29.97;
    if (videos.empty()) {
        OUTPUTERROR << "NO VIDEO IS PROVIDED, USING DEFAULT FPS/CODEC" << endl;
    } else {
        // MARK: USING THE FIRST VIDEO'S CODEC ANF FPS
        // MARK: ASSUMING AT LEAST ONE VIDEO
        codec = static_cast<int>(videos.front().get(CAP_PROP_FOURCC));  // Get Codec Type- Int form
        fps = videos.front().get(CAP_PROP_FPS);
    }

    string outputPath = (fs::path(path) / "OUTPUT.AVI").string();

    VideoWriter output(outputPath, codec, fps, maxSize, true);
    if (!output.isOpened()) {
        OUTPUTERROR << "Cannot open " << outputPath << " for output" << endl;
        return 1;
    }

    randomInit(maxSize, 100, output);
    IntroRandom(argc, argv);


    const int blendFrame = (int)(1.5 * fps);
    const int staticFrame = (int)(1.5 * fps);

    //namedWindow("prev", WINDOW_AUTOSIZE);
    //namedWindow("curr", WINDOW_AUTOSIZE);

    Mat prev = getRandomLastFrame();
    //images.push_back(prev.clone());
    Mat resized;
    for (const auto& curr : images) {
        resizeImage(curr, resized, maxSize);

        //imshow("prev", prev);
        //imshow("curr", resized);
        //waitKey();

        crossDissolve(prev, resized, output, blendFrame);
        staticImage(resized, output, staticFrame);
        resized.copyTo(prev);
    }

    for (auto& curr : videos) {
        resizeImage(getFirstFrame(curr), resized, maxSize);  // resizing is done in place
        crossDissolve(prev, resized, output, blendFrame);
        resizeVideo(curr, output, maxSize);
        resizeImage(getLastFrame(curr), resized, maxSize);
        resized.copyTo(prev);
    }
    crossDissolve(prev, Mat::zeros(maxSize, CV_8UC3), output, blendFrame);
    return 0;
}