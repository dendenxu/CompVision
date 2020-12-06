#include "include.hpp"

/** Add a single line caption to a video
 * Note that the fontScale and fontThickness is in proportion to the video's size
 * We'd open a window "Preview" to preview the generated video
 * ! Speed of "Preview" is NOT the actuall FPS of the output video
 */
void addCaption(string text, VideoCapture& src, VideoWriter& dst)
{
    int width = (int)src.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)src.get(CAP_PROP_FRAME_HEIGHT);
    double fps = src.get(CAP_PROP_FPS);

    int fontScale = 1.0 / 1080 * width;
    int fontThickness = 2.0 / 1080 * width;
    int offset = 40.0 / 1080 * height;
    Size textsize = getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, fontThickness, 0);
    Point org((width - textsize.width) / 2, (height - textsize.height) - offset);
    Mat frame;
    int frameCount = 0;
    namedWindow("Preview");
    while (1) {
        src >> frame;
        if (frame.empty()) {
            break;
        }
        OUTPUTINFO << "Writing caption \"" << text << "\" to frame #" << dec << frameCount << endl;
        putText(frame, text, org, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, Scalar(255, 255, 255), fontThickness, LINE_AA);
        imshow("Preview", frame);
        dst << frame;
        frameCount++;
        // MARK: 1 ms
        waitKey(1);
    }
}

/** Resize a video
 * You can choose to preserve its aspect ratio
 * And you can also choose to rewind the video to the beginning
 * before the resizing starts
 */
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

/** Resize a image (Mat)
 * You can choose to preserve aspect ratio
 */
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

/** Blend two frames: f1 and f2
 * With cross-dissolve and write the generated sequence to video
 * count is the number of blending frame you need
 */
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

/** Print a static image to a video writer
 * count is the number of frames the image is to be printed
 */
void staticImage(const Mat& image, VideoWriter& video, int count)
{
    for (int i = 0; i < count; i++) {
        video << image;
    }
}

/** Get the last frame of a video
 * Note that the VideoCapture object's video pointer will be modified
 */
Mat getLastFrame(VideoCapture& video)
{
    Mat last;
    video.set(CAP_PROP_POS_FRAMES, video.get(CAP_PROP_FRAME_COUNT) - 1);  // Last but one frame
    video >> last;                                                        // Get last frame
    return last;
}

/** Get the first (non-black) frame of a video
 * You can choose to skip the first few black frames
 */
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

/** Convert a string to an all lower case string
 * used to compare file extension
 * a duplicate with cv::toLowerCase
 */
string toLowerString(const string& str)
{
    string result;
    std::transform(str.begin(), str.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
}

/** Main function of generating a intro video for homework 1 */
int IntroVideo(int argc, char* argv[])
{
    // TODO: WRITE HELP MESSAGE
    if (argc < 2) {
        OUTPUTERROR << "Not enough parameters" << endl;
    }
    string path = argv[1];
    Size maxSize(0, 0);
    vector<VideoCapture> videos;  // all videos in the folder
    vector<Mat> images;           // all images in the fiolder

    /** Get the maximum width/height */
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

    /** Properties to be used in VideoWriter */
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

    // MARK: 1 ms per frame generation
    randomInit(maxSize, 100, 1, output); /** Set static variable for random.cpp */
    IntroRandom(argc, argv);             /** Generate random introduction for the video */

    const int blendFrame = (int)(1.5 * fps);
    const int staticFrame = (int)(1.5 * fps);

    Mat prev = getRandomLastFrame();
    //images.push_back(prev.clone());
    Mat resized;
    for (const auto& curr : images) {
        resizeImage(curr, resized, maxSize);
        crossDissolve(prev, resized, output, blendFrame); /** Add cross-dissolve blending */
        staticImage(resized, output, staticFrame);        /** Make the image last a while */
        resized.copyTo(prev);                             // ! if we use assignment here, prev will use the same underlying matrix as resized
    }

    for (auto& curr : videos) {
        resizeImage(getFirstFrame(curr), resized, maxSize);  // resizing is done in place
        crossDissolve(prev, resized, output, blendFrame);    /** Add cross-dissolve blending */
        resizeVideo(curr, output, maxSize);                  /** Generated the resized video */
        resizeImage(getLastFrame(curr), resized, maxSize);   /** Get the last frame to be cross-dissolved later */
        resized.copyTo(prev);                                // ! if we use assignment here, prev will use the same underlying matrix as resized
    }
    crossDissolve(prev, Mat::zeros(maxSize, CV_8UC3), output, blendFrame);

    output.release();  // ! close the new video for caption

    VideoCapture raw(outputPath); // Reopen the closed video for input
    if (!raw.isOpened()) {
        OUTPUTERROR << "Cannot open " << outputPath << " for reading" << endl;
    }
    string outputPathCaption = (fs::path(path) / "OUTPUT_CAPTION.AVI").string();
    output = VideoWriter(outputPathCaption, codec, fps, maxSize, true);  // reassign the output variable

    addCaption("3180105504 Xu Zhen Presents", raw, output);

    return 0;
}