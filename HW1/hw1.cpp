#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#define OUTPUTERROR (cout << "[ERROR] " << __func__ << ": ")
#define OUTPUTWARNING (cout << "[WARNING] " << __func__ << ": ")
#define OUTPUTINFO (cout << "[INFO] " << __func__ << ": ")

constexpr auto AtomAndRookWidth = 400;
constexpr auto ColorDarkBlue = 0x3d7ea6;
constexpr auto ColorLightBlue = 0x5c969e;
constexpr auto ColorDarkPink = 0xffa5a5;
constexpr auto ColorLightPink = 0xfcdada;

using namespace cv;
using namespace std;

Scalar HexString2ColorScalar(string str);
Scalar Hex2ColorScalar(unsigned long color, bool hasAlpha = false);
void MyEllipse(Mat img, double angle, int width = AtomAndRookWidth);
void MyFilledCircle(Mat img, Point center, int width = AtomAndRookWidth / 32);
void MyLine(Mat img, Point start, Point end);
void MyPolygon(Mat img);
void DrawCircles();
void DrawAtomAndRook();

int VideoIO(int argc, char* argv[]);
int VideoCompare(int argc, char* argv[]);

void VideoIOHelpMessage();
void VideoCompareHelpMessage();

double getPSNR(const Mat& I1, const Mat& I2);
Scalar getMSSIM(const Mat& I1, const Mat& I2);

int imageFilter(int argc, char* argv[]);
void sharpen(const Mat& img, Mat& result);
static void imageFilterHelp(char* progName);

void VideoIOHelpMessage()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "This program shows how to write video files." << endl
        << "You can extract the R or G or B color channel of the input video." << endl
        << "Usage:" << endl
        << "./video-write <input_video_name> [R|G|B] [Y|N]" << endl
        << "------------------------------------------------------------------------------" << endl
        << endl;
}

void VideoCompareHelpMessage()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "This program shows how to read a video file with OpenCV. In addition, it "
        << "tests the similarity of two input videos first with PSNR, and for the frames "
        << "below a PSNR trigger value, also with MSSIM." << endl
        << "Usage:" << endl
        << "./video-input-psnr-ssim <referenceVideo> <useCaseTestVideo> <PSNR_Trigger_Value> <Wait_Between_Frames> " << endl
        << "--------------------------------------------------------------------------" << endl
        << endl;
}

int main(int argc, char* argv[])
{
    return imageFilter(argc, argv);
    //return VideoIO(argc, argv);
    //return VideoCompare(argc, argv);
    //DrawCircles();
    //DrawAtomAndRook();
    // waitKey(0);
}

static void imageFilterHelp(char* progName)
{
    cout << endl
        << "This program shows how to filter images with mask: the write it yourself and the"
        << "filter2d way. " << endl
        << "Usage:" << endl
        << progName << " [image_path -- default lena.jpg] [G -- grayscale] " << endl
        << endl;
}

int imageFilter(int argc, char* argv[])
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
                saturate_cast<uchar>
                (5 * current[i] - current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
        }
    }
    result.row(0).setTo(Scalar(0));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));
}

int VideoCompare(int argc, char* argv[])
{
    VideoCompareHelpMessage();
    if (argc != 5) {
        cout << "Not enough parameters" << endl;
        return -1;
    }

    stringstream conversion;

    const string sourceReference = argv[1], sourceCompareWith = argv[2];
    int psnrTriggerValue, delay;
    conversion << argv[3] << endl
        << argv[4];  // put in the strings

    conversion >> psnrTriggerValue >> delay;  // take out the numbers

    int frameNumber = -1;  // Frame counter

    VideoCapture captureReference(sourceReference), captureUnderTest(sourceCompareWith);

    if (!captureReference.isOpened()) {
        OUTPUTERROR
            << "Could not open reference " << sourceReference << endl;
        return -1;
    }
    if (!captureUnderTest.isOpened()) {
        OUTPUTERROR
            << "Could not open case test " << sourceCompareWith << endl;
        return -1;
    }

    Size referenceSize = Size(static_cast<int>(captureReference.get(CAP_PROP_FRAME_WIDTH)), static_cast<int>(captureReference.get(CAP_PROP_FRAME_HEIGHT)));
    Size underTestSize = Size(static_cast<int>(captureUnderTest.get(CAP_PROP_FRAME_WIDTH)), static_cast<int>(captureUnderTest.get(CAP_PROP_FRAME_HEIGHT)));

    double frameTime = 1 / captureReference.get(CAP_PROP_FPS) * 1000;
    OUTPUTINFO
        << "frameTime is " << frameTime << "ms" << endl;

    if (referenceSize != underTestSize) {
        OUTPUTERROR
            << "Inputs have different size [referenceSize, underTestSize]: " << referenceSize << ", " << underTestSize << ". Closing..." << endl;
        return -1;
    }

    const char* WINDOW_REFERENCE = "Reference Video";
    const char* WINDOW_UNDERTEST = "Under Test Video";

    // Windows Definition and Positioning
    namedWindow(WINDOW_REFERENCE, WINDOW_AUTOSIZE);
    namedWindow(WINDOW_UNDERTEST, WINDOW_AUTOSIZE);
    moveWindow(WINDOW_REFERENCE, 400, 0);
    moveWindow(WINDOW_UNDERTEST, referenceSize.width + 400, 0);

    OUTPUTINFO
        << "Reference frame resolution: Width=" << referenceSize.width << ", Height=" << referenceSize.height << " of frame count: " << captureReference.get(CAP_PROP_FRAME_COUNT) << endl;

    OUTPUTINFO
        << "PSNR trigger value: " << fixed << setprecision(3) << psnrTriggerValue << endl;

    Mat frameReference, frameUnderTest;
    double psnrValue;
    Scalar mssimValue;

    while (1) {
        auto start = chrono::steady_clock::now();

        captureReference >> frameReference;
        captureUnderTest >> frameUnderTest;

        if (frameReference.empty() || frameUnderTest.empty()) {
            cout << " < < < Game over! > > > ";
            break;
        }

        frameNumber++;
        OUTPUTINFO
            << "Frame: " << frameNumber << "# ";

        psnrValue = getPSNR(frameReference, frameUnderTest);

        // OUTPUTINFO
        cout << fixed << setprecision(3) << psnrValue << "dB";

        if (psnrValue < psnrTriggerValue && psnrValue) {
            mssimValue = getMSSIM(frameReference, frameUnderTest);

            cout << " MSSIM: "
                << " R " << setiosflags(ios::fixed) << setprecision(2) << mssimValue.val[2] * 100 << "%"
                << " G " << setiosflags(ios::fixed) << setprecision(2) << mssimValue.val[1] * 100 << "%"
                << " B " << setiosflags(ios::fixed) << setprecision(2) << mssimValue.val[0] * 100 << "%";
        }

        cout << "; ";

        imshow(WINDOW_REFERENCE, frameReference);
        imshow(WINDOW_UNDERTEST, frameReference);

        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapsed_seconds = end - start;
        //! not using delay here
        auto timeToWait = frameTime - elapsed_seconds.count() * 1000;

        OUTPUTINFO
            << "TimeToWait: " << timeToWait << "ms";

        cout << endl;
        if (timeToWait > 0) {
            char c = static_cast<char>(waitKey(timeToWait));
            if (c == 27) break;  // ?
        }
    }
    return 0;
}

// ![get-psnr]: Peak signal-to-noise ratio
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);

    Scalar s = sum(s1);

    double sse = s.val[0] + s.val[1] + s.val[2];  // sum channels

    if (sse <= 1e-10) {
        return 0;  // for small values, we return zero
    }
    else {
        double mse = sse / static_cast<double>(s1.channels() * s1.total());
        double psnr = 10.0 * log10(255.0 * 255.0 / mse);  // Peak Signal-to-noise Ratio
        return psnr;
    }
}
// ![get-psnr]: Peak signal-to-noise ratio

// ![get-mssim]: Structural similarity: Image quality assessment from error visibility to structural similarity
Scalar getMSSIM(const Mat& I1, const Mat& I2)
{
    const double C1 = (255 * 0.01) * (255 * 0.01);  // 6.5025
    const double C2 = (255 * 0.03) * (255 * 0.03);  // 58.5225

    int d = CV_32F;
    Mat I1_1, I2_1;
    I1.convertTo(I1_1, d);
    I2.convertTo(I2_1, d);

    Mat I1_2 = I1_1.mul(I1_1);
    Mat I2_2 = I2_1.mul(I2_1);
    Mat I1_I2 = I1_1.mul(I2_1);

    Mat mu1, mu2;
    GaussianBlur(I1_1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2_1, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma1_sigma2;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma1_sigma2, Size(11, 11), 1.5);
    sigma1_sigma2 -= mu1_mu2;

    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma1_sigma2 + C2;
    t3 = t1.mul(t2);  // t3 = ((2*mu1_mu2 + C1).*(2*sigma1_sigma_2 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);  // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);       // ssim_map = t3./t1;
    Scalar mssim = mean(ssim_map);  // average of ssim_map
    return mssim;
}
// ![get-mssim]: Structural similarity: Image quality assessment from error visibility to structural similarity

int VideoIO(int argc, char* argv[])
{
    // TODO: FINISH THIS
    VideoIOHelpMessage();

    if (argc != 4) {
        OUTPUTERROR << "Not enough parameters" << endl;
        return -1;
    }

    const string source = argv[1];
    const bool askOutputType = argv[3][0] == 'Y';  // If false it will use the inputs codec type

    VideoCapture inputVideo(source);

    if (!inputVideo.isOpened()) {
        OUTPUTERROR << "Could not open the input video: " << source << endl;
        return -1;
    }

    string::size_type pAt = source.find_last_of('.');                 // Find extension point
    const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";  // Form the new name with container
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));       // Get Codec Type- Int form
    // Transform from int to char via Bitwise operators
    char EXT[] = { (char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8), (char)((ex & 0XFF0000) >> 16), (char)((ex & 0XFF000000) >> 24), 0 };

    Size S = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),  // Acquire input size
        (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo;  // Open the output

    //if (askOutputType)
    //outputVideo.open(NAME, ex = -1, inputVideo.get(CAP_PROP_FPS), S, true);
    //else

    // ! Not using askOutputType bool here
    // BUG: Normally, a windows named Video Compression should be opended for the user to select the CODEC
    // if we pass ex = -1 in this case, but on my machine, it's not opening up correctly.
    outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened()) {
        OUTPUTERROR << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }

    OUTPUTINFO << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
        << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
    OUTPUTINFO << "Input codec type: " << EXT << endl;
    int channel = 2;  // Select the channel to save
    switch (argv[2][0]) {
    case 'R': channel = 2; break;
    case 'G': channel = 1; break;
    case 'B': channel = 0; break;
    }
    Mat src, res;
    vector<Mat> spl;
    while (1) {  //Show the image captured in the window and repeat

        inputVideo >> src;       // read
        if (src.empty()) break;  // check if at end
        split(src, spl);         // process - extract only the correct channel
        for (int i = 0; i < 3; ++i)
            if (i != channel)
                spl[i] = Mat::zeros(S, spl[0].type());
        merge(spl, res);
        //outputVideo.write(res); //save or
        outputVideo << res;
    }
    OUTPUTINFO << "Finished writing" << endl;
    return 0;
}

void DrawCircles()
{
    Mat image = Mat::zeros(300, 600, CV_8UC3);
    //! OpenCV doesn't accept alpha channel color
    circle(image, Point(250, 150), 100, HexString2ColorScalar("#3d7ea6"), -100);
    circle(image, Point(350, 150), 100, HexString2ColorScalar("#fcdadaaa"), -100);
    //circle(image, Point(250, 150), 100, Hex2ColorScalar(ColorDarkBlue), -100);
    //circle(image, Point(350, 150), 100, Hex2ColorScalar(ColorLightPink), -100);
    imshow("Display Window", image);
    //waitKey(0);
}

void DrawAtomAndRook()
{
    char atom_window[] = "Drawing 1: Atom";
    char rook_window[] = "Drawing 2: Rook";

    Mat atom_image = Mat::zeros(AtomAndRookWidth, AtomAndRookWidth, CV_8UC3);
    Mat rook_image = Mat::zeros(AtomAndRookWidth, AtomAndRookWidth, CV_8UC3);

    MyEllipse(atom_image, 90);
    MyEllipse(atom_image, 0);
    MyEllipse(atom_image, 45);
    MyEllipse(atom_image, -45);
    MyFilledCircle(atom_image, Point(AtomAndRookWidth / 2, AtomAndRookWidth / 2));

    rectangle(rook_image,
        Point(0, 7 * AtomAndRookWidth / 8),
        Point(AtomAndRookWidth, AtomAndRookWidth),
        Hex2ColorScalar(ColorDarkPink),
        FILLED,
        LINE_AA);

    MyPolygon(rook_image);

    MyLine(rook_image, Point(0, 15 * AtomAndRookWidth / 16), Point(AtomAndRookWidth, 15 * AtomAndRookWidth / 16));
    MyLine(rook_image, Point(AtomAndRookWidth / 4, 7 * AtomAndRookWidth / 8), Point(AtomAndRookWidth / 4, AtomAndRookWidth));
    MyLine(rook_image, Point(AtomAndRookWidth / 2, 7 * AtomAndRookWidth / 8), Point(AtomAndRookWidth / 2, AtomAndRookWidth));
    MyLine(rook_image, Point(3 * AtomAndRookWidth / 4, 7 * AtomAndRookWidth / 8), Point(3 * AtomAndRookWidth / 4, AtomAndRookWidth));

    imshow(atom_window, atom_image);
    moveWindow(atom_window, 0, 200);
    imshow(rook_window, rook_image);
    moveWindow(rook_window, AtomAndRookWidth, 200);
    //waitKey(0);
}

void MyEllipse(Mat img, double angle, int width)
{
    int thickness = 2;
    int lineType = LINE_AA;

    ellipse(img, Point(width / 2, width / 2), Size(width / 4, width / 16), angle, 0, 360, Hex2ColorScalar(ColorDarkPink), thickness, lineType);
}

void MyFilledCircle(Mat img, Point center, int width)
{
    circle(img, center, width, Hex2ColorScalar(ColorLightBlue), FILLED, LINE_AA);
}

void MyLine(Mat img, Point start, Point end)
{
    int thickness = 2;
    int lineType = LINE_AA;

    line(img, start, end, Hex2ColorScalar(ColorDarkBlue), thickness, lineType);
}

void MyPolygon(Mat img)
{
    int lineType = LINE_AA;
    Point rook_points[1][20];
    rook_points[0][0] = Point(AtomAndRookWidth / 4, 7 * AtomAndRookWidth / 8);
    rook_points[0][1] = Point(3 * AtomAndRookWidth / 4, 7 * AtomAndRookWidth / 8);
    rook_points[0][2] = Point(3 * AtomAndRookWidth / 4, 13 * AtomAndRookWidth / 16);
    rook_points[0][3] = Point(11 * AtomAndRookWidth / 16, 13 * AtomAndRookWidth / 16);
    rook_points[0][4] = Point(19 * AtomAndRookWidth / 32, 3 * AtomAndRookWidth / 8);
    rook_points[0][5] = Point(3 * AtomAndRookWidth / 4, 3 * AtomAndRookWidth / 8);
    rook_points[0][6] = Point(3 * AtomAndRookWidth / 4, AtomAndRookWidth / 8);
    rook_points[0][7] = Point(26 * AtomAndRookWidth / 40, AtomAndRookWidth / 8);
    rook_points[0][8] = Point(26 * AtomAndRookWidth / 40, AtomAndRookWidth / 4);
    rook_points[0][9] = Point(22 * AtomAndRookWidth / 40, AtomAndRookWidth / 4);
    rook_points[0][10] = Point(22 * AtomAndRookWidth / 40, AtomAndRookWidth / 8);
    rook_points[0][11] = Point(18 * AtomAndRookWidth / 40, AtomAndRookWidth / 8);
    rook_points[0][12] = Point(18 * AtomAndRookWidth / 40, AtomAndRookWidth / 4);
    rook_points[0][13] = Point(14 * AtomAndRookWidth / 40, AtomAndRookWidth / 4);
    rook_points[0][14] = Point(14 * AtomAndRookWidth / 40, AtomAndRookWidth / 8);
    rook_points[0][15] = Point(AtomAndRookWidth / 4, AtomAndRookWidth / 8);
    rook_points[0][16] = Point(AtomAndRookWidth / 4, 3 * AtomAndRookWidth / 8);
    rook_points[0][17] = Point(13 * AtomAndRookWidth / 32, 3 * AtomAndRookWidth / 8);
    rook_points[0][18] = Point(5 * AtomAndRookWidth / 16, 13 * AtomAndRookWidth / 16);
    rook_points[0][19] = Point(AtomAndRookWidth / 4, 13 * AtomAndRookWidth / 16);
    const Point* ppt[1] = { rook_points[0] };
    int npt[] = { 20 };
    fillPoly(img,
        ppt,
        npt,
        1,
        Hex2ColorScalar(0xdddddd),
        lineType);
}

Scalar Hex2ColorScalar(unsigned long color, bool hasAlpha)
{
    cout << "[DEBUG] " << __func__ << ": Receiving color 0x" << setfill('0') << setw(hasAlpha ? 4 * 2 : 3 * 2) << right << hex << color << ". hasAlpha: " << boolalpha << hasAlpha << endl;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha = 0xff;
    if (hasAlpha) {
        alpha = color & 0xff;
        color >>= 8;
    }
    blue = color & 0xff;
    green = (color >> 8) & 0xff;
    red = (color >> 16) & 0xff;

    /** OpenCV uses BGR */
    return Scalar(blue, green, red, alpha / 0xff);
}

Scalar HexString2ColorScalar(string str)
{
    if (str[0] != '#' || (str.length() != 1 + 3 * 2 && str.length() != 1 + 4 * 2)) {
        cout << "[WARNING] " << __func__ << ": Unrecognized hex string format for string " << str << ". Example: #222831 or #22283130. Using color #3d7ea6 instead." << endl;
        return HexString2ColorScalar("#3d7ea6");
    }
    unsigned long color;
    stringstream ss;
    ss << hex << &str[1];
    ss >> color;  // color is now the hex value representation
    if (str.length() == 1 + 4 * 2) {
        return Hex2ColorScalar(color, true);
    }
    else {
        return Hex2ColorScalar(color, false);
    }
}