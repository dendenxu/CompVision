#include "include.hpp"

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
        imshow(WINDOW_UNDERTEST, frameUnderTest);

        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapsed_seconds = end - start;
        //! not using delay here
        auto timeToWait = frameTime - elapsed_seconds.count() * 1000;

        OUTPUTINFO
            << "TimeToWait: " << timeToWait << "ms";

        cout << endl;
        if (timeToWait > 0) {
            char c = static_cast<char>(waitKey((int)timeToWait));
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
    } else {
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
    char EXT[] = {(char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8), (char)((ex & 0XFF0000) >> 16), (char)((ex & 0XFF000000) >> 24), 0};

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

int PlaygroundVideo(int argc, char* argv[])
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

    Mat last(getLastFrame(inputVideo)), resizedLast;

    if (last.empty()) {
        OUTPUTERROR << "Last frame of the video is empty" << endl;
    } else {
        resizeImage(last, resizedLast, newSize);
        crossDissolve(resizedLast, resized, outputVideo, (int)fps);
    }

    crossDissolve(resized, resizedSobel, outputVideo, (int)fps);

    waitKey();

    return 0;
}