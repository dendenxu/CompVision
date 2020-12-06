#include "include.hpp"
#define CORRECTION(x) (static_cast<int>((static_cast<double>(x)) / 900 * WINDOW_WIDTH))
#define WRITEOUTPUT                                                             \
    imshow(window_name, image);                                                 \
    if (output.isOpened()) output << image;                                     \
    randomLastFrame = image;                                                    \
    if (waitKey(frameTime) >= 0) {                                              \
        OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl; \
        return -1;                                                              \
    }

static auto repeatCount = 100;
static auto frameTime = 5;  // ! in ms
static auto WINDOW_WIDTH = 1920;
static auto WINDOW_HEIGHT = 1080;
static auto xMin = -WINDOW_WIDTH / 2;
static auto xMax = 3 * WINDOW_WIDTH / 2;
static auto yMin = -WINDOW_HEIGHT / 2;
static auto yMax = 3 * WINDOW_HEIGHT / 2;
static auto axesMin = 0;
static auto axesMax = CORRECTION(200);
static auto angleMin = 0;
static auto angleMax = CORRECTION(180);
static auto lineWidthMax = CORRECTION(10);
static auto recLineWidthMax = CORRECTION(10);
static auto elliLineWidthMax = CORRECTION(9);
static auto fontScaleMin = 0 + 0.1;
static auto fontScaleMax = CORRECTION(5) + 0.1;
static auto fontThickness = CORRECTION(5);
static auto headFontThickness = CORRECTION(5);
static auto headFontScale = CORRECTION(3) + 0.0;
static VideoWriter output;
static auto randomLastFrame = Mat();

Mat getRandomLastFrame() { return randomLastFrame; }

void randomInit(Size size, int repeat, VideoWriter& writer)
{
    repeatCount = repeat;  // init using parameters
    frameTime = 5;         // ! in ms
    WINDOW_WIDTH = size.width;
    WINDOW_HEIGHT = size.height;
    xMin = -WINDOW_WIDTH / 2;
    xMax = 3 * WINDOW_WIDTH / 2;
    yMin = -WINDOW_HEIGHT / 2;
    yMax = 3 * WINDOW_HEIGHT / 2;
    axesMin = 0;
    axesMax = CORRECTION(200);
    angleMin = 0;
    angleMax = CORRECTION(180);
    lineWidthMax = CORRECTION(10);
    recLineWidthMax = CORRECTION(10);
    elliLineWidthMax = CORRECTION(9);
    fontScaleMin = 0 + 0.1;
    fontScaleMax = CORRECTION(5) + 0.1;
    fontThickness = CORRECTION(5);
    headFontThickness = CORRECTION(5);
    headFontScale = CORRECTION(3) + 0.0;

    output = writer;
}

int ImageRandom(int argc, char* argv[])
{
    const string window_name = "Drawing Random Stuff";

    RNG rng(0xffffffff);
    Mat image = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    imshow(window_name, image);
    //setWindowProperty(WIN_RANDOM, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    waitKey(frameTime);

    drawRandomLines(image, window_name, rng);
    drawRandomRectangles(image, window_name, rng);
    drawRandomEllipses(image, window_name, rng);
    drawRandomPolylines(image, window_name, rng);
    drawRandomFilledPolygons(image, window_name, rng);
    drawRandomCircles(image, window_name, rng);
    displayRandomText(image, window_name, rng);
    displayBigEnd(image, window_name, rng);

    waitKey(1);  // ! wait a ms
    return 0;
}

int IntroRandom(int argc, char* argv[])
{
    const string window_name = "Drawing Random Stuff";

    RNG rng(0xffffffff);
    Mat image = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    imshow(window_name, image);
    //setWindowProperty(WIN_RANDOM, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    waitKey(frameTime);

    drawRandomLines(image, window_name, rng);
    drawRandomRectangles(image, window_name, rng);
    //drawRandomEllipses(image, window_name, rng);
    //drawRandomPolylines(image, window_name, rng);
    //drawRandomFilledPolygons(image, window_name, rng);
    drawRandomCircles(image, window_name, rng);
    displayRandomText(image, window_name, rng);
    displayBigEnd(image, window_name, rng);

    waitKey(1);  // ! wait a ms
    return 0;
}

Scalar randomColor(RNG& rng)
{
    uint color = (uint)rng;
    OUTPUTINFO << "Getting random number: 0x" << hex << setfill('0') << setw(6) << color << endl;
    return Scalar(color & 0xff, (color >> 8) & 0xff, (color >> 16) & 0xff);
}

int drawRandomLines(Mat image, const string window_name, RNG rng)
{
    Point pt1, pt2;

    for (int i = 0; i < repeatCount; i++) {
        pt1.x = rng.uniform(xMin, xMax);
        pt1.y = rng.uniform(yMin, yMax);
        pt2.x = rng.uniform(xMin, xMax);
        pt2.y = rng.uniform(yMin, yMax);

        line(image, pt1, pt2, randomColor(rng), rng.uniform(1, lineWidthMax), LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}

int drawRandomRectangles(Mat image, const string window_name, RNG rng)
{
    Point pt1, pt2;

    for (int i = 0; i < repeatCount; i++) {
        pt1.x = rng.uniform(xMin, xMax);
        pt1.y = rng.uniform(yMin, yMax);
        pt2.x = rng.uniform(xMin, xMax);
        pt2.y = rng.uniform(yMin, yMax);

        rectangle(image, pt1, pt2, randomColor(rng), MAX(rng.uniform(-3, recLineWidthMax), -1), LINE_AA);
        WRITEOUTPUT
    }

    return 0;
}

int drawRandomEllipses(Mat image, const string window_name, RNG rng)
{
    Point center;
    Size axes;
    double angle;
    for (int i = 0; i < repeatCount; i++) {
        center.x = rng.uniform(xMin, xMax);
        center.y = rng.uniform(yMin, yMax);
        axes.width = rng.uniform(axesMin, axesMax);
        axes.height = rng.uniform(axesMin, axesMax);
        angle = rng.uniform(angleMin, angleMax);

        OUTPUTINFO << "Getting axes" << axes << "for index: " << i << endl;

        ellipse(image, center, axes, angle, angle - 100, angle + 200, randomColor(rng), rng.uniform(-1, elliLineWidthMax), LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}

int drawRandomPolylines(Mat image, const string window_name, RNG rng)
{
    Point pt[2][3];
    const Point* ppt[2] = {pt[0], pt[1]};
    int npt[] = {3, 3};
    for (int i = 0; i < repeatCount; i++) {
        pt[0][0].x = rng.uniform(xMin, xMax);
        pt[0][0].y = rng.uniform(yMin, yMax);
        pt[0][1].x = rng.uniform(xMin, xMax);
        pt[0][1].y = rng.uniform(yMin, yMax);
        pt[0][2].x = rng.uniform(xMin, xMax);
        pt[0][2].y = rng.uniform(yMin, yMax);
        pt[1][0].x = rng.uniform(xMin, xMax);
        pt[1][0].y = rng.uniform(yMin, yMax);
        pt[1][1].x = rng.uniform(xMin, xMax);
        pt[1][1].y = rng.uniform(yMin, yMax);
        pt[1][2].x = rng.uniform(xMin, xMax);
        pt[1][2].y = rng.uniform(yMin, yMax);
        polylines(image, ppt, npt, 2, true, randomColor(rng), rng.uniform(1, lineWidthMax), LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}
int drawRandomFilledPolygons(Mat image, const string window_name, RNG rng)
{
    Point pt[2][3];
    const Point* ppt[2] = {pt[0], pt[1]};
    int npt[] = {3, 3};
    for (int i = 0; i < repeatCount; i++) {
        pt[0][0].x = rng.uniform(xMin, xMax);
        pt[0][0].y = rng.uniform(yMin, yMax);
        pt[0][1].x = rng.uniform(xMin, xMax);
        pt[0][1].y = rng.uniform(yMin, yMax);
        pt[0][2].x = rng.uniform(xMin, xMax);
        pt[0][2].y = rng.uniform(yMin, yMax);
        pt[1][0].x = rng.uniform(xMin, xMax);
        pt[1][0].y = rng.uniform(yMin, yMax);
        pt[1][1].x = rng.uniform(xMin, xMax);
        pt[1][1].y = rng.uniform(yMin, yMax);
        pt[1][2].x = rng.uniform(xMin, xMax);
        pt[1][2].y = rng.uniform(yMin, yMax);
        fillPoly(image, ppt, npt, 2, randomColor(rng), LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}
int drawRandomCircles(Mat image, const string window_name, RNG rng)
{
    for (int i = 0; i < repeatCount; i++) {
        Point center;
        center.x = rng.uniform(xMin, xMax);
        center.y = rng.uniform(yMin, yMax);
        circle(image, center, rng.uniform(0, 300), randomColor(rng), rng.uniform(-1, elliLineWidthMax), LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}

// TODO: Implement this
int displayRandomText(Mat image, const string window_name, RNG rng)
{
    Point org;
    for (int i = 1; i < repeatCount; i++) {
        org.x = rng.uniform(xMin, xMax);
        org.y = rng.uniform(yMin, yMax);

        putText(image, __func__, org, rng.uniform(0, 8), rng.uniform(fontScaleMin, fontScaleMax), randomColor(rng), rng.uniform(1, fontThickness), LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}

// TODO: Implement this
int displayBigEnd(Mat image, const string window_name, RNG rng)
{
    Size textsize = getTextSize("Wakanda Forever", FONT_HERSHEY_SCRIPT_SIMPLEX, headFontScale, headFontThickness, 0);
    Point org((WINDOW_WIDTH - textsize.width) / 2, (WINDOW_HEIGHT - textsize.height) / 2);
    Scalar startColor = randomColor(rng) / 10;

    int step = 2;
    for (int i = 0; i < 255; i += step) {
        image -= Scalar::all(step * 2);  // ! The image should fade faster than the text
        for (auto& value : startColor.val) {
            value += step;
            value = saturate_cast<uchar>(value);
        }
        putText(image, "Wakanda Forever", org, FONT_HERSHEY_SCRIPT_SIMPLEX, headFontScale, startColor, headFontThickness, LINE_AA);
        WRITEOUTPUT
    }
    return 0;
}