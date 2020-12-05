#include "include.hpp"
#define CORRECTION(x) (static_cast<int>((static_cast<double>(x)) / 900 * WINDOW_WIDTH))

constexpr auto repeatCount = 100;
constexpr auto frameTime = 5;  // ! in ms
constexpr auto WINDOW_WIDTH = 1920;
constexpr auto WINDOW_HEIGHT = 1080;
constexpr auto xMin = -WINDOW_WIDTH / 2;
constexpr auto xMax = 3 * WINDOW_WIDTH / 2;
constexpr auto yMin = -WINDOW_HEIGHT / 2;
constexpr auto yMax = 3 * WINDOW_HEIGHT / 2;

constexpr auto axesMin = 0;
constexpr auto axesMax = CORRECTION(200);
constexpr auto angleMin = 0;
constexpr auto angleMax = CORRECTION(180);

constexpr auto lineWidthMax = CORRECTION(10);
constexpr auto recLineWidthMax = CORRECTION(10);
constexpr auto elliLineWidthMax = CORRECTION(9);

constexpr auto fontScaleMin = 0 + 0.1;
constexpr auto fontScaleMax = CORRECTION(5) + 0.1;
constexpr auto fontThickness = CORRECTION(5);
constexpr auto headFontThickness = CORRECTION(5);
constexpr auto headFontScale = CORRECTION(3) + 0.0;

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

    waitKey();
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
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
        imshow(window_name, image);
        if (waitKey(frameTime) >= 0) {
            return -1;
        }
    }
    return 0;
}