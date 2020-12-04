#include "include.hpp"

constexpr auto NUMBER = 100;
constexpr auto DELAY = 5; // ! in ms
constexpr auto WINDOW_WIDTH = 900;
constexpr auto WINDOW_HEIGHT = 600;
constexpr auto WIN_W_MIN = -WINDOW_WIDTH / 2;
constexpr auto WIN_W_MAX = 3 * WINDOW_WIDTH / 2;
constexpr auto WIN_H_MIN = -WINDOW_HEIGHT / 2;
constexpr auto WIN_H_MAX = 3 * WINDOW_HEIGHT / 2;

constexpr auto ELLI_AXES_MIN = 0;
constexpr auto ELLI_AXES_MAX = 200;
constexpr auto ELLI_ANGLE_MIN = 0;
constexpr auto ELLI_ANGLE_MAX = 180;

int imageRandom(int argc, char* argv[])
{
    const string window_name = "Drawing Random Stuff";

    RNG rng(0xffffffff);
    Mat image = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    imshow(window_name, image);
    //setWindowProperty(WIN_RANDOM, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    waitKey(DELAY);

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

    for (int i = 0; i < NUMBER; i++) {
        pt1.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt1.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt2.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt2.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);

        line(image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
    }
    return 0;
}

int drawRandomRectangles(Mat image, const string window_name, RNG rng)
{
    Point pt1, pt2;

    for (int i = 0; i < NUMBER; i++) {
        pt1.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt1.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt2.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt2.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);

        rectangle(image, pt1, pt2, randomColor(rng), MAX(rng.uniform(-3, 10), -1), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
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
    for (int i = 0; i < NUMBER; i++) {
        center.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        center.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        axes.width = rng.uniform(ELLI_AXES_MIN, ELLI_AXES_MAX);
        axes.height = rng.uniform(ELLI_AXES_MIN, ELLI_AXES_MAX);
        angle = rng.uniform(ELLI_ANGLE_MIN, ELLI_ANGLE_MAX);

        ellipse(image, center, axes, angle, angle - 100, angle + 200, randomColor(rng), rng.uniform(-1, 9), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
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
    for (int i = 0; i < NUMBER; i++) {
        pt[0][0].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[0][0].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[0][1].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[0][1].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[0][2].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[0][2].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[1][0].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[1][0].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[1][1].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[1][1].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[1][2].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[1][2].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        polylines(image, ppt, npt, 2, true, randomColor(rng), rng.uniform(1, 10), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
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
    for (int i = 0; i < NUMBER; i++) {
        pt[0][0].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[0][0].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[0][1].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[0][1].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[0][2].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[0][2].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[1][0].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[1][0].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[1][1].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[1][1].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        pt[1][2].x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        pt[1][2].y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        fillPoly(image, ppt, npt, 2, randomColor(rng), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
    }
    return 0;
}
int drawRandomCircles(Mat image, const string window_name, RNG rng)
{
    for (int i = 0; i < NUMBER; i++) {
        Point center;
        center.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        center.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);
        circle(image, center, rng.uniform(0, 300), randomColor(rng), rng.uniform(-1, 9), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
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
    for (int i = 1; i < NUMBER; i++) {
        org.x = rng.uniform(WIN_W_MIN, WIN_W_MAX);
        org.y = rng.uniform(WIN_H_MIN, WIN_H_MAX);

        putText(image, __func__, org, rng.uniform(0, 8), rng.uniform(0.1, 5.1), randomColor(rng), rng.uniform(1, 10), LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
            OUTPUTERROR << "Something is wrong when drawing: " << dec << i << endl;
            return -1;
        }
    }
    return 0;
}
// TODO: Implement this
int displayBigEnd(Mat image, const string window_name, RNG rng)
{
    Size textsize = getTextSize("Wakanda Forever", FONT_HERSHEY_SCRIPT_SIMPLEX, 3, 5, 0);
    Point org((WINDOW_WIDTH - textsize.width) / 2, (WINDOW_HEIGHT - textsize.height) / 2);
    Scalar startColor = randomColor(rng) / 10;

    int step = 2;
    for (int i = 0; i < 255; i += step) {
        image -= Scalar::all(step*2); // ! The image should fade faster than the text
        for (auto& value : startColor.val) {
            value += step;
            value = saturate_cast<uchar>(value);
        }
        putText(image, "Wakanda Forever", org, FONT_HERSHEY_SCRIPT_SIMPLEX, 3, startColor, 5, LINE_AA);
        imshow(window_name, image);
        if (waitKey(DELAY) >= 0) {
            return -1;
        }
    }
    return 0;
}