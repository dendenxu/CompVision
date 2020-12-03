#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#define w 400
constexpr auto ColorDarkBlue = 0x3d7ea6;
constexpr auto ColorLightBlue = 0x5c969e;
constexpr auto ColorDarkPink = 0xffa5a5;
constexpr auto ColorLightPink = 0xfcdada;

using namespace cv;
using namespace std;
Scalar HexString2ColorScalar(string str);
Scalar Hex2ColorScalar(unsigned long color, bool hasAlpha = false);
void MyEllipse(Mat img, double angle, int width = w);
void MyFilledCircle(Mat img, Point center, int width = w / 32);
void MyLine(Mat img, Point start, Point end);
void MyPolygon(Mat img);

int main()
{
    // Mat image = Mat::zeros(300, 600, CV_8UC3);
    // ! OpenCV doesn't accept alpha channel color
    // circle(image, Point(250, 150), 100, HexString2ColorScalar("#3d7ea6"), -100);
    // circle(image, Point(350, 150), 100, HexString2ColorScalar("#fcdadaaa"), -100);
    //circle(image, Point(250, 150), 100, Hex2ColorScalar(ColorDarkBlue), -100);
    //circle(image, Point(350, 150), 100, Hex2ColorScalar(ColorLightPink), -100);
    // imshow("Display Window", image);
    // waitKey(0);

    char atom_window[] = "Drawing 1: Atom";
    char rook_window[] = "Drawing 2: Rook";

    Mat atom_image = Mat::zeros(w, w, CV_8UC3);
    Mat rook_image = Mat::zeros(w, w, CV_8UC3);

    MyEllipse(atom_image, 90);
    MyEllipse(atom_image, 0);
    MyEllipse(atom_image, 45);
    MyEllipse(atom_image, -45);
    MyFilledCircle(atom_image, Point(w / 2, w / 2));

    rectangle(rook_image,
              Point(0, 7 * w / 8),
              Point(w, w),
              Hex2ColorScalar(ColorDarkPink),
              FILLED,
              LINE_AA);

    MyPolygon(rook_image);

    MyLine(rook_image, Point(0, 15 * w / 16), Point(w, 15 * w / 16));
    MyLine(rook_image, Point(w / 4, 7 * w / 8), Point(w / 4, w));
    MyLine(rook_image, Point(w / 2, 7 * w / 8), Point(w / 2, w));
    MyLine(rook_image, Point(3 * w / 4, 7 * w / 8), Point(3 * w / 4, w));

    imshow(atom_window, atom_image);
    moveWindow(atom_window, 0, 200);
    imshow(rook_window, rook_image);
    moveWindow(rook_window, w, 200);
    waitKey(0);
    return 0;
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
    rook_points[0][0] = Point(w / 4, 7 * w / 8);
    rook_points[0][1] = Point(3 * w / 4, 7 * w / 8);
    rook_points[0][2] = Point(3 * w / 4, 13 * w / 16);
    rook_points[0][3] = Point(11 * w / 16, 13 * w / 16);
    rook_points[0][4] = Point(19 * w / 32, 3 * w / 8);
    rook_points[0][5] = Point(3 * w / 4, 3 * w / 8);
    rook_points[0][6] = Point(3 * w / 4, w / 8);
    rook_points[0][7] = Point(26 * w / 40, w / 8);
    rook_points[0][8] = Point(26 * w / 40, w / 4);
    rook_points[0][9] = Point(22 * w / 40, w / 4);
    rook_points[0][10] = Point(22 * w / 40, w / 8);
    rook_points[0][11] = Point(18 * w / 40, w / 8);
    rook_points[0][12] = Point(18 * w / 40, w / 4);
    rook_points[0][13] = Point(14 * w / 40, w / 4);
    rook_points[0][14] = Point(14 * w / 40, w / 8);
    rook_points[0][15] = Point(w / 4, w / 8);
    rook_points[0][16] = Point(w / 4, 3 * w / 8);
    rook_points[0][17] = Point(13 * w / 32, 3 * w / 8);
    rook_points[0][18] = Point(5 * w / 16, 13 * w / 16);
    rook_points[0][19] = Point(w / 4, 13 * w / 16);
    const Point* ppt[1] = {rook_points[0]};
    int npt[] = {20};
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
    } else {
        return Hex2ColorScalar(color, false);
    }
}