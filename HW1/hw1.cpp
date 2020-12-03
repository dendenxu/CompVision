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

int main()
{
    Mat image = Mat::zeros(300, 600, CV_8UC3);
    circle(image, Point(250, 150), 100, HexString2ColorScalar("#3d7ea6"), -100);
    circle(image, Point(350, 150), 100, HexString2ColorScalar("#fcdada"), -100);
    //circle(image, Point(250, 150), 100, Hex2ColorScalar(ColorDarkBlue), -100);
    //circle(image, Point(350, 150), 100, Hex2ColorScalar(ColorLightPink), -100);
    imshow("Display Window", image);
    waitKey(0);

    // char atom_window[] = "Drawing 1: Atom";
    // char rook_window[] = "Drawing 2: Rook";

    // Mat atom_image = Mat::zeros(w, w, CV_8UC3);
    // Mat rook_image = Mat::zeros(w, w, CV_8UC3);

    return 0;
}

void MyLine(Mat img, Point start, Point end)
{
    int thickness = 2;
    int lineType = LINE_AA;

    line(img, start, end, HexString2ColorScalar("#3d7ea6"), thickness, lineType);
}

Scalar Hex2ColorScalar(unsigned long color, bool hasAlpha)
{
    cout << "[DEBUG] " << __func__ << ": Receiving color 0x" << setfill('0') << setw(hasAlpha ? 4 * 2 : 3 * 2) << right << hex << color << endl;
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

    /** OpenCV uses BGRA */
    return Scalar(blue, green, red, alpha);
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