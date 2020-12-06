#include "include.hpp"

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
    const Point* ppt[1] = {rook_points[0]};
    int npt[] = {20};
    fillPoly(img,
             ppt,
             npt,
             1,
             Hex2ColorScalar(0xdddddd),
             lineType);
}

Scalar Hex2ColorScalar(uint color, bool hasAlpha)
{
    cout << "[DEBUG] " << __func__ << ": Receiving color 0x" << setfill('0') << setw(hasAlpha ? 4 * 2 : 3 * 2) << right << hex << color << ". hasAlpha: " << boolalpha << hasAlpha << endl;
    uchar red;
    uchar green;
    uchar blue;
    uchar alpha = 0xff;
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
    uint color;
    stringstream ss;
    ss << hex << &str[1];
    ss >> color;  // color is now the hex value representation
    if (str.length() == 1 + 4 * 2) {
        return Hex2ColorScalar(color, true);
    } else {
        return Hex2ColorScalar(color, false);
    }
}