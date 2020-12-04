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

using namespace cv;
using namespace std;

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
Scalar Hex2ColorScalar(uint color, bool hasAlpha = false);
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

int imageShow(int argc, char* argv[]);

static Scalar randomColor(RNG& rng);
int drawRandomLines(Mat image, const string window_name, RNG rng);
int drawRandomRectangles(Mat image, const string window_name, RNG rng);
int drawRandomEllipses(Mat image, const string window_name, RNG rng);
int drawRandomPolylines(Mat image, const string window_name, RNG rng);
int drawRandomFilledPolygons(Mat image, const string window_name, RNG rng);
int drawRandomCircles(Mat image, const string window_name, RNG rng);
int displayRandomText(Mat image, const string window_name, RNG rng);
int displayBigEnd(Mat image, const string window_name, RNG rng);
int imageRandom(int argc, char* argv[]);

