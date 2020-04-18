#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

struct Config
{
    /// @brief Path to dataset from this opencv-extra PR
    std::string imgDir = "/home/dmitry/projects/opencv_extra/testdata/calib3d/fisheye/51-50-00-43-f8-00/";

    std::string imgOut = "result.png";
    /// @note Chessboard pattern parameters

    /// @brief Chessboard pattern size
    cv::Size patternSize = cv::Size(6, 9);
    /// @brief Chessboard cell size
    double cellSizeMm = 75.0;

    bool debug = false;
};

void parseConfig(int argc, char** argv, Config &c)
{
    for (int i = 0 ; i < argc; i++)
    {
        ///< @note Path to dataset with fisheye PNG images
        if (!strcmp(argv[i], "--img-dir"))
            c.imgDir = argv[++i];
        else if (!strcmp(argv[i], "--debug"))
            c.debug = true;
    }
}

std::vector<cv::Point3f> generatePatternObjectFeatures(uint width,
                                                       uint height,
                                                       double cellSize)
{
    std::vector<cv::Point3f> patternObjectFeatures;
    for(uint i = 0; i < height; i++)
        for(uint j = 0; j < width; j++)
            patternObjectFeatures.push_back(cv::Point3f(cellSize * j, float(cellSize * i), 0));

    return patternObjectFeatures;
}

std::vector<std::vector<cv::Point3f>> generateObjectsFeatures(size_t size,
                                                              uint width,
                                                              uint height,
                                                              double cellSize)
{
    auto patternObjectFeatures =
            generatePatternObjectFeatures(width, height, cellSize);
    std::vector<std::vector<cv::Point3f>> objectsFeatures(size);

    std::fill(objectsFeatures.begin(),
              objectsFeatures.end(),
              patternObjectFeatures);

    return objectsFeatures;
}

int main(int argc, char** argv)
{
    Config c;

    parseConfig(argc, argv, c);

    std::vector<cv::String> imgPaths;

    cv::glob(c.imgDir + "/*.png", imgPaths);

#if 0 // Intrinsics which you can achive with PR version of OpenCV
    cv::Matx33f K(338.99126016399202, 0, 636.11384328104941,
                  0, 337.84486287472214, 542.52596483696834,
                  0, 0, 1);

    cv::Vec4f D(-0.01017814115694435, 0.066693053611465325,
                -0.018450144605964301, 0.00081304613011604499);
#endif

    // camera intrinsics
    cv::Matx33f K;
    cv::Vec4f D;

    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Point2f>> allFeatures;
    for (auto &imgPath : imgPaths)
    {
        // read image
        cv::Mat imgSrc = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

        // filter image
        cv::Mat img;
        cv::GaussianBlur(imgSrc, img, cv::Size(3, 3), 0);

        // find chessboards
        std::vector<cv::Point2f> features;
        cv::findChessboardCorners(img, c.patternSize, features,
                                  cv::CALIB_CB_ADAPTIVE_THRESH |
                                  cv::CALIB_CB_NORMALIZE_IMAGE);

        cv::cornerSubPix(imgSrc, features, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS |
                         cv::TermCriteria::MAX_ITER, 100, 0.01));

        // if not enough cesboard features found
        if(features.empty() || features.size() != c.patternSize.area())
        {
            continue;
        }

        // collect all features for camera calibration
        images.push_back(imgSrc);
        allFeatures.push_back(features);

        std::cout << "Capturing images..." << std::endl;
    }

    // calibrate camera

    // generate chesbsoard 3D pattern model
    auto objectsFeatures = generateObjectsFeatures(allFeatures.size(),
                                                   c.patternSize.width,
                                                   c.patternSize.height,
                                                   c.cellSizeMm);

    // calibrate camera intrinsics
    std::vector<cv::Mat> rvecs, tvecs;
    double err = cv::fisheye::calibrate(objectsFeatures, allFeatures,
                           images.front().size(),
                           K, D, rvecs, tvecs,
                           cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW,
                           cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 2e-2));

    std::cout << "Reprojection error: " << err << std::endl;

    // set different focal length to undistorted images to see more:
    // FOR DEMONSTRATION PURPOSE ONLY
    cv::Matx33f Kout = cv::Matx33f(K);
    Kout(0,0) /= 4;
    Kout(1,1) /= 4;

    // demonstartion
    for(size_t i = 0; i < images.size(); ++i)
    {
        auto const &img = images[i];
        auto const& features = allFeatures[i];
        auto const& objectFeatures = objectsFeatures[i];
        auto const& rvec = rvecs[i];
        auto const& tvec = tvecs[i];

        // unditort image and undistort features which are detected with cv::findChessboardCorners()
        cv::Mat imgUn;
        std::vector<cv::Point2f> featuresUn;
        cv::fisheye::undistortImage(img, imgUn, K, D, Kout);
        cv::fisheye::undistortPoints(features, featuresUn, K, D, cv::Matx33f::eye(), Kout);

        // draw undistorted features
        for(auto const &f : featuresUn)
        {
            cv::circle(imgUn, f, 3, cv::Scalar::all(255));
        }

        // project 3D real object points on fisheye image using inforamtion
        // about its relative pose from intrinsics calibration with cv::fisheye::projectPoints()
        std::vector<cv::Point2f> featuresRep;
        cv::fisheye::projectPoints(objectFeatures, featuresRep, rvec, tvec, K, D);

        // draw projected images
        for(auto const &f : featuresRep)
        {
            cv::circle(img, f, 3, cv::Scalar::all(255));
        }


        // display
        cv::Mat imgDbg;
        cv::hconcat(img, imgUn, imgDbg);

        cv::namedWindow("result", cv::WINDOW_NORMAL);
        cv::imshow("result", imgDbg);

        // display more debug
        if(c.debug)
        {
            cv::imshow("cv::projectPoints() result", img);
            cv::imshow("cv::undistortPoints() result", imgUn);
        }

        char key = cv::waitKey(0);
        if(key == 'q')
        {
            cv::imwrite(c.imgOut, img);

            break;
        }
    }

    return 0;
}
