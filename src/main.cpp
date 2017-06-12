#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

int main(int argc, char** argv){
    if(argc < 2){
        printf("Falha ao receber argumentos.\n");
        exit(1);
    }
    // C++ gradient calculation. 
    // Read image
    Mat img = imread(argv[1]);
    img.convertTo(img, CV_32F, 1/255.0);
    
    // Calculate gradients gx, gy
    Mat gx, gy; 
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    // C++ Calculate gradient magnitude and direction (in degrees)
    Mat mag, angle; 
    cartToPolar(gx, gy, mag, angle, 1);

    imshow("gx", gx);
    imshow("gy", gy);
    imshow("mag", mag);
    imshow("angle", angle);
    waitKey(0);

    return 0;
}