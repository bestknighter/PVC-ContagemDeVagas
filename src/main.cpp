#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;


// Começa GLCM (matriz de homogeneidades)



// Termina GLCM

// Começa Hough (Transformada de Hough (rho theta), mesmo alpha)



// Termina Hough


int main(int argc, char** argv){
    if(argc < 2){
        printf("Falha ao receber argumentos.\n");
        exit(1);
    }
    // Read image
    Mat img = imread(argv[1]);
    img.convertTo(img, CV_32F, 1/255.0);
    
    // Calculate gradients gx, gy
    Mat gx, gy; 
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    Mat mag, angle; 
    cartToPolar(gx, gy, mag, angle, 1);

    imshow("img", img);
    imshow("gx", gx);
    imshow("gy", gy);
    imshow("mag", mag);
    imshow("angle", angle);

    // Começa GLCM (matriz de homogeneidades)



    // Termina GLCM

    // Começa Hough (Transformada de Hough (rho theta), mesmo alpha)



    // Termina Hough

    waitKey(0);

    return 0;
}

// Começa GLCM (matriz de homogeneidades)



// Termina GLCM

// Começa Hough (Transformada de Hough (rho theta), mesmo alpha)



// Termina Hough
