#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;


// Começa GLCM (matriz de homogeneidades)



// Termina GLCM

// Começa Hough (Transformada de Hough (rho theta), mesmo alpha)
#ifndef M_PI
    #define M_PI 3.141592653589793238L
#endif
#define BIN_THRES 0.5 // Threshold da binarizacao
#define HOUGH_THRES 300 // Acumulador
#define TAM_DILATA 6 // Pixels
#define TAM_ERODE 5 // Pixels
#define TAM_LINHA 1000 // Pixels
// Termina Hough


int main(int argc, char** argv){
    if(argc < 2){
        printf("Falha ao receber argumentos.\n");
        exit(1);
    }
    // Read image
    Mat img = imread(argv[1], IMREAD_COLOR);

    if(!img.data) {
        printf("Falha ao abrir a imagem.\n");
        exit(1);
    }

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
    {
        // Binariza
        Mat binarizado, gray, operado;
        cvtColor(mag, gray, COLOR_BGR2GRAY);
        threshold(gray, binarizado, BIN_THRES, 255, THRESH_BINARY);
        binarizado.convertTo(binarizado, CV_8UC1);
        imshow("Hough - binarizado", binarizado);

        // Faz fechamento customizado
        morphologyEx(binarizado, operado, MORPH_DILATE, getStructuringElement( MORPH_ELLIPSE, Size(TAM_DILATA, TAM_DILATA) ));
        morphologyEx(operado, operado, MORPH_ERODE, getStructuringElement( MORPH_ELLIPSE, Size(TAM_ERODE, TAM_ERODE) ));
        imshow("Hough - operado", operado);

        { // Acha as linhas por Hough
            std::vector<Vec2f> lines;
            HoughLines(operado, lines, 1, M_PI/180., HOUGH_THRES);
            
            Mat houghWithLines = gray.clone();
            cvtColor(operado, houghWithLines, COLOR_GRAY2BGR);
            for( unsigned int i = 0; i < lines.size(); i++ ) {
                float rho = lines[i][0];
                float theta = lines[i][1];

                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;

                Point pt1(cvRound( x0 + TAM_LINHA*(-b) ), cvRound( y0 + TAM_LINHA*a ));
                Point pt2(cvRound( x0 - TAM_LINHA*(-b) ), cvRound( y0 - TAM_LINHA*a ));
                line( houghWithLines, pt1, pt2, Scalar(0,0,255), 1, 8 );
            }

            imshow("Hough - Linhas", houghWithLines);
        }

        // { // Acha os segmentos de linhas por Hough
        // }

    }
    // Termina Hough

    waitKey(0);

    return 0;
}

// Começa GLCM (matriz de homogeneidades)



// Termina GLCM

// Começa Hough (Transformada de Hough (rho theta), mesmo alpha)



// Termina Hough
