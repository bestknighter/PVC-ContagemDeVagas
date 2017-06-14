#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;


// Começa GLCM (matriz de homogeneidades)

#define WINDOW_SIZE 32
#define LIMIAR_CINZA 32
#define DISTANCE 2
#define OFFSET_0 Point(0, 1)
#define OFFSET_45 Point(-1, 1)
#define OFFSET_90 Point(-1, 0)
#define OFFSET_135 Point(-1, -1)




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
    float GLCM0[WINDOW_SIZE][WINDOW_SIZE];
    float GLCM45[WINDOW_SIZE][WINDOW_SIZE];
    float GLCM90[WINDOW_SIZE][WINDOW_SIZE];
    float GLCM135[WINDOW_SIZE][WINDOW_SIZE];

    {
        Mat gray;
        cvtColor(mag, gray, COLOR_BGR2GRAY);
        Mat window[WINDOW_SIZE][WINDOW_SIZE];

        for(int grayX = 0; grayX < gray.rows; grayX += WINDOW_SIZE){ // Faz a janela andar pela imagem
            for(int grayY = 0; grayY < gray.cols; grayY += WINDOW_SIZE){
                if((gray.rows > (grayX + WINDOW_SIZE)) && (gray.cols > (grayY + WINDOW_SIZE))){ // Trata erro de janela ficar fora da imagem
                    // Compute 0 degrees
                    for(int i = 0; i < window.rows - (OFFSET_0.x * DISTANCE); i++){
                        for(int j = 0; j < window.cols - (OFFSET_0.y * DISTANCE); j++){
                            //printf("\n");

                            int posY = j + grayY; // Posição real na imagem + deslocamento da janela
                            int posX = i + grayX;

                            //printf("Posicao real: (%d, %d)\n", posX, posY);

                            int posY2 = posY + (DISTANCE * OFFSET_0.y); // Posição do pixel de comparação
                            int posX2 = posX + (DISTANCE * OFFSET_0.x);

                            //printf("Posicao real COMPARADO: (%d, %d)\n", posX2, posY2);

                            int value_i = gray.at<unsigned char>(Point(posY, posX)); // Valor do pixel sendo testado
                            int value_j = gray.at<unsigned char>(Point(posY2, posX2)); // Valor do pixel comparado

                            //printf("Valor do pixel TESTADO: %d - COMPARADO: %d\n", value_i, value_j);

                            int index_i = value_i *  LIMIAR_CINZA/ 256; // Ajuste do valor de pixel para o tamanho da janela (LIMIAR_CINZA tons de cinza)
                            int index_j = value_j * LIMIAR_CINZA / 256;

                            //printf("Valor do pixel ajustado para %d tons de cinza... TESTADO: %d - COMPARADO: %d\n", LIMIAR_CINZA, index_i, index_j);

                            GLCM0[index_i][index_j]++;
                            GLCM0[index_j][index_i]++;

                            //printf("\n");

                            //GLCM0[(int) (img.at<unsigned char>(Point(j, i)) * LIMIAR_CINZA / 256)][(int) (img.at<unsigned char>(Point(j + (DISTANCE * OFFSET_0.y), i + (DISTANCE * OFFSET_0.x))) * LIMIAR_CINZA / 256)]++;
                        }
                        //printf("\n");
                    }
                    // Normalizar
                    {
                        int soma = 0;
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                soma += GLCM0[i][j];
                            }
                        }
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                GLCM0[i][j] /= soma;
                            }
                        }
                    }
                }

                if((gray.rows > (grayX + WINDOW_SIZE)) && (gray.cols > (grayY + WINDOW_SIZE))){
                    // Compute 45 degrees
                    for(int i = -(OFFSET_45.x * DISTANCE); i < window.rows; i++){
                        for(int j = 0; j < window.cols - (OFFSET_45.y * DISTANCE); j++){
                            int index_i = (int) (gray.at<unsigned char>(Point(grayY + j, grayX + i)) * LIMIAR_CINZA / 256);
                            int index_j = (int) (gray.at<unsigned char>(Point(grayY + j + (DISTANCE * OFFSET_45.y), grayX + i + (DISTANCE * OFFSET_45.x))) * LIMIAR_CINZA / 256);
                            GLCM45[index_i][index_j]++;
                            GLCM45[index_j][index_i]++;
                        }
                    }
                    // Normalizar
                    {
                        int soma = 0;
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                soma += GLCM45[i][j];
                            }
                        }
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                GLCM45[i][j] /= soma;
                            }
                        }
                    }
                }

                if((gray.rows > (grayX + WINDOW_SIZE)) && (gray.cols > (grayY + WINDOW_SIZE))){
                    // Compute 90 degrees
                    for(int i = -(OFFSET_90.x * DISTANCE); i < window.rows; i++){
                        for(int j = 0 - (OFFSET_90.y * DISTANCE); j < window.cols; j++){
                            int index_i = (int) (gray.at<unsigned char>(Point(grayY + j, grayX + i)) * LIMIAR_CINZA / 256);
                            int index_j = (int) (gray.at<unsigned char>(Point(grayY + j + (DISTANCE * OFFSET_90.y), grayX + i + (DISTANCE * OFFSET_90.x))) * LIMIAR_CINZA / 256);
                            GLCM45[index_i][index_j]++;
                            GLCM45[index_j][index_i]++;
                        }
                    }
                    // Normalizar
                    {
                        int soma = 0;
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                soma += GLCM90[i][j];
                            }
                        }
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                GLCM90[i][j] /= soma;
                            }
                        }
                    }
                }

                if((gray.rows > (grayX + WINDOW_SIZE)) && (gray.cols > (grayY + WINDOW_SIZE))){
                    for(int i = -(OFFSET_135.x * DISTANCE); i < window.rows; i++){
                        for(int j = -(OFFSET_135.y * DISTANCE); j < window.cols; j++){
                            int index_i = (int) (gray.at<unsigned char>(Point(grayY + j, grayX + i)) * LIMIAR_CINZA / 256);
                            int index_j = (int) (gray.at<unsigned char>(Point(grayY + j + (DISTANCE * OFFSET_135.y), grayX + i + (DISTANCE * OFFSET_135.x))) * LIMIAR_CINZA / 256);
                            GLCM45[index_i][index_j]++;
                            GLCM45[index_j][index_i]++;
                        }
                    }
                    // Normalizar
                    {
                        int soma = 0;
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                soma += GLCM135[i][j];
                            }
                        }
                        for(int i = 0; i < WINDOW_SIZE; i++){
                            for(int j = 0; j < WINDOW_SIZE; j++){
                                GLCM135[i][j] /= soma;
                            }
                        }
                    }
                }

                // SALVAR CARACTERISTICAS
            }
        }
    }

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
