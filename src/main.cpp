#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;


// Começa GLCM (matriz de homogeneidades)

#define WINDOW_SIZE 32
#define LIMIAR_CINZA 32
#define DISTANCE 2
#define OFFSET_0 Point(1, 0)
#define OFFSET_45 Point(1, -1)
#define OFFSET_90 Point(0, -1)
#define OFFSET_135 Point(-1, -1)
#define NUM_FEATURES 4 // Entropia, Energia, Homogeneidade e Correlacao
#define FEATURES_ORDER "entropia energia homogeneidade correlacao"

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
    int featuresWidth = img.cols/WINDOW_SIZE;
    int featuresHeight = img.rows/WINDOW_SIZE;
    Mat featuresMatGLCM0( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
    Mat featuresMatGLCM45( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
    Mat featuresMatGLCM90( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
    Mat featuresMatGLCM135( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
    {
        Mat gray;
        img.convertTo(gray, CV_8U, 255.);
        cvtColor(gray, gray, COLOR_BGR2GRAY);
        float GLCM0[WINDOW_SIZE][WINDOW_SIZE] = {0.};
        float GLCM45[WINDOW_SIZE][WINDOW_SIZE] = {0.};
        float GLCM90[WINDOW_SIZE][WINDOW_SIZE] = {0.};
        float GLCM135[WINDOW_SIZE][WINDOW_SIZE] = {0.};
        float featuresGLCM0[NUM_FEATURES] = {0.};
        float featuresGLCM45[NUM_FEATURES] = {0.};
        float featuresGLCM90[NUM_FEATURES] = {0.};
        float featuresGLCM135[NUM_FEATURES] = {0.};

        for(int grayY = 0; grayY < gray.rows; grayY += WINDOW_SIZE){ // Faz a janela andar pela imagem
            for(int grayX = 0; grayX < gray.cols; grayX += WINDOW_SIZE){
                if(((grayX + WINDOW_SIZE) >= gray.cols) || ((grayY + WINDOW_SIZE) >= gray.rows)){ // Trata erro de janela ficar fora da imagem
                    continue;
                }

                Point start;
                Point end;

                // Compute 0 degrees
                Point offset = OFFSET_0 * DISTANCE;
                start.x = offset.x < 0 ? -offset.x : 0;
                start.y = offset.y < 0 ? -offset.y : 0;
                end = Point(WINDOW_SIZE, WINDOW_SIZE) - offset;
                for(int j = start.y; j < end.y; j++){
                    for(int i = start.x; i < end.x; i++){
                        //printf("\n");

                        int posY = j + grayY; // Posição real na imagem + deslocamento da janela
                        int posX = i + grayX;

                        //printf("Posicao real: (%d, %d)\n", posX, posY);

                        int posY2 = posY + (DISTANCE * OFFSET_0.y); // Posição do pixel de comparação
                        int posX2 = posX + (DISTANCE * OFFSET_0.x);

                        //printf("Posicao real COMPARADO: (%d, %d)\n", posX2, posY2);

                        int value_i = gray.at<unsigned char>(posY, posX); // Valor do pixel sendo testado
                        int value_j = gray.at<unsigned char>(posY2, posX2); // Valor do pixel comparado

                        //printf("Valor do pixel TESTADO: %d - COMPARADO: %d\n", value_i, value_j);

                        int index_i = value_i * LIMIAR_CINZA / 256; // Ajuste do valor de pixel para o tamanho da janela (LIMIAR_CINZA tons de cinza)
                        int index_j = value_j * LIMIAR_CINZA / 256;

                        //printf("Valor do pixel ajustado para %d tons de cinza... TESTADO: %d - COMPARADO: %d\n", LIMIAR_CINZA, index_i, index_j);

                        GLCM0[index_i][index_j]++;
                        GLCM0[index_j][index_i]++;

                        //printf("\n");

                        //GLCM0[(int) (img.at<unsigned char>(Point(j, i)) * LIMIAR_CINZA / 256)][(int) (img.at<unsigned char>(Point(j + (DISTANCE * OFFSET_0.y), i + (DISTANCE * OFFSET_0.x))) * LIMIAR_CINZA / 256)]++;
                    }
                    //printf("\n");
                }
                // Normalizar e salvar características no vetor
                {
                    float soma = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            soma += GLCM0[i][j];
                        }
                    }
                    float entropy = 0, energy = 0, homogeneity = 0, mean = 0, varianceSqr = 0, correlation = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            GLCM0[i][j] /= soma;
                            entropy += (0 == GLCM0[i][j]) ? 0 : GLCM0[i][j]*std::log(GLCM0[i][j]);
                            energy += GLCM0[i][j]*GLCM0[i][j];
                            homogeneity += GLCM0[i][j]/(1+(i-j)*(i-j));
                            mean += i*GLCM0[i][j];
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            varianceSqr += GLCM0[i][j]*(i-mean)*(i-mean);
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            correlation += GLCM0[i][j]*(i-mean)*(j-mean)/varianceSqr;
                        }
                    }
                    featuresGLCM0[0] = -entropy;
                    featuresGLCM0[1] = energy;
                    featuresGLCM0[2] = homogeneity;
                    featuresGLCM0[3] = correlation;
                }
            
                // Compute 45 degrees
                offset = OFFSET_45 * DISTANCE;
                start.x = offset.x < 0 ? -offset.x : 0;
                start.y = offset.y < 0 ? -offset.y : 0;
                end = Point(WINDOW_SIZE, WINDOW_SIZE) - offset;
                for(int j = start.y; j < end.y; j++){
                    for(int i = start.x; i < end.x; i++){
                        int index_i = (int) (gray.at<unsigned char>(grayY + j, grayX + i) * LIMIAR_CINZA / 256);
                        int index_j = (int) (gray.at<unsigned char>(grayY + j + offset.y, grayX + i + offset.x) * LIMIAR_CINZA / 256);
                        GLCM45[index_i][index_j]++;
                        GLCM45[index_j][index_i]++;
                    }
                }
                // Normalizar e salvar características no vetor
                {
                    float soma = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            soma += GLCM45[i][j];
                        }
                    }
                    float entropy = 0, energy = 0, homogeneity = 0, mean = 0, varianceSqr = 0, correlation = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            GLCM45[i][j] /= soma;
                            entropy += (0 == GLCM45[i][j]) ? 0 : GLCM45[i][j]*std::log(GLCM45[i][j]);
                            energy += GLCM45[i][j]*GLCM45[i][j];
                            homogeneity += GLCM45[i][j]/(1+(i-j)*(i-j));
                            mean += i*GLCM45[i][j];
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            varianceSqr += GLCM45[i][j]*(i-mean)*(i-mean);
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            correlation += GLCM45[i][j]*(i-mean)*(j-mean)/varianceSqr;
                        }
                    }
                    featuresGLCM45[0] = -entropy;
                    featuresGLCM45[1] = energy;
                    featuresGLCM45[2] = homogeneity;
                    featuresGLCM45[3] = correlation;
                }

                // Compute 90 degrees
                offset = OFFSET_90 * DISTANCE;
                start.x = offset.x < 0 ? -offset.x : 0;
                start.y = offset.y < 0 ? -offset.y : 0;
                end = Point(WINDOW_SIZE, WINDOW_SIZE) - offset;
                for(int j = start.y; j < end.y; j++){
                    for(int i = start.x; i < end.x; i++){
                        int index_i = (int) (gray.at<unsigned char>(grayY + j, grayX + i) * LIMIAR_CINZA / 256);
                        int index_j = (int) (gray.at<unsigned char>(grayY + j + offset.y, grayX + i + offset.x) * LIMIAR_CINZA / 256);
                        GLCM90[index_i][index_j]++;
                        GLCM90[index_j][index_i]++;
                    }
                }
                // Normalizar e salvar características no vetor
                {
                    float soma = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            soma += GLCM90[i][j];
                        }
                    }
                    float entropy = 0, energy = 0, homogeneity = 0, mean = 0, varianceSqr = 0, correlation = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            GLCM90[i][j] /= soma;
                            entropy += (0 == GLCM90[i][j]) ? 0 : GLCM90[i][j]*std::log(GLCM90[i][j]);
                            energy += GLCM90[i][j]*GLCM90[i][j];
                            homogeneity += GLCM90[i][j]/(1+(i-j)*(i-j));
                            mean += i*GLCM90[i][j];
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            varianceSqr += GLCM90[i][j]*(i-mean)*(i-mean);
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            correlation += GLCM90[i][j]*(i-mean)*(j-mean)/varianceSqr;
                        }
                    }
                    featuresGLCM90[0] = -entropy;
                    featuresGLCM90[1] = energy;
                    featuresGLCM90[2] = homogeneity;
                    featuresGLCM90[3] = correlation;
                }

                // Compute 135 degrees
                offset = OFFSET_135 * DISTANCE;
                start.x = offset.x < 0 ? -offset.x : 0;
                start.y = offset.y < 0 ? -offset.y : 0;
                end = Point(WINDOW_SIZE, WINDOW_SIZE) - offset;
                for(int j = start.y; j < end.y; j++){
                    for(int i = start.x; i < end.x; i++){
                        int index_i = (int) (gray.at<unsigned char>(grayY + j, grayX + i) * LIMIAR_CINZA / 256);
                        int index_j = (int) (gray.at<unsigned char>(grayY + j + offset.y, grayX + i + offset.x) * LIMIAR_CINZA / 256);
                        GLCM135[index_i][index_j]++;
                        GLCM135[index_j][index_i]++;
                    }
                }
                // Normalizar e salvar características no vetor
                {
                    float soma = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            soma += GLCM135[i][j];
                        }
                    }
                    float entropy = 0, energy = 0, homogeneity = 0, mean = 0, varianceSqr = 0, correlation = 0;
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            GLCM135[i][j] /= soma;
                            entropy += (0 == GLCM135[i][j]) ? 0 : GLCM135[i][j]*std::log(GLCM135[i][j]);
                            energy += GLCM135[i][j]*GLCM45[i][j];
                            homogeneity += GLCM135[i][j]/(1+(i-j)*(i-j));
                            mean += i*GLCM135[i][j];
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            varianceSqr += GLCM135[i][j]*(i-mean)*(i-mean);
                        }
                    }
                    for(int i = 0; i < LIMIAR_CINZA; i++){
                        for(int j = 0; j < LIMIAR_CINZA; j++){
                            correlation += GLCM135[i][j]*(i-mean)*(j-mean)/varianceSqr;
                        }
                    }
                    featuresGLCM135[0] = -entropy;
                    featuresGLCM135[1] = energy;
                    featuresGLCM135[2] = homogeneity;
                    featuresGLCM135[3] = correlation;
                }

                // SALVAR CARACTERISTICAS
                {
                    for(int k = 0; k < NUM_FEATURES; k++) {
                        featuresMatGLCM0.at<float>(grayY/WINDOW_SIZE, grayX/WINDOW_SIZE, k) = featuresGLCM0[k];
                        featuresMatGLCM45.at<float>(grayY/WINDOW_SIZE, grayX/WINDOW_SIZE, k) = featuresGLCM45[k];
                        featuresMatGLCM90.at<float>(grayY/WINDOW_SIZE, grayX/WINDOW_SIZE, k) = featuresGLCM90[k];
                        featuresMatGLCM135.at<float>(grayY/WINDOW_SIZE, grayX/WINDOW_SIZE, k) = featuresGLCM135[k];
                    }
                }
            }
        }
        FileStorage ymlFeatures("Features.yml", FileStorage::WRITE);
        ymlFeatures << "FeatureOrder" << FEATURES_ORDER;
        ymlFeatures << "GLCM-0" << featuresMatGLCM0;
        ymlFeatures << "GLCM-45" << featuresMatGLCM45;
        ymlFeatures << "GLCM-90" << featuresMatGLCM90;
        ymlFeatures << "GLCM-135" << featuresMatGLCM135;
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
