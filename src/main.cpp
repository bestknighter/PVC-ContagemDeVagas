#include <opencv2/opencv.hpp>

using namespace cv;

// Começa GLCM (matriz de homogeneidades)

#define WINDOW_SIZE 32
#define LIMIAR_CINZA 32
#define X_STEP 8
#define Y_STEP 8
#define DISTANCE 3
#define OFFSET_0 Point(1, 0)
#define OFFSET_45 Point(1, -1)
#define OFFSET_90 Point(0, -1)
#define OFFSET_135 Point(-1, -1)
#define NUM_FEATURES 1 // Homogeneidade
#define FEATURES_ORDER "homogeneidade"

Mat ExtractFeatureMat(Mat featuresMat, int featureNum);

// Termina GLCM

// Começa Hough (Transformada de Hough (rho theta), mesmo alpha)

#ifndef M_PI
	#define M_PI 3.141592653589793238L
#endif
#define BIN_THRES 0.5 // Threshold da binarizacao
#define DIST_RES 1
#define ANGLE_RES M_PI/180.
#define HOUGH_THRES 300 // Acumulador
#define HOUGHP_THRES 75 // Acumulador
#define MIN_LINE_LEN 5
#define MAX_LINE_GAP 25
#define TAM_DILATA 7 // Pixels
#define TAM_ERODE 5 // Pixels
#define TAM_LINE 3000 // Pixels

// Termina Hough

// Começa AKM

#define SENSIBILITY 5.
#define LINE_THRES 0.5
#define SEGLINE_THRES 0.65
#define LINE_MAX_DIST 75

std::vector<Vec2f> AKM( std::vector<Vec2f> input, float threshold, float lineMaxDist = LINE_MAX_DIST, unsigned int minLines = 1 );
std::vector<Vec4i> AKM( std::vector<Vec4i> input, float threshold, float lineMaxDist = LINE_MAX_DIST, unsigned int minLines = 1 );
void FilterByAKM( std::vector<Vec2f>& input, float threshold, float lineMaxDist = LINE_MAX_DIST, unsigned int minLines = 1 );
// Termina AKM

// Começa Auxiliary Funcs

double linesSimilarity( Vec2f lineA, Vec2f lineB, float maxDistance = 50 );
double linesSimilarity( Vec4i segLineA, Vec4i segLineB, float maxDistance = 100 );
Vec2f ConvertSegLineToLine(Vec4i segLine);

// Termina Auxiliary Funcs

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

	// imshow("img", img);
	// imshow("gx", gx);
	// imshow("gy", gy);
	// imshow("mag", mag);
	// imshow("angle", angle);

	Mat magSave;
	mag.convertTo(magSave, CV_8UC1, 255);
	imwrite("./debug-data/sobel-mag.jpg", magSave);

	// Começa GLCM (matriz de homogeneidades)
	int featuresWidth = std::ceil((img.cols-(WINDOW_SIZE-1))/X_STEP);
	int featuresHeight = std::ceil((img.rows-(WINDOW_SIZE-1))/Y_STEP);
	Mat featuresMatGLCM0( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
	Mat featuresMatGLCM45( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
	Mat featuresMatGLCM90( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
	Mat featuresMatGLCM135( {featuresHeight, featuresWidth, NUM_FEATURES}, CV_32F);
	{
		Mat gray;
		img.convertTo(gray, CV_8U, 255.);
		cvtColor(gray, gray, COLOR_BGR2GRAY);
		float GLCM0[LIMIAR_CINZA][LIMIAR_CINZA] = {0.};
		float GLCM45[LIMIAR_CINZA][LIMIAR_CINZA] = {0.};
		float GLCM90[LIMIAR_CINZA][LIMIAR_CINZA] = {0.};
		float GLCM135[LIMIAR_CINZA][LIMIAR_CINZA] = {0.};
		float featuresGLCM0[NUM_FEATURES] = {0.};
		float featuresGLCM45[NUM_FEATURES] = {0.};
		float featuresGLCM90[NUM_FEATURES] = {0.};
		float featuresGLCM135[NUM_FEATURES] = {0.};

		printf("Computando GLCMs:\n0%%... ");

		for(int grayY = 0; grayY + WINDOW_SIZE < gray.rows; grayY += Y_STEP){ // Faz a janela andar pela imagem

			if(grayY > gray.rows/2 && grayY - Y_STEP <= gray.rows/2) {
				printf("50%%... ");
			}

			for(int grayX = 0; grayX + WINDOW_SIZE < gray.cols; grayX += X_STEP){
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
					float homogeneity = 0;
					for(int i = 0; i < LIMIAR_CINZA; i++){
						for(int j = 0; j < LIMIAR_CINZA; j++){
							GLCM0[i][j] /= soma;
							homogeneity += GLCM0[i][j]/(1+(i-j)*(i-j));
						}
					}
					featuresGLCM0[0] = homogeneity;
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
					float homogeneity = 0;
					for(int i = 0; i < LIMIAR_CINZA; i++){
						for(int j = 0; j < LIMIAR_CINZA; j++){
							GLCM45[i][j] /= soma;
							homogeneity += GLCM45[i][j]/(1+(i-j)*(i-j));
						}
					}
					featuresGLCM45[0] = homogeneity;
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
					float homogeneity = 0;
					for(int i = 0; i < LIMIAR_CINZA; i++){
						for(int j = 0; j < LIMIAR_CINZA; j++){
							GLCM90[i][j] /= soma;
							homogeneity += GLCM90[i][j]/(1+(i-j)*(i-j));
						}
					}
					featuresGLCM90[0] = homogeneity;
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
					float homogeneity = 0;
					for(int i = 0; i < LIMIAR_CINZA; i++){
						for(int j = 0; j < LIMIAR_CINZA; j++){
							GLCM135[i][j] /= soma;
							homogeneity += GLCM135[i][j]/(1+(i-j)*(i-j));
						}
					}
					featuresGLCM135[0] = homogeneity;
				}

				// SALVAR CARACTERISTICAS
				for(int k = 0; k < NUM_FEATURES; k++) {
					featuresMatGLCM0.at<float>(grayY/Y_STEP, grayX/X_STEP, k) = featuresGLCM0[k];
					featuresMatGLCM45.at<float>(grayY/Y_STEP, grayX/X_STEP, k) = featuresGLCM45[k];
					featuresMatGLCM90.at<float>(grayY/Y_STEP, grayX/X_STEP, k) = featuresGLCM90[k];
					featuresMatGLCM135.at<float>(grayY/Y_STEP, grayX/X_STEP, k) = featuresGLCM135[k];
				}
			}
		}

		printf("100%%\n");

		{
			FileStorage ymlFeatures("./debug-data/Features.yml", FileStorage::WRITE);
			ymlFeatures << "FeatureOrder" << FEATURES_ORDER;
			ymlFeatures << "GLCM-0" << featuresMatGLCM0;
			ymlFeatures << "GLCM-45" << featuresMatGLCM45;
			ymlFeatures << "GLCM-90" << featuresMatGLCM90;
			ymlFeatures << "GLCM-135" << featuresMatGLCM135;

			Mat homogeneidade;

			ExtractFeatureMat(featuresMatGLCM0, 0).convertTo(homogeneidade, CV_8U, 255);
			imwrite("./debug-data/GLCM0-Homogeneidade.jpg", homogeneidade);

			ExtractFeatureMat(featuresMatGLCM45, 0).convertTo(homogeneidade, CV_8U, 255);
			imwrite("./debug-data/GLCM45-Homogeneidade.jpg", homogeneidade);

			ExtractFeatureMat(featuresMatGLCM90, 0).convertTo(homogeneidade, CV_8U, 255);
			imwrite("./debug-data/GLCM90-Homogeneidade.jpg", homogeneidade);

			ExtractFeatureMat(featuresMatGLCM135, 0).convertTo(homogeneidade, CV_8U, 255);
			imwrite("./debug-data/GLCM135-Homogeneidade.jpg", homogeneidade);
		}
	}

	// Termina GLCM

	printf("GLCMs computadas. Calculando Transformada de Hough...\n");

	// Começa Hough (Transformada de Hough (rho theta), mesmo alpha)

	{
		// Binariza
		Mat binarizado, gray, operado;
		cvtColor(mag, gray, COLOR_BGR2GRAY);
		threshold(gray, binarizado, BIN_THRES, 255, THRESH_BINARY/100);
		binarizado.convertTo(binarizado, CV_8UC1);
		// imshow("Hough - binarizado", binarizado);
		imwrite("./debug-data/binarizado.jpg", binarizado);

		// Faz fechamento customizado
		morphologyEx(binarizado, operado, MORPH_DILATE, getStructuringElement( MORPH_ELLIPSE, Size(TAM_DILATA, TAM_DILATA) ));
		morphologyEx(operado, operado, MORPH_ERODE, getStructuringElement( MORPH_ELLIPSE, Size(TAM_ERODE, TAM_ERODE) ));
		// imshow("Hough - operado", operado);
		imwrite("./debug-data/morph-operado.jpg", operado);

		std::vector<Vec2f> linesDirections;
		{ // Acha as linhas por Hough
			std::vector<Vec2f> lines;
			HoughLines(operado, lines, DIST_RES, ANGLE_RES, HOUGH_THRES);
			
			Mat houghWithLines = gray.clone();
			cvtColor(operado, houghWithLines, COLOR_GRAY2BGR);
			for( unsigned int i = 0; i < lines.size(); i++ ) {
				float rho = lines[i][0];
				float theta = lines[i][1];

				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;

				Point pt1(cvRound( x0 + TAM_LINE*(-b) ), cvRound( y0 + TAM_LINE*a ));
				Point pt2(cvRound( x0 - TAM_LINE*(-b) ), cvRound( y0 - TAM_LINE*a ));
				line( houghWithLines, pt1, pt2, Scalar(0,0,255), 1, 8 );
			}

			// imshow("Hough - Linhas", houghWithLines);
			imwrite("./debug-data/hough-linhas.jpg", houghWithLines);

			// Compute e desenha a clusterização adaptativa
			std::vector<Vec2f> clusters = AKM(lines, LINE_THRES);
			FilterByAKM(clusters, 0.9, 100000, 2);

			linesDirections = clusters;

			Mat houghWithClusteredLines = gray.clone();
			cvtColor(operado, houghWithClusteredLines, COLOR_GRAY2BGR);
			for( unsigned int i = 0; i < clusters.size(); i++ ) {
				float rho = clusters[i][0];
				float theta = clusters[i][1];

				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;

				Point pt1(cvRound( x0 + TAM_LINE*(-b) ), cvRound( y0 + TAM_LINE*a ));
				Point pt2(cvRound( x0 - TAM_LINE*(-b) ), cvRound( y0 - TAM_LINE*a ));
				line( houghWithClusteredLines, pt1, pt2, Scalar(0,0,255), 2, 8 );
			}

			// imshow("Hough - Linhas Clusterizadas", houghWithClusteredLines);
			imwrite("./debug-data/hough-linhas-cluster.jpg", houghWithClusteredLines);
		}
		printf("Hough computado. Calculando Hough probabilistico...\n");
		{ // Acha os segmentos de linhas por Hough
			std::vector<Vec4i> segLines;
			HoughLinesP(operado, segLines, DIST_RES, ANGLE_RES, HOUGHP_THRES, MIN_LINE_LEN, MAX_LINE_GAP);
		
			Mat houghWithLines = Mat::zeros(gray.rows, gray.cols, gray.type());
			// Mat houghWithLines = operado.clone();
			cvtColor(houghWithLines, houghWithLines, COLOR_GRAY2BGR);
			for( unsigned int i = 0; i < segLines.size(); i++ ) {
				line( houghWithLines, Point(segLines[i][0], segLines[i][1]), Point(segLines[i][2], segLines[i][3]), Scalar(0,0,255), 1, 8 );
			}
		
			// imshow("Hough - Segmento de Linhas", houghWithLines);
			imwrite("./debug-data/hough-segmentolinhas.jpg", houghWithLines);

			for(unsigned int i = 0; i < segLines.size(); i++) {
				Vec2f lineI = ConvertSegLineToLine(segLines[i]);
				unsigned int j;
				for(j = 0; j < linesDirections.size(); j++) {
					if(linesSimilarity(linesDirections[j], lineI, 1000) > 0.95) {
						break;
					}
				}
				if(j >= linesDirections.size()) {
					segLines.erase(segLines.begin() + i--);
				}
			}

			// Compute e desenha a clusterização adaptativa
			std::vector<Vec4i> clusters = AKM(segLines, SEGLINE_THRES, 100);
		
			Mat houghWithClusteredLines = Mat::zeros(gray.rows, gray.cols, gray.type());
			// Mat houghWithClusteredLines = operado.clone();
			cvtColor(houghWithClusteredLines, houghWithClusteredLines, COLOR_GRAY2BGR);
			for( unsigned int i = 0; i < clusters.size(); i++ ) {
				line( houghWithClusteredLines, Point(clusters[i][0], clusters[i][1]), Point(clusters[i][2], clusters[i][3]), Scalar(0,0,255), 1, 8 );
			}
		
			// imshow("Hough - Segmento de Linhas Clusterizado", houghWithClusteredLines);
			imwrite("./debug-data/hough-segmentolinhas-cluster.jpg", houghWithClusteredLines);
		}
	}

	// Termina Hough

	waitKey(0);
	printf("Pronto!\n");
	return 0;
}

// Começa GLCM (matriz de homogeneidades)

Mat ExtractFeatureMat(Mat featuresMat, int featureNum) {
	int sizeY = featuresMat.size[0];
	int sizeX = featuresMat.size[1];
	Mat feature(sizeY, sizeX, CV_32F);
	for(int j = 0; j < sizeY; j++){
		for(int i = 0; i < sizeX; i++){
			feature.at<float>(j, i) = featuresMat.at<float>(j, i, featureNum);
		}
	}
	return feature;
}

// Termina GLCM

// Começa AKM

std::vector<Vec2f> AKM( std::vector<Vec2f> input, float threshold, float lineMaxDist, unsigned int minLines ) {
	if(input.size() < 1) {
		return std::vector<Vec2f>();
	}
	std::vector<Vec2f> means;
	std::vector<unsigned int> amount;
	means.push_back(input[0]);
	amount.push_back(1);
	for(unsigned int i = 1; i < input.size(); ++i) {
		unsigned int closestK = 0;
		double biggestSimilarity = 0.;
		for(unsigned int k = 0; k < means.size(); ++k) {
			double similarity = linesSimilarity(input[i], means[k], lineMaxDist);
			if(similarity > biggestSimilarity) {
				biggestSimilarity = similarity;
				closestK = k;
			}
		}
		if(biggestSimilarity > threshold) {
			Vec2f oldMean = means[closestK] * ((float)amount[closestK]++);
			means[closestK] = (oldMean + input[i])/((float)amount[closestK]);
			for(unsigned int a = 0; a < means.size(); ++a) {
				for(unsigned int b = a+1; b < means.size(); ++b) {
					double similarity = linesSimilarity(means[a], means[b], lineMaxDist);
					if(similarity > threshold) {
						float newAmount = amount[b]+amount[a];
						Vec2f newMean = ( means[b]*((float)amount[b]) + means[a]*((float)amount[a]) )/newAmount;
						means[a] = newMean;
						amount[a] = newAmount;
						means.erase(means.begin()+b);
						amount.erase(amount.begin()+ b--);
					}
				}
			}
		} else {
			means.push_back(input[i]);
			amount.push_back(1);
		}
	}
	for(unsigned int a = 0; a < means.size(); ++a) {
		if(amount[a] < minLines) {
			means.erase(means.begin()+a);
			amount.erase(amount.begin()+ a--);
		}
	}
	return means;
}

std::vector<Vec4i> AKM( std::vector<Vec4i> input, float threshold, float lineMaxDist, unsigned int minLines ) {
	if(input.size() < 1) {
		return std::vector<Vec4i>();
	}
	std::vector<Vec4i> means;
	std::vector<unsigned int> amount;
	means.push_back(input[0]);
	amount.push_back(1);
	for(unsigned int i = 1; i < input.size(); ++i) {
		unsigned int closestK = 0;
		double biggestSimilarity = 0.;
		for(unsigned int k = 0; k < means.size(); ++k) {
			double similarity = linesSimilarity(input[i], means[k], lineMaxDist);
			if(similarity > biggestSimilarity) {
				biggestSimilarity = similarity;
				closestK = k;
			}
		}
		if(biggestSimilarity > threshold) {
			Vec4i oldMean = means[closestK] * ((float)amount[closestK]++);
			means[closestK] = (oldMean + input[i])/((float)amount[closestK]);
			for(unsigned int a = 0; a < means.size(); ++a) {
				for(unsigned int b = a+1; b < means.size(); ++b) {
					double similarity = linesSimilarity(means[b], means[a], lineMaxDist);
					if(similarity > threshold) {
						float newAmount = amount[b]+amount[a];
						Vec4i newMean = ( means[b]*((float)amount[b]) + means[a]*((float)amount[a]) )/newAmount;
						means[a] = newMean;
						amount[a] = newAmount;
						means.erase(means.begin()+b);
						amount.erase(amount.begin()+ b--);
					}
				}
			}
		} else {
			means.push_back(input[i]);
			amount.push_back(1);
		}
	}
	for(unsigned int a = 0; a < means.size(); ++a) {
		if(amount[a] < minLines) {
			means.erase(means.begin()+a);
			amount.erase(amount.begin()+ a--);
		}
	}
	return means;
}

void FilterByAKM( std::vector<Vec2f>& input, float threshold, float lineMaxDist, unsigned int minLines ) {
	std::vector<Vec2f> clusters = AKM(input, threshold, lineMaxDist, minLines);
	for(unsigned int i = 1; i < input.size(); ++i) {
		double biggestSimilarity = 0.;
		for(unsigned int k = 0; k < clusters.size(); ++k) {
			double similarity = linesSimilarity(input[i], clusters[k], lineMaxDist);
			if(similarity > biggestSimilarity) {
				biggestSimilarity = similarity;
			}
		}
		if(biggestSimilarity < threshold) {
			input.erase(input.begin()+ i--);
		}
	}
}

double linesSimilarity(Vec2f lineA, Vec2f lineB, float maxDistance) {
	double angleSim = 1 - std::abs(lineA[1] - lineB[1])/(M_PI/2);
	double distSim = 1 - std::abs(lineA[0] - lineB[0])/maxDistance;
	return angleSim * (distSim > 0 ? distSim : 0);
}

double linesSimilarity( Vec4i segLineA, Vec4i segLineB, float maxDistance ) {

	Vec2f midA((segLineA[0]+segLineA[2])/2, (segLineA[1]+segLineA[3])/2);
	Vec2f midB((segLineB[0]+segLineB[2])/2, (segLineB[1]+segLineB[3])/2);

	Vec2f lineA = ConvertSegLineToLine(segLineA);
	Vec2f lineB = ConvertSegLineToLine(segLineB);

	double distSim = 1 - std::sqrt( std::pow( midA[0]-midB[0], 2) + std::pow( midA[1]-midB[1], 2) )/maxDistance;

	return linesSimilarity(lineA, lineB, maxDistance) * (distSim > 0 ? distSim : 0);
}

Vec2f ConvertSegLineToLine(Vec4i segLine) {
	Point mid((segLine[0]+segLine[2])/2, (segLine[1]+segLine[3])/2);

	float theta = std::atan2( segLine[3]-segLine[1], segLine[2]-segLine[0] ) + 90*M_PI/180;
	float rho = mid.x*std::cos(theta) + mid.y*std::sin(theta);

	return Vec2f(rho, theta);
}

// Termina AKM
