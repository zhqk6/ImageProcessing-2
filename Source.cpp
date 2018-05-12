#include <highgui.h>
#include <cv.h>  
#include <ml.h>
#include <cvaux.h>
#include <iostream>
#include <math.h>
#include <windows.h>
#include <random>

using namespace std;
using namespace cv;


Mat shifting(Mat input)
//shifting function which is used for shifting 4 parts of rectangles on padded image, so that we can get the correct image.
//It returns input itself which is after shifting. 
//Input is padded input image.
//Padded image can be separated equally for 4 parts, named upleft, upright, downleft, downright.
{
	Mat upleft, upright, downleft, downright, storage;
	upleft = input(Rect(0, 0, input.cols / 2, input.rows / 2));
	upright = input(Rect(input.cols / 2, 0, input.cols / 2, input.rows / 2));
	downleft = input(Rect(0, input.rows / 2, input.cols / 2, input.rows / 2));
	downright = input(Rect(input.cols / 2, input.rows / 2, input.cols / 2, input.rows / 2));
	upleft.copyTo(storage);
	downright.copyTo(upleft);
	storage.copyTo(downright);
	upright.copyTo(storage);
	downleft.copyTo(upright);
	storage.copyTo(downleft);
	//like[1 2;3 4],1<==>4,2<==>3;
	return input;
}

void spectrum_(Mat input, string name)
//spectrum_ function which is used for producing the 2-channel input image's spectrum
//input is input image, name is the title of image which is showed.
{
	Mat ImgPlane[2];
	Mat ImgMagnitude;
	split(input, ImgPlane);
	magnitude(ImgPlane[0], ImgPlane[1], ImgMagnitude);
	//calculate the magnitude of the 2-channel image;

	ImgMagnitude += Scalar::all(1);
	log(ImgMagnitude, ImgMagnitude);
	normalize(ImgMagnitude, ImgMagnitude, 0, 1, CV_MINMAX);
	imshow(name, ImgMagnitude);
}

Mat mydft(Mat inputIMG, int k)
//DFT function
//inputIMG is padded input image(larger than original image).
//it returns the magnitude Fourier spectrums of the image after DFT and after normalization if k=1;
//it returns the 2-channel MergedImg which is before magnitude() if k=0 .(Because this parameter will be used for filtering and IDFT later);
{

	Mat MergedImg;
	Mat ImgPlane[2] = { inputIMG , Mat::zeros(inputIMG.size(), CV_32F) };
	merge(ImgPlane, 2, MergedImg);
	//Create  2-channels Mat MergerdImg.

	dft(MergedImg, MergedImg);
	if (k == 0) {
		return MergedImg;
	}
	split(MergedImg, ImgPlane);
	//Do DFT transfer and seperate the Merged image to 2 planes after DFT.

	Mat ImgMagnitude;
	magnitude(ImgPlane[0], ImgPlane[1], ImgMagnitude);
	ImgMagnitude += Scalar::all(1);
	log(ImgMagnitude, ImgMagnitude);
	//Calculate the magnitude and get the log of magnitude. 
	//It's log(1+sqrt(RE()^2+IM()^2)) 

	ImgMagnitude = shifting(ImgMagnitude);
	//We need to separate the whole magnitude image to 4 rectangle parts with equal size. 
	//And exchange the place of part 'upleft' and part 'downright',exchange 'upright' and 'downleft'. So that we can get our final spectrum

	normalize(ImgMagnitude, ImgMagnitude, 0, 1, CV_MINMAX);

	if (k == 1) {
		return ImgMagnitude;
	}
	//k=0, we can get 2-channel Mat MergedImg(after dft()). 
	//k=1, we can get the spectrum of the MergedImg(after dft()).
}

void ILPF(Mat input, int D0)
//ILPF Function which is used for creating ideal low pass filter and filtering
//input is padded input image, in this program parameter D0 is 40
{
	int a, b, i, j;
	a = (input.rows);
	b = (input.cols);
	Mat ILPFkernel(a, b, CV_32FC2);
	// ILPFkernel.size = input.size

	float D;
	for (i = 0; i < a; i++) {
		for (j = 0; j < b; j++) {
			D = float(sqrt((a / 2 - i)*(a / 2 - i) + (b / 2 - j)*(b / 2 - j)));
			if (D <= D0) {
				ILPFkernel.at<Vec2f>(i, j)[0] = 1.0;
				ILPFkernel.at<Vec2f>(i, j)[1] = 1.0;
			}
			else {
				ILPFkernel.at<Vec2f>(i, j)[0] = 0.0;
				ILPFkernel.at<Vec2f>(i, j)[1] = 0.0;
			}
			//if D<=D0, 2 channel of ILPFkernel is 1,otherwise 0.
		}
	}
	spectrum_(ILPFkernel,"spectrum of ideal filter");

	Mat Result;
	Mat A = mydft(input, 0);
	A = shifting(A);
	Result = A.mul(ILPFkernel);
	//A is returned value(MergedImg) of mydft. 
	//Element-wise multiplication is done between GLPFkernel and A.

	spectrum_(Result, "Spectrum after ideal filter");
	Result = shifting(Result);
	//Get the result spectrum. We also need to shift before do idft.

	Mat IDFT_, IDFT;
	idft(Result, IDFT_, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(IDFT_, IDFT_, 0, 1, CV_MINMAX);
	IDFT = IDFT_(Rect(0, 0, a / 2, b / 2));
	imshow("ILPF output", IDFT);
	//Inverse DFT and get the upleft rectangle of the result
}

void GLPF(Mat input, int D0)
//GLPF Function which is used for creating Gaussian low pass filter and filtering
//input is padded input image, in this program parameter D0 is 40
{
	int a, b, i, j;
	a = (input.rows);
	b = (input.cols);
	double D;
	Mat GLPFkernel(a, b, CV_32FC2);
	for (i = 0; i < a; i++) {
		for (j = 0; j < b; j++) {
			D = double(sqrt((a / 2 - i)*(a / 2 - i) + (b / 2 - j)*(b / 2 - j)));
			GLPFkernel.at<Vec2f>(i, j)[0] = exp(-(D*D) / (2 * D0*D0));
			GLPFkernel.at<Vec2f>(i, j)[1] = exp(-(D*D) / (2 * D0*D0));
			//Create Gaussian kernel, function exp() needs double type
		}
	}
	spectrum_(GLPFkernel, "spectrum of Gaussian filter");

	Mat Result;
	Mat A = mydft(input, 0);
	A = shifting(A);
	Result = A.mul(GLPFkernel);
	//A is returned value(MergedImg) of mydft. 
	//Element-wise multiplication is done between GLPFkernel and A.

	spectrum_(Result, "Spectrum after Gaussian filter");
	Result = shifting(Result);
	//Get the result spectrum. We also need to shift before do idft.

	Mat IDFT_, IDFT;
	idft(Result, IDFT_, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(IDFT_, IDFT_, 0, 1, CV_MINMAX);
	IDFT = IDFT_(Rect(0, 0, a / 2, b / 2));
	imshow("GLPF output", IDFT);
	//Inverse DFT and get the upleft rectangle of the result
}

Mat degrade(Mat input, float k, int m ,int stv, int state) {
//degrade function is used for image degradation.
//including degrade function H(u,v) and Gaussian noise() N(u,v)
//G(u,v)=H(u,v)F(u,v)+N(u,v)
//input is the input image. m is the mean of Gaussian noise.
//stv is the standard deviation of Gaussian noise
//if state=0, it returns degradeF, the degrade function H(u,v)
//if state=1, it returns degradeImg, the degraded image G(u,v)
	int a, b, i, j;
	a = (input.rows);
	b = (input.cols);
	Mat degradeF(a, b, CV_32FC2);
	float D;
	for (i = 0; i < a; i++) {
		for (j = 0; j < b; j++) {
			D = (i - a / 2.0)*(i - a / 2.0) + (j - b / 2.0)*(j - b / 2.0);
			degradeF.at<Vec2f>(i, j)[0]=  exp((-k)*(pow(D,(5.0/6.0))));
			degradeF.at<Vec2f>(i, j)[1] = exp((-k)*(pow(D, (5.0 / 6.0))));
		}
	}
	if (state == 0) {
		return degradeF;
	}
	//The atmospheric turbulence model

	Mat MergedImg =mydft(input,0);
	MergedImg = shifting(MergedImg);
	Mat H;
	H = degradeF.mul(MergedImg);
	H = shifting(H);
	Mat IDFT_,IDFT;
	idft(H, IDFT_, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(IDFT_, IDFT_, 0, 1, CV_MINMAX);
	imshow("Image after atmospheric turbulence model", IDFT_);
	//show the image after H(u,v)F(u,v)

	Mat G (a, b, CV_32F);
	randn(G,m,stv);
	G = G / 255;
	//Create random Gaussian noise

	Mat MergedG = mydft(G,0);
	MergedG= shifting(MergedG);
	Mat degradeImg = MergedG + H;
	//Mat degradeshift = shifting(degradeImg);
	idft(degradeImg, IDFT, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(IDFT, IDFT, 0, 1, CV_MINMAX);
	imshow("Image after atmospheric turbulence model and Gaussian noise", IDFT);
	if (state == 1) {
		return degradeImg;
	}
	//add Gaussian noise
}

Mat inverF(Mat degradedImg, Mat degradeFunction, int D0, double n) {
//function inverF is create Butterworth filter kernel, and then do inverse filtering
//degradedImg is degraded image G(u,v)
//degradeFunction is degade function H(u,v)
//D0 is the radius of Butterworth filter. n is the order of the Butterworth filter
//F(u,v)=BWkernel(u,v)(G(u,v)/H(u,v)).
//It returns the F(u,v) which is after filtering
	int a, b, i, j;
	a = (degradedImg.rows);
	b = (degradedImg.cols);
	Mat BWkernel(a, b, CV_32FC2);
	double D;
	for (i = 0; i < a; i++) {
		for (j = 0; j < b; j++) {
			D = (sqrt((a / 2- i)*(a / 2 - i) + (b / 2 - j)*(b / 2 - j)));
			BWkernel.at<Vec2f>(i, j)[0] = 1 / (1 + (pow(double(D / D0), (2.0 * n))));
			BWkernel.at<Vec2f>(i, j)[1] = 1 / (1 + (pow(double(D / D0), (2.0 * n))));
		}
	}
	//Create Butterworth kernel

	Mat trans(a,b,CV_32FC2);
	Mat Result(a, b, CV_32FC2);
	trans = degradedImg  / degradeFunction;
	Result = BWkernel.mul(trans);
	return Result;
	//Do inverse filtering

}

Mat wiener( Mat degradedImg, Mat degradeFunction, float K) {
//function wiener is do wiener filtering
//Returns the result after filtering
//degradedImg is degraded image G(u,v)
//degradeFunction is degade function H(u,v)
//K is constant K
	int a, b, i, j;
	a = (degradedImg.rows);
	b = (degradedImg.cols);
	Mat square(a, b, CV_32FC1);
	Mat trans1(a, b, CV_32FC2);
	Mat trans2(a, b, CV_32FC1);
	Mat trans2_(a, b, CV_32FC2);
	Mat trans3(a, b, CV_32FC2);
	Mat wiener1(a, b, CV_32FC2);

	Mat plane[2];
	split(degradeFunction,plane);
	magnitude(plane[0],plane[1],square);

	trans1 =(square + K);
	trans2 = square /trans1;
	Mat plane_[2] = { square,Mat::zeros(square.size(),CV_32F) };
	merge(plane_,2, trans2_);
	
	trans3 = trans2_ / degradeFunction;
	//Wiener Filter
    
	wiener1 = trans3.mul(degradedImg);
	return wiener1;
}

double SNR(Mat input1, Mat input2) 
//SNR function, calculate the ratio of original image and the image after wiener filtering
//input 1 is image after wiener filtering
//input 2 is original image
{
	double Ratio;
	Mat diff, square;
	int i, j;
	diff = (input1 - input2).mul(input1 - input2);
	square= (input1).mul(input1);
	Ratio=(sum(square)/sum(diff)).val[0];
	return Ratio;
}

int main(int argc, char ** argv)
{
	Mat A = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original Image cameraman.png", A);
	int i, j;
	i = 2 * (A.rows);
	j = 2 * (A.cols);
	//The length of the input should be 2^n to calculate DFT

	Mat filledImg;
	copyMakeBorder(A, filledImg, 0, i - A.rows, 0, j - A.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat TransA;
	filledImg.convertTo(TransA, CV_32F);
	//Fill the image use 0 because the length is 2^n, 'filledImg' is the filled image.
	//And change the type of filledImg from unsigned to float, we need to use float type later.

	TransA /= 255;
	Mat TransA1 = TransA.clone();
	Mat TransA2 = TransA.clone();
	imshow("The padded image", TransA);
	//Normalize and show the padded image 'TransA'
	Mat spectrum = mydft(TransA, 1);
	imshow("Spectrum of the original Image cameraman.png", spectrum);

	ILPF(TransA1, 40); //Do ideal low pass filter
	GLPF(TransA2, 40); //Do Gaussian low pass filter

	Mat B = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original Image noturbulence.png", B);
	Mat TransB,TransB_;
	B.convertTo(TransB, CV_32F);
	TransB /= 255;
	TransB_=TransB.clone();
	Mat Degrade_function=degrade(TransB,0.0025,0,1,0);
	Mat Degraded_Image = degrade(TransB, 0.0025, 0, 1, 1);
	//Do image degradation. Degrade_function is H(u,v).Degraded_Image is G(u,v)

	Degraded_Image = shifting(Degraded_Image);
	Mat Result1=inverF(Degraded_Image, Degrade_function,70,10.0);
	Mat IIDFT,IIDFT1;
	Result1 = shifting(Result1);
	idft(Result1, IIDFT, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(IIDFT, IIDFT, 0, 1, CV_MINMAX);
	imshow("Inverse filter r=70 output", IIDFT);
	Mat Result2 = inverF(Degraded_Image, Degrade_function, 100, 10.0);
	Result2 = shifting(Result2);
	idft(Result2, IIDFT1, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(IIDFT1, IIDFT1, 0, 1, CV_MINMAX);
	imshow("Inverse filter r=100 output", IIDFT1);
	//Do inverse filter. Result1 is with 100 radius. Result2 is with 70 radius

	Mat wiener1=wiener( Degraded_Image, Degrade_function, 1.0);
	Mat wiener2=wiener( Degraded_Image, Degrade_function, 0.01);
	Mat wiener3=wiener( Degraded_Image, Degrade_function, 0.0001);
	wiener1 = shifting(wiener1);
	wiener2  = shifting(wiener2);
	wiener3 = shifting(wiener3);
	//Do wiener filtering

	Mat Widft1, Widft2, Widft3;
	idft(wiener1, Widft1, DFT_REAL_OUTPUT);
	normalize(Widft1, Widft1, 0, 1, CV_MINMAX);
	imshow("wiener output K=1", Widft1);

	idft(wiener2, Widft2, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(Widft2, Widft2, 0, 1, CV_MINMAX);
	imshow("wiener output K=10^-2", Widft2);

	idft(wiener3, Widft3, DFT_SCALE | DFT_REAL_OUTPUT);
	normalize(Widft3, Widft3, 0, 1, CV_MINMAX);
	imshow("wiener ourput K=10^-4", Widft3);
	//show the image after wiener filter

	double SNR1 = SNR(Widft1, TransB_);
	double SNR2 = SNR(Widft2, TransB_);
	double SNR3 = SNR(Widft3, TransB_);
	cout << "For K=1,      SNR1 is \n" << SNR1 << endl;
	cout << "For K=0.01,   SNR2 is \n" << SNR2 << endl;
	cout << "For K=0.0001, SNR3 is \n" << SNR3 << endl;
	//Calculate the Signal to Noise Ratio

	Mat spectrum2 = mydft(TransB, 1);
	imshow("Spectrum of the original Image noturbulence.png", spectrum2);

	waitKey();
	return 0;
}
