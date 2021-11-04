#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

int main()
{

	//Version check code is from Satya Mallick's post https://learnopencv.com/how-to-find-opencv-version-python-cpp/
	std::cout << "OpenCV version : " << CV_VERSION << std::endl;
	std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
	std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
	std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
	std::cout << "*****************************************************" << std::endl;
	
	std::string imDir = "C:\\Users\\batuh\\OneDrive\\Masaüstü\\lena.png"; //Change it to a proper image path for your system
	cv::Mat inputImage = cv::imread(imDir, cv::IMREAD_UNCHANGED);


	cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
	cv::imshow("Input Image", inputImage);
	cv::waitKey();

	//Test of CPU functions
	cv::Mat grayImage;
	cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
	cv::namedWindow("Gray Image", cv::WINDOW_NORMAL);
	cv::imshow("Gray Image", grayImage);
	cv::waitKey();

	//Test of GPU availability and functions
	int cudaEnabDevCount = cv::cuda::getCudaEnabledDeviceCount();

	if(cudaEnabDevCount)
		std::cout << "Number of available CUDA device(s): " << cudaEnabDevCount << std::endl;
	else
		std::cout << "You don't have any available CUDA device(s)" << std::endl;
	std::cout << "*****************************************************" << std::endl;

	std::cout << "List of all available CUDA device(s):" << std::endl;
	for (int devId = 0; devId < cudaEnabDevCount; ++devId) {
		cv::cuda::setDevice(devId);
		std::cout << "Available "; cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	}
	std::cout << "*****************************************************" << std::endl;

	cv::cuda::DeviceInfo cudaDeviceInfo;
	bool devCompatib = false;

	std::cout << "List of all compatiable CUDA device(s):" << std::endl;
	for (int devId = 0; devId < cudaEnabDevCount; ++devId) {
		cudaDeviceInfo = cv::cuda::DeviceInfo::DeviceInfo(devId);
		devCompatib = cudaDeviceInfo.isCompatible();

		if (devCompatib)
			std::cout << "Compatiable "; cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	}
	std::cout << "*****************************************************" << std::endl;

	cv::cuda::GpuMat inputImageGpu, imageGrayGpu;
	inputImageGpu.upload(inputImage);
	cv::cuda::cvtColor(inputImageGpu, imageGrayGpu, cv::COLOR_BGR2GRAY);

	cv::Mat imageGrayCpu;
	imageGrayGpu.download(imageGrayCpu);

	cv::namedWindow("Gray Image GPU", cv::WINDOW_NORMAL);
	cv::imshow("Gray Image GPU", imageGrayCpu);
	cv::waitKey();

	cv::destroyAllWindows();

}


