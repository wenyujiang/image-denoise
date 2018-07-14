#include<opencv2/opencv.hpp>
#include"img_proc.h"
#include<stdio.h> 

using namespace cv;
using namespace std;


//**********************设置NLTV算法参数*************************

//    sRadius                搜索区域半径
//    nRadius                邻域区域半径
//    fParam                 滤波参数
//    deviation              高斯核标准差
//    lamda                  平衡参数
//    theta                  平衡参数
//    pathInput              含噪图像路径
//    pathClean              无噪图像路径

//****************************************************************

//设置算法参数
IMP_S32 sRadius   = 5;
IMP_S32 nRadius   = 5;
IMP_F32 fParam    = 22;
IMP_F32 deviation = 2;
IMP_F32 lamda     = 1;
IMP_F32 theta     = 50;
string  pathInput = "G:\\opencv_image\\lena_256_20.png";
string  pathClean = "G:\\opencv_image\\lena_256_0.png";


/**************主函数开始*****************/
void main()
{ 	
	
	//申明变量
	IMP_F32 time;
	IMP_MAT imgInput, imgClean, imgOutput;
    IMP_U08 *workBuf = NULL;
	NLTVHandle *nltvhandle = (NLTVHandle*)malloc(sizeof(NLTVHandle));
	NLTVParam  *nltvParam  = (NLTVParam *)malloc(sizeof(NLTVParam));

	//加载图像
	imgClean = imread(pathClean, 0);
	imgInput = imread(pathInput, 0);
	imgOutput.create(imgInput.rows, imgInput.cols, CV_8UC1);

	//图像路径检查
	if (!imgInput.data || !imgClean.data)
	{
		printf("该路径下没有找到含噪图像或无噪图像!\n");
		printf("按任意键退出程序\n");
		getchar();
		exit(0);
	}
	

	//算法参数赋值到nltvParam
	nltvParam->sRadius   = sRadius;
	nltvParam->nRadius   = nRadius;
	nltvParam->fParam    = fParam;
	nltvParam->deviation = deviation;
	nltvParam->lamda	 = lamda;
	nltvParam->theta     = theta;
	nltvParam->height    = imgInput.rows;
	nltvParam->width     = imgInput.cols;
	
	//nltv去噪，并开始计时
	time = (IMP_F32)getTickCount();

	NltvDenoising((unsigned char*)imgInput.data, (unsigned char*)imgOutput.data, nltvhandle, nltvParam);

    time = IMP_F32((getTickCount() - time) / getTickFrequency());

	//输出耗时、PSNR、去噪后的图像
	printf("耗时：%.2f秒\n", time);
	printf("PSNR = %.2f\n", psnr(imgOutput, imgClean));
	printf("SSIM = %.3f\n", ssim(imgOutput, imgClean));

	namedWindow("imgOutput", WINDOW_NORMAL);
	imshow("imgOutput", imgOutput);
	waitKey(0);
}
