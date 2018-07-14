#include<opencv2/opencv.hpp>
#include"img_proc.h"
#include<stdio.h> 

using namespace cv;
using namespace std;


//**********************����NLTV�㷨����*************************

//    sRadius                ��������뾶
//    nRadius                ��������뾶
//    fParam                 �˲�����
//    deviation              ��˹�˱�׼��
//    lamda                  ƽ�����
//    theta                  ƽ�����
//    pathInput              ����ͼ��·��
//    pathClean              ����ͼ��·��

//****************************************************************

//�����㷨����
IMP_S32 sRadius   = 5;
IMP_S32 nRadius   = 5;
IMP_F32 fParam    = 22;
IMP_F32 deviation = 2;
IMP_F32 lamda     = 1;
IMP_F32 theta     = 50;
string  pathInput = "G:\\opencv_image\\lena_256_20.png";
string  pathClean = "G:\\opencv_image\\lena_256_0.png";


/**************��������ʼ*****************/
void main()
{ 	
	
	//��������
	IMP_F32 time;
	IMP_MAT imgInput, imgClean, imgOutput;
    IMP_U08 *workBuf = NULL;
	NLTVHandle *nltvhandle = (NLTVHandle*)malloc(sizeof(NLTVHandle));
	NLTVParam  *nltvParam  = (NLTVParam *)malloc(sizeof(NLTVParam));

	//����ͼ��
	imgClean = imread(pathClean, 0);
	imgInput = imread(pathInput, 0);
	imgOutput.create(imgInput.rows, imgInput.cols, CV_8UC1);

	//ͼ��·�����
	if (!imgInput.data || !imgClean.data)
	{
		printf("��·����û���ҵ�����ͼ�������ͼ��!\n");
		printf("��������˳�����\n");
		getchar();
		exit(0);
	}
	

	//�㷨������ֵ��nltvParam
	nltvParam->sRadius   = sRadius;
	nltvParam->nRadius   = nRadius;
	nltvParam->fParam    = fParam;
	nltvParam->deviation = deviation;
	nltvParam->lamda	 = lamda;
	nltvParam->theta     = theta;
	nltvParam->height    = imgInput.rows;
	nltvParam->width     = imgInput.cols;
	
	//nltvȥ�룬����ʼ��ʱ
	time = (IMP_F32)getTickCount();

	NltvDenoising((unsigned char*)imgInput.data, (unsigned char*)imgOutput.data, nltvhandle, nltvParam);

    time = IMP_F32((getTickCount() - time) / getTickFrequency());

	//�����ʱ��PSNR��ȥ����ͼ��
	printf("��ʱ��%.2f��\n", time);
	printf("PSNR = %.2f\n", psnr(imgOutput, imgClean));
	printf("SSIM = %.3f\n", ssim(imgOutput, imgClean));

	namedWindow("imgOutput", WINDOW_NORMAL);
	imshow("imgOutput", imgOutput);
	waitKey(0);
}
