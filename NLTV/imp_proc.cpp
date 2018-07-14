#include"img_proc.h"


void Neighborhood(IMP_U08 *curH, IMP_S32 curW, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat)
{

	IMP_S32 i, j;
	IMP_S32 wTemp;
	IMP_U08 *pTem;
	IMP_S32 count = 0;

	for (i = -nRadius; i <= nRadius; i++)
	{
		//将指针移动到邻域点所在行的第一个元素的地址
		pTem = curH + step * i;
		for (j = -nRadius; j <= nRadius; j++)
		{
			//计算该邻点是这一行的第几个元素
			wTemp = curW + j;
			//取出这个邻点，保存在nearMat中。
			nearMat[count] = pTem[wTemp];
			count++;
		}
	}

}

void NearMatofSearchPoint(IMP_U08 *curH, IMP_S32 curW, IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat)
{
	/*
	找到搜索窗口内的所有点的邻点，并将它们保存起来。
    
	假设当前点是c3，搜索半径是1，邻域半径是1。图像矩阵如下：

	a1 a2 a3 a4 a5
	b1 b2 b3 b4 b5
	c1 c2 c3 c4 c5
	d1 d2 d3 d4 d5
	e1 e2 e3 e4 e5

	则c3的搜索点是b2 b3 b4 c2 c3 c4 d2 d3 d4

	所以：
	     b2的邻域点是a1 a2 a3 b1 b2 b3 c1 c2 c3
	     b3的邻域点是a2 a3 a4 b2 b3 b4 c2 c3 c4
	...
		 d3的邻域点是c2 c3 c4 d2 d3 d4 e2 e3 e4
		 d4的邻域点是c3 c4 c5 d3 d4 d5 e3 e4 e5

	将这些搜索点的邻域按向量保存起来：a1 a2 a3 b1 b2 b3 c1 c2 c3、a2 a3 a4 b2 b3 b4 c2 c3 c4、...、c3 c4 c5 d3 d4 d5 e3 e4 e5
	*/
	IMP_S32 i, j;
	IMP_S32 wTemp;
	IMP_U08 *pTem1, *pTem2;
	IMP_S32 nearMatSize = (2 * nRadius + 1)*(2 * nRadius + 1);
	for (i = -sRadius; i <= sRadius; i++)
	{
		pTem1 = curH + i*step;
		for (j = -sRadius; j <= sRadius; j++)
		{
			wTemp = j + curW;
			Neighborhood(pTem1, wTemp, nRadius, step, nearMat);
			nearMat += ngx_align(nearMatSize, aligSize);
		}
	}
}

void weigth(IMP_U08 *nearOfCurP,	 IMP_U08 *nearOfSeaP,	  IMP_F32 *pgaussian, IMP_F32 h, 
	        IMP_S32 sRadius,         IMP_S32 nRadius, IMP_F32 *pweight,   IMP_F32 *sumweight)
{
	/*
	计算权重：
	把之前得到的搜索点的邻域一组一组的取出来，与当前点的邻域进行非局部相似度权重计算。
	*/
	IMP_S32 i, j;
	IMP_S32 mlen, nlen;
	IMP_U08 *ptmp1, *ptmp2;
	IMP_F32 dis;
	IMP_F32 sum, weightSum;
	IMP_F32 H;
	mlen = (2 * sRadius + 1)*(2 * sRadius + 1);
	nlen = (2 * nRadius + 1)*(2 * nRadius + 1);
	ptmp1 = nearOfCurP;
	ptmp2 = nearOfSeaP;
	weightSum = 0;
	H = h*h;
	for (i = 0; i < mlen; i++)
	{   
		sum = 0;
		for (j = 0; j < nlen; j++)
		{
			dis = (IMP_F32)(ptmp1[j] - ptmp2[j]);
			sum += pgaussian[j] * dis*dis;
		}
		pweight[i] = exp(-sum/H);
		weightSum += pweight[i];
		ptmp2 += ngx_align(nlen, aligSize);
	}
	*sumweight = weightSum;
}

void searchPoint(IMP_U08 *p, IMP_S32 w, IMP_S32 n, IMP_S32 step, IMP_U08 *nearMat)
{
	Neighborhood(p, w, n, step, nearMat);
}

void gaussianKernal(IMP_S32 kernelSize, IMP_F32 sigma, IMP_F32 *kernel)
{
	IMP_S32 halfSize = (kernelSize - 1) / 2;
	IMP_F32 *K = (IMP_F32 *)malloc(kernelSize*kernelSize*sizeof(IMP_F32));
	IMP_F32 Sum = 0;

	//生成二维高斯核
	IMP_F32 s2 = 2.0 * sigma * sigma;

	for (int i = (-halfSize); i <= halfSize; i++)
	{
		IMP_S32 m = i + halfSize;
		for (IMP_S32 j = (-halfSize); j <= halfSize; j++)
		{
			IMP_S32 n = j + halfSize;
			IMP_F32 v = exp(-(1.0*i*i + 1.0*j*j) / s2);
			kernel[m * kernelSize + n] = v;
			Sum += v;
		}
	}
	for (int i = 0; i < kernelSize*kernelSize; i++)
	{
		kernel[i] = kernel[i] / Sum;
	}
}

void NltvGetMemSize(IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 width, IMP_S32 height, IMP_S32* bufSize)
{
	IMP_S32 usedSize = 0;
	IMP_S32 nearLen = (2 * nRadius + 1)*(2 * nRadius + 1);
	IMP_S32 searchLen = (2 * sRadius + 1)*(2 * sRadius + 1);
	IMP_S32 cpySize = nRadius + sRadius;

	usedSize += ngx_align(nearLen   * sizeof(IMP_U08), aligSize);			   //累加邻域内存大小,对齐
	usedSize += ngx_align(searchLen * sizeof(IMP_U08), aligSize);              //累加搜索区域内存大小,对齐
	usedSize += ngx_align(nearLen   * sizeof(IMP_U08), aligSize) * searchLen;  //累加搜索区域的邻域内存大小,对齐
	usedSize += ngx_align(searchLen * sizeof(IMP_F32), aligSize);			   //累加weight内存大小,对齐
	usedSize += ngx_align(nearLen   * sizeof(IMP_F32), aligSize);              //累加高斯核大小内存,对齐
	usedSize += ngx_align(width + 2 * cpySize, aligSize) * (height + 2 * cpySize) * sizeof(IMP_U08); //边缘拓展后的图像内存，对齐

	*bufSize = usedSize;
}

void NltvAllocMem(IMP_U08* workBuf, NLTVHandle* nltvHandle, IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 width, IMP_S32 height)
{

	IMP_U08 *workBufTemp = workBuf;
	IMP_S32 cpySize = sRadius + nRadius;
	int nearLen = (2 * nRadius + 1)*(2 * nRadius + 1);
	int searchLen = (2 * sRadius + 1)*(2 * sRadius + 1);

	//workBuf中划分出内存给nltvHandle->pnearMat使用,用来存储当前点的邻域矩阵
	nltvHandle->pnearMat = workBufTemp;
	workBufTemp += ngx_align(nearLen * sizeof(IMP_U08), aligSize);

	//workBuf中划分出内存给nltvHandle->psearchMat使用,用来存储当前点的搜索矩阵
	nltvHandle->psearchMat = workBufTemp;
	workBufTemp += ngx_align(searchLen * sizeof(IMP_U08), aligSize);

	//workBuf中划分出内存给nltvHandle->pnearMatOfsearchPoint使用，用来存储搜索窗内的所有点的邻域矩阵
	nltvHandle->pnearMatOfsearchPoint = workBufTemp;
	workBufTemp += ngx_align(nearLen  *sizeof(IMP_U08), aligSize) * searchLen;

	//workBuf中划分出内存给nltvHandle->pwightMat使用，用来存储权重
	nltvHandle->pwightMat = (IMP_F32*)workBufTemp;
	workBufTemp += ngx_align(searchLen * sizeof(IMP_F32), aligSize);

	//workBuf中划分出内存给nltvHandle->pgaussinaKernel使用，用来存储高斯核矩阵
	nltvHandle->pgaussinaKernel = (IMP_F32*)workBufTemp;
	workBufTemp += ngx_align(nearLen * sizeof(IMP_F32), aligSize);

	//workBuf中划分出内存给nltvHandle->workBuf使用，用来存储边缘拓展后的矩阵
	nltvHandle->workBuf = workBufTemp;


}

void NltvDenoisingPerpare(IMP_U08* srcImg,NLTVHandle* nltvHandle, NLTVParam* param, IMP_U08* workBuf)
{
	IMP_S32 bufSize1 = 0;
	IMP_S32 padSize = param->sRadius + param->nRadius;
	IMP_S32 step = ngx_align(param->width + 2 * padSize, aligSize);

	//计算算法所需内存大小
	NltvGetMemSize(param->sRadius, param->nRadius, param->width, param->height, &bufSize1);

	//申请内存
	workBuf = (IMP_U08 *)malloc(bufSize1*sizeof(IMP_U08));
	if (workBuf == NULL)
	{
		printf("申请内存失败，请确定可用内存是否充足!\n");
		printf("按任意键退出程序!\n");
		getchar();
		exit(0);
	}
	else
	{
		printf("需要内存：%d KB\n",bufSize1>>10);
	}


	//划分内存
	NltvAllocMem(workBuf, nltvHandle, param->sRadius, param->nRadius, param->width, param->height);

	//图像边缘镜像拓展
	BoundaryExpansion(srcImg, nltvHandle->workBuf, padSize, param->height, param->width, step);

	//生成高斯核
	gaussianKernal(2 * param->nRadius + 1, param->deviation, nltvHandle->pgaussinaKernel);

}

void NltvDenoisingProcess(NLTVHandle* nltvhandle, NLTVParam* nltvParam, IMP_U08* srcImg, IMP_U08* dstImg)
{
	//变量申明
	IMP_S32 i, j, k;
	IMP_S32 width, height;
	IMP_F32 weightedSum, sumWeight;
	IMP_F32 lamda, theta;
	IMP_F32 *pwightMat;
	IMP_U08 *pnearMat;
	IMP_U08 *psearchMat;
	IMP_U08 *pnearMatOfsearchPoint;
	IMP_S32 nRadius;
	IMP_S32 sRadius;
	IMP_S32 fParam;
	IMP_S32 newWidth, newHeight, padSize;
	IMP_U08 *pTempimg = NULL;
	IMP_U08 *pTempresult = NULL;
	IMP_S32 searchLen; 
	IMP_S32 step;
	IMP_F32* pgaussianMat=NULL;

	//变量赋值
	padSize = nltvParam->sRadius + nltvParam->nRadius;
	width   = nltvParam->width;
	height  = nltvParam->height;
	lamda   = nltvParam->lamda;
	theta   = nltvParam->theta;
	nRadius = nltvParam->nRadius;
	sRadius = nltvParam->sRadius;
    fParam  = nltvParam->fParam;
	newWidth  = width + 2 * padSize;
	newHeight = height + 2 * padSize;
	searchLen = (2 * sRadius + 1)*(2 * sRadius + 1);

	
	pwightMat  = nltvhandle->pwightMat;
	pnearMat   = nltvhandle->pnearMat;
	psearchMat = nltvhandle->psearchMat;
	pnearMatOfsearchPoint = nltvhandle->pnearMatOfsearchPoint;
	pgaussianMat = nltvhandle->pgaussinaKernel;

	step = ngx_align(newWidth, aligSize);

	pTempresult = dstImg;
	pTempimg = srcImg + padSize*step;

	for (i = padSize; i < newHeight - padSize; i++)
	{
		for (j = padSize; j < newWidth - padSize; j++)
		{
			weightedSum = 0;
			//获取当前点邻域
			Neighborhood(pTempimg, j, nRadius, step, pnearMat);
			//获取所有搜索点的邻域
			NearMatofSearchPoint(pTempimg, j, sRadius, nRadius, step, pnearMatOfsearchPoint);
			//计算权重
			weigth(pnearMat, pnearMatOfsearchPoint, pgaussianMat, fParam, sRadius, nRadius,pwightMat, &sumWeight);
			//获取当前点的所有搜索点
			searchPoint(pTempimg, j, sRadius, step, psearchMat);
			//计算新的像素值
			computNewValue(pwightMat,psearchMat,pTempimg,lamda,theta,sumWeight,sRadius,nRadius,pTempresult,j);

		}
		//指针移到下一行
		pTempimg += step;
		pTempresult += width;
	}
}

void BoundaryExpansion(IMP_U08 *input, IMP_U08 *output, IMP_S32 ExpansionSize, IMP_S32 heigth, IMP_S32 width, IMP_S32 step)
{
	/*
	将输入的矩阵进行边界镜像拓展。假设拓展的大小是2

	拓展前： 

	 a1 b1 c1 d1 e1
	 a2 b2 c2 d2 e2
	 a3 b3 c3 d3 e3
	 a4 b4 c4 d4 e4
	 a5 b5 c5 d5 e5

	首先进行方向的拓展，如下：

	 a2 b2 c2 d2 e2
	 a1 b1 c1 d1 e1
	 a1 b1 c1 d1 e1
	 a2 b2 c2 d2 e2
	 a3 b3 c3 d3 e3
	 a4 b4 c4 d4 e4
	 a5 b5 c5 d5 e5
	 a5 b5 c5 d5 e5
	 a4 b4 c4 d4 e4

	 接着，在行方向拓展的基础上进行列方向拓展。

	 b2 a2 a2 b2 c2 d2 e2 e2 d2
	 b1 a1 a1 b1 c1 d1 e1 e1 d1
	 b1 a1 a1 b1 c1 d1 e1 e1 d1
	 b2 a2 a2 b2 c2 d2 e2 e2 d2
	 b3 a3 a3 b3 c3 d3 e3 e3 d3
	 b4 a4 a4 b4 c4 d4 e4 e4 d4
	 b5 a5 a5 b5 c5 d5 e5 e5 d5
	 b5 a5 a5 b5 c5 d5 e5 e5 d5
	 b4 a4 a4 b4 c4 d4 e4 e4 d4

	*/

	IMP_S32 i, j;
	IMP_S32 outputWidth = width + 2 * ExpansionSize;
	IMP_S32 outputHeight = heigth + 2 * ExpansionSize;
	IMP_S32 offset;

	IMP_U08 *outputTmp = output;

	//拓展行方向
	for (i = 0; i < outputHeight; i++)
	{
		for (j = 0; j < outputWidth; j++)
		{
			if (i >= ExpansionSize && i < outputHeight - ExpansionSize &&
				j >= ExpansionSize && j < outputWidth - ExpansionSize)
			{
				offset = (i - ExpansionSize)*width + (j - ExpansionSize);
				outputTmp[j] = input[offset];
			}
			else if (i < ExpansionSize&&j >= ExpansionSize && j < outputWidth - ExpansionSize)
			{
				offset = (ExpansionSize - i - 1) * width + (j - ExpansionSize);
				outputTmp[j] = input[offset];
			}
			else if (i > ExpansionSize + heigth - 1 && j >= ExpansionSize && j < outputWidth - ExpansionSize)
			{
				offset = (2 * heigth + ExpansionSize - 1 - i) * width + (j - ExpansionSize);
				outputTmp[j] = input[offset];
			}
		}
		outputTmp += step;
	}


	//拓展列方向
	outputTmp = output;
	for (i = 0; i < outputHeight; i++)
	{
		for (j = 0; j < outputWidth; j++)
		{
			if (j < ExpansionSize)
			{
				offset = i * step + 2 * ExpansionSize - 1 - j;
				outputTmp[j] = output[offset];
			}
			else if (j > ExpansionSize + width - 1)
			{
				offset = i * step + 2 * (ExpansionSize + width) - 1 - j;
				outputTmp[j] = output[offset];
			}
		}
		outputTmp += step;
	}
}

void NltvDenoising(IMP_U08* srcImg, IMP_U08* dstImg, NLTVHandle* nltvHandle, NLTVParam* param)
{
	//申明工作内存指针
	IMP_U08 *workBuf = NULL;
	//初始化所需工作内存的大小
	IMP_S32 bufSize1 = 0;
	//计算图像边界拓展的大小，为sRadius+nRadius。
	IMP_S32 padSize = param->sRadius + param->nRadius;

	//前期准备
	NltvDenoisingPerpare(srcImg,nltvHandle, param, workBuf);

	//去噪
	NltvDenoisingProcess(nltvHandle, param, nltvHandle->workBuf, dstImg);

}

IMP_F32 psnr(IMP_MAT I1, IMP_MAT I2)
{
	IMP_S32 Size = I1.rows;
	I1.convertTo(I1, CV_32FC1);
	I2.convertTo(I2, CV_32FC1);
	IMP_F32 Mse = 0;
	IMP_F32 PSNR;
	IMP_MAT ii;
	ii = I1 - I2;
	for (IMP_S32 i = 0; i < Size; i++)
	{
		IMP_F32 *p = ii.ptr<IMP_F32>(i);
		for (IMP_S32 j = 0; j < Size; j++)
		{
			p[j] = p[j] * p[j];
			Mse = Mse + p[j];
		}
	}
	Mse = Mse / (Size * Size);
	PSNR = 10.0 * log10((255 * 255) / Mse);
	return PSNR;
}

float ssim(Mat i1, Mat i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32FC1;
	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I1_2 = I1.mul(I1);
	Mat I2_2 = I2.mul(I2);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigam2_2, sigam12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
	sigam2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
	sigam12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigam12 + C2;
	t3 = t1.mul(t2);

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigam2_2 + C2;
	t1 = t1.mul(t2);

	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);

	float SSIM = (mssim.val[0] + mssim.val[1] + mssim.val[2]);

	return SSIM;
}

void computNewValue(IMP_F32 *pweightMat,
					IMP_U08 *psearchMat,
					IMP_U08 *src,
					IMP_F32 lamda,
					IMP_F32 theta,
					IMP_F32 sumWeight,
					IMP_S32 sRadius,
					IMP_S32 nRadius,
					IMP_U08 *dst,
					IMP_S32 colIdx)
{
	/*
	利用权重、当前点的搜索点来加权求和出新的像素值
	*/
	IMP_S32 k;
	IMP_S32 searchLen = (2 * sRadius + 1) * (2 * sRadius + 1);
	IMP_F32 weightedSum = 0;
	IMP_S32 padSize = sRadius + nRadius;
	for (k = 0; k < searchLen; k++)
	{
		weightedSum += pweightMat[k] * psearchMat[k];
		
	}
	dst[colIdx - padSize] = (IMP_U08)((lamda*src[colIdx] + theta*(weightedSum - src[colIdx]))
		                               / (lamda + theta*(sumWeight - 1)));

}