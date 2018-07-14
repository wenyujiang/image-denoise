#include"img_proc.h"


void Neighborhood(IMP_U08 *curH, IMP_S32 curW, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat)
{

	IMP_S32 i, j;
	IMP_S32 wTemp;
	IMP_U08 *pTem;
	IMP_S32 count = 0;

	for (i = -nRadius; i <= nRadius; i++)
	{
		//��ָ���ƶ�������������еĵ�һ��Ԫ�صĵ�ַ
		pTem = curH + step * i;
		for (j = -nRadius; j <= nRadius; j++)
		{
			//������ڵ�����һ�еĵڼ���Ԫ��
			wTemp = curW + j;
			//ȡ������ڵ㣬������nearMat�С�
			nearMat[count] = pTem[wTemp];
			count++;
		}
	}

}

void NearMatofSearchPoint(IMP_U08 *curH, IMP_S32 curW, IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat)
{
	/*
	�ҵ����������ڵ����е���ڵ㣬�������Ǳ���������
    
	���赱ǰ����c3�������뾶��1������뾶��1��ͼ��������£�

	a1 a2 a3 a4 a5
	b1 b2 b3 b4 b5
	c1 c2 c3 c4 c5
	d1 d2 d3 d4 d5
	e1 e2 e3 e4 e5

	��c3����������b2 b3 b4 c2 c3 c4 d2 d3 d4

	���ԣ�
	     b2���������a1 a2 a3 b1 b2 b3 c1 c2 c3
	     b3���������a2 a3 a4 b2 b3 b4 c2 c3 c4
	...
		 d3���������c2 c3 c4 d2 d3 d4 e2 e3 e4
		 d4���������c3 c4 c5 d3 d4 d5 e3 e4 e5

	����Щ�������������������������a1 a2 a3 b1 b2 b3 c1 c2 c3��a2 a3 a4 b2 b3 b4 c2 c3 c4��...��c3 c4 c5 d3 d4 d5 e3 e4 e5
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
	����Ȩ�أ�
	��֮ǰ�õ��������������һ��һ���ȡ�������뵱ǰ���������зǾֲ����ƶ�Ȩ�ؼ��㡣
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

	//���ɶ�ά��˹��
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

	usedSize += ngx_align(nearLen   * sizeof(IMP_U08), aligSize);			   //�ۼ������ڴ��С,����
	usedSize += ngx_align(searchLen * sizeof(IMP_U08), aligSize);              //�ۼ����������ڴ��С,����
	usedSize += ngx_align(nearLen   * sizeof(IMP_U08), aligSize) * searchLen;  //�ۼ���������������ڴ��С,����
	usedSize += ngx_align(searchLen * sizeof(IMP_F32), aligSize);			   //�ۼ�weight�ڴ��С,����
	usedSize += ngx_align(nearLen   * sizeof(IMP_F32), aligSize);              //�ۼӸ�˹�˴�С�ڴ�,����
	usedSize += ngx_align(width + 2 * cpySize, aligSize) * (height + 2 * cpySize) * sizeof(IMP_U08); //��Ե��չ���ͼ���ڴ棬����

	*bufSize = usedSize;
}

void NltvAllocMem(IMP_U08* workBuf, NLTVHandle* nltvHandle, IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 width, IMP_S32 height)
{

	IMP_U08 *workBufTemp = workBuf;
	IMP_S32 cpySize = sRadius + nRadius;
	int nearLen = (2 * nRadius + 1)*(2 * nRadius + 1);
	int searchLen = (2 * sRadius + 1)*(2 * sRadius + 1);

	//workBuf�л��ֳ��ڴ��nltvHandle->pnearMatʹ��,�����洢��ǰ����������
	nltvHandle->pnearMat = workBufTemp;
	workBufTemp += ngx_align(nearLen * sizeof(IMP_U08), aligSize);

	//workBuf�л��ֳ��ڴ��nltvHandle->psearchMatʹ��,�����洢��ǰ�����������
	nltvHandle->psearchMat = workBufTemp;
	workBufTemp += ngx_align(searchLen * sizeof(IMP_U08), aligSize);

	//workBuf�л��ֳ��ڴ��nltvHandle->pnearMatOfsearchPointʹ�ã������洢�������ڵ����е���������
	nltvHandle->pnearMatOfsearchPoint = workBufTemp;
	workBufTemp += ngx_align(nearLen  *sizeof(IMP_U08), aligSize) * searchLen;

	//workBuf�л��ֳ��ڴ��nltvHandle->pwightMatʹ�ã������洢Ȩ��
	nltvHandle->pwightMat = (IMP_F32*)workBufTemp;
	workBufTemp += ngx_align(searchLen * sizeof(IMP_F32), aligSize);

	//workBuf�л��ֳ��ڴ��nltvHandle->pgaussinaKernelʹ�ã������洢��˹�˾���
	nltvHandle->pgaussinaKernel = (IMP_F32*)workBufTemp;
	workBufTemp += ngx_align(nearLen * sizeof(IMP_F32), aligSize);

	//workBuf�л��ֳ��ڴ��nltvHandle->workBufʹ�ã������洢��Ե��չ��ľ���
	nltvHandle->workBuf = workBufTemp;


}

void NltvDenoisingPerpare(IMP_U08* srcImg,NLTVHandle* nltvHandle, NLTVParam* param, IMP_U08* workBuf)
{
	IMP_S32 bufSize1 = 0;
	IMP_S32 padSize = param->sRadius + param->nRadius;
	IMP_S32 step = ngx_align(param->width + 2 * padSize, aligSize);

	//�����㷨�����ڴ��С
	NltvGetMemSize(param->sRadius, param->nRadius, param->width, param->height, &bufSize1);

	//�����ڴ�
	workBuf = (IMP_U08 *)malloc(bufSize1*sizeof(IMP_U08));
	if (workBuf == NULL)
	{
		printf("�����ڴ�ʧ�ܣ���ȷ�������ڴ��Ƿ����!\n");
		printf("��������˳�����!\n");
		getchar();
		exit(0);
	}
	else
	{
		printf("��Ҫ�ڴ棺%d KB\n",bufSize1>>10);
	}


	//�����ڴ�
	NltvAllocMem(workBuf, nltvHandle, param->sRadius, param->nRadius, param->width, param->height);

	//ͼ���Ե������չ
	BoundaryExpansion(srcImg, nltvHandle->workBuf, padSize, param->height, param->width, step);

	//���ɸ�˹��
	gaussianKernal(2 * param->nRadius + 1, param->deviation, nltvHandle->pgaussinaKernel);

}

void NltvDenoisingProcess(NLTVHandle* nltvhandle, NLTVParam* nltvParam, IMP_U08* srcImg, IMP_U08* dstImg)
{
	//��������
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

	//������ֵ
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
			//��ȡ��ǰ������
			Neighborhood(pTempimg, j, nRadius, step, pnearMat);
			//��ȡ���������������
			NearMatofSearchPoint(pTempimg, j, sRadius, nRadius, step, pnearMatOfsearchPoint);
			//����Ȩ��
			weigth(pnearMat, pnearMatOfsearchPoint, pgaussianMat, fParam, sRadius, nRadius,pwightMat, &sumWeight);
			//��ȡ��ǰ�������������
			searchPoint(pTempimg, j, sRadius, step, psearchMat);
			//�����µ�����ֵ
			computNewValue(pwightMat,psearchMat,pTempimg,lamda,theta,sumWeight,sRadius,nRadius,pTempresult,j);

		}
		//ָ���Ƶ���һ��
		pTempimg += step;
		pTempresult += width;
	}
}

void BoundaryExpansion(IMP_U08 *input, IMP_U08 *output, IMP_S32 ExpansionSize, IMP_S32 heigth, IMP_S32 width, IMP_S32 step)
{
	/*
	������ľ�����б߽羵����չ��������չ�Ĵ�С��2

	��չǰ�� 

	 a1 b1 c1 d1 e1
	 a2 b2 c2 d2 e2
	 a3 b3 c3 d3 e3
	 a4 b4 c4 d4 e4
	 a5 b5 c5 d5 e5

	���Ƚ��з������չ�����£�

	 a2 b2 c2 d2 e2
	 a1 b1 c1 d1 e1
	 a1 b1 c1 d1 e1
	 a2 b2 c2 d2 e2
	 a3 b3 c3 d3 e3
	 a4 b4 c4 d4 e4
	 a5 b5 c5 d5 e5
	 a5 b5 c5 d5 e5
	 a4 b4 c4 d4 e4

	 ���ţ����з�����չ�Ļ����Ͻ����з�����չ��

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

	//��չ�з���
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


	//��չ�з���
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
	//���������ڴ�ָ��
	IMP_U08 *workBuf = NULL;
	//��ʼ�����蹤���ڴ�Ĵ�С
	IMP_S32 bufSize1 = 0;
	//����ͼ��߽���չ�Ĵ�С��ΪsRadius+nRadius��
	IMP_S32 padSize = param->sRadius + param->nRadius;

	//ǰ��׼��
	NltvDenoisingPerpare(srcImg,nltvHandle, param, workBuf);

	//ȥ��
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
	����Ȩ�ء���ǰ�������������Ȩ��ͳ��µ�����ֵ
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