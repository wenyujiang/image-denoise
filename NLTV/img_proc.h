/*******************************************
// 作者：蒋文宇
// 时间：2018年3月
// 版权：浙江工业大学信息学院
// e-mail： jwydyx@yeah.net
//
// 功能：NLTV图像去噪
*********************************************/

#include <opencv2/opencv.hpp>
using namespace cv;

#define aligSize 4
#define ngx_align(x, aligSize) ((x + aligSize - 1) / aligSize*aligSize)



typedef unsigned char IMP_U08;
typedef int           IMP_S32;
typedef float         IMP_F32;
typedef double        IMP_F64;
typedef Mat           IMP_MAT;

struct  NLTVHandle
{
	IMP_F32 *pwightMat;					//权重
	IMP_F32 *pgaussinaKernel;			//高斯核
	IMP_U08 *pnearMat;					//当前点的邻域
	IMP_U08 *psearchMat;				//搜索点
	IMP_U08 *pnearMatOfsearchPoint;     //搜索点的邻域
	IMP_U08 *workBuf;                   //边缘拓展后的图像
};

struct  NLTVParam
{
	IMP_F32 theta;      //平衡参数
	IMP_F32 deviation;  //高斯核标准差
	IMP_F32 lamda;      //平衡参数
	IMP_S32 nRadius;    //邻域半径
	IMP_S32 sRadius;    //搜索半径
	IMP_S32 fParam;     //滤波参数
	IMP_S32 width;      //图像的宽
	IMP_S32 height;     //图像的高
};

/****************************************/
//      Neighborhood：求当前点的邻域矩阵
//
//      curH      -I      当前点所在行的首地址指针
//      curW      -I      当前点所在列
//      nRadius   -I      当前点的邻域半径，邻域矩阵的大小为(2*n+1)*(2*n+1)
//      step      -I      图像的列数（对齐）
//      nearMat   -O      指向当前点邻域矩阵的指针,邻域矩阵大小为(2*n+1)*(2*n+1)
//
/****************************************/
void Neighborhood(IMP_U08 *curH, IMP_S32 curW, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat);

/****************************************/
//      NearMatofSearchPoint：求搜索点的邻域矩阵( 当前点的搜索区域内的所有点（包括当前点本身）称为搜索点 )
//
//      curH         -I      当前点所在行的首地址指针
//      curW         -I      当前点所在列
//      sRadius      -I      当前点的搜索半径，搜索半径的大小为 (2*m+1)*(2*m+1)
//      nRadius     - I      当前点的邻域半径，邻域矩阵的大小为 (2*n+1)*(2*n+1)
//      step         -I      图像的列数（对齐）
//      nearMat      -O      指向搜索点的邻域矩阵的指针, 该矩阵大小为 [(2*m+1)*(2*m+1)] * [(2*n+1)*(2*n+1)]
//
/****************************************/
void NearMatofSearchPoint(IMP_U08 *curH, IMP_S32 curW, IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat);

/****************************************/
//      weigth：求出当前点与搜索点的相似性权重
//
//      nearOfCurP     -I     当前点的邻域矩阵
//      nearOfSeaP     -I     搜索点的邻域矩阵
//      pgaussian      -I     高斯加权矩阵
//      h              -I     参数
//      sRadius        -I     搜索区域半径
//      nRadius        -I     邻域区域半径
//      pweight        -O     相似性权重矩阵
//      sumweight      -O     相似性权重之和
//
/****************************************/
void weigth(IMP_U08 *nearOfCurP, IMP_U08 *nearOfSeaP, IMP_F32 *pgaussian, IMP_F32 h,
	        IMP_S32 sRadius,     IMP_S32 nRadius,     IMP_F32 *pweight,   IMP_F32 *sumweight);

/****************************************/
//      searchPoint：求当前点的搜索点
//
//      p         -I      当前点所在行的首地址指针
//      w         -I      当前点所在列
//      n         -I      当前点的搜索半径，大小为(2*n+1)*(2*n+1)
//      width     -I      图像的列数
//      nearMat   -O      指向当前点搜索点矩阵的指针,大小为(2*n+1)*(2*n+1)
//
/****************************************/
void searchPoint(IMP_F32 *p, IMP_S32 w, IMP_S32 n, IMP_S32 width, IMP_F32 *nearMat);

/****************************************/
//      gaussianKernal：求高斯核
//
//      kernelSize    -I     高斯核大小
//      sigma         -I     高斯核标准差
//      kernel        -O     高斯核
//
/****************************************/
void gaussianKernal(IMP_S32 kernelSize, IMP_F32 sigma, IMP_F32 *kernel);

/****************************************/
//      NltvDenoisingPerpare：nltv去噪准备工作
//
//      srcImg        -I     输入图像
//      nltvHandle    -I     nltv结构体
//      param         -I     nltv参数
//      workBuf       -I/O   工作内存
/****************************************/
void NltvDenoisingPerpare(IMP_U08* srcImg,NLTVHandle* nltvHandle, NLTVParam* param, IMP_U08* workBuf);


/****************************************/
//      NltvDenoisingProcess：nltv去噪处理
//
//      nltvHandle    -I     nltv结构体
//      param         -I     nltv参数
//      inPut         -I     输入图像
//      outPut        -O     输出图像
//      step          -I     内存对齐宽
/****************************************/
void NltvDenoisingProcess(NLTVHandle* nltvHandle, NLTVParam* param, IMP_U08* inPut, IMP_U08* outPut, IMP_S32 step);


/****************************************/
//      NltvGetMemSize：计算所需内存大小
//
//      sRadius         -I      搜索半径
//      nRadius         -I      邻域半径
//		width			-I      图像宽
//		height          -I      图像高
//		bufSize         -O      所需内存大小
/****************************************/
void NltvGetMemSize(IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 width, IMP_S32 height, IMP_S32* bufSize);


/****************************************/
//      NltvAllocMem：划分内存
//
//      workBuf         -I      待划分内存
//      nltvHandle      -O      nltv结构体
//		sRadius			-I      搜索半径
//		nRadius         -I      邻域半径
//		width           -I      图像宽
//      height          -I      图像高
/****************************************/
void NltvAllocMem(IMP_U08* workBuf, NLTVHandle* nltvHandle, IMP_S32 sRadius, IMP_S32 nRadius,IMP_S32 width, IMP_S32 height);


/****************************************/
//      BoundaryExpansion：图像边缘镜像拓展
//
//      input            -I      输入图像
//      output           -0      输出图像
//		ExpansionSize	 -I      拓展大小
//		heigth           -I      输入图像高
//		width            -I      输入图像宽
//      step             -I      内存对齐后的宽
/****************************************/
void BoundaryExpansion(IMP_U08 *input, IMP_U08 *output, IMP_S32 ExpansionSize, IMP_S32 heigth, IMP_S32 width, IMP_S32 step);


/****************************************/
//      NltvDenoising：nltv去噪
//
//      srcImg          -I      输入图像
//      dstImg          -0      输出图像
//		nltvHandle	    -I      nltv结构体
//		param           -I      nltv参数
/****************************************/
void NltvDenoising(IMP_U08* srcImg, IMP_U08* dstImg, NLTVHandle* nltvHandle, NLTVParam* param);


/****************************************/
//      psnr：计算两幅图像之间的psnr，返回类型为IMP_F32
//
//      I1         -I      图像1
//      I2         -I      图像2
/****************************************/
IMP_F32 psnr(IMP_MAT I1, IMP_MAT I2);


/****************************************************
//ssim: 计算图1和图2的SSIM
//
//参数
i1           -I      图像1
i2           -I      图像2
//返回值：ssim
//
*****************************************************/
float ssim(Mat i1, Mat  i2);


/*******************************************
//computNewValue：：计算新的像素值
//参数：
//		pweightMat   -I     weight
//		psearchMat   -I     搜索区域
//		src          -I     输入图像
//		lamda		 -I     lamda
//		theta	     -I     theta
//		sRadius      -I     搜索半径
//		nRadius      -I     邻域半径
//		dst			 -0     输出图像
//		colIdx       -I     输入点所在列索引
*******************************************/
void computNewValue(IMP_F32 *pweightMat,
					IMP_U08 *psearchMat,
					IMP_U08 *src,
					IMP_F32 lamda,
					IMP_F32 theta,
					IMP_F32 sumWeight,
					IMP_S32 sRadius,
					IMP_S32 nRadius,
					IMP_U08 *dst,
					IMP_S32 colIdx);
