/*******************************************
// ���ߣ�������
// ʱ�䣺2018��3��
// ��Ȩ���㽭��ҵ��ѧ��ϢѧԺ
// e-mail�� jwydyx@yeah.net
//
// ���ܣ�NLTVͼ��ȥ��
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
	IMP_F32 *pwightMat;					//Ȩ��
	IMP_F32 *pgaussinaKernel;			//��˹��
	IMP_U08 *pnearMat;					//��ǰ�������
	IMP_U08 *psearchMat;				//������
	IMP_U08 *pnearMatOfsearchPoint;     //�����������
	IMP_U08 *workBuf;                   //��Ե��չ���ͼ��
};

struct  NLTVParam
{
	IMP_F32 theta;      //ƽ�����
	IMP_F32 deviation;  //��˹�˱�׼��
	IMP_F32 lamda;      //ƽ�����
	IMP_S32 nRadius;    //����뾶
	IMP_S32 sRadius;    //�����뾶
	IMP_S32 fParam;     //�˲�����
	IMP_S32 width;      //ͼ��Ŀ�
	IMP_S32 height;     //ͼ��ĸ�
};

/****************************************/
//      Neighborhood����ǰ����������
//
//      curH      -I      ��ǰ�������е��׵�ַָ��
//      curW      -I      ��ǰ��������
//      nRadius   -I      ��ǰ�������뾶���������Ĵ�СΪ(2*n+1)*(2*n+1)
//      step      -I      ͼ������������룩
//      nearMat   -O      ָ��ǰ����������ָ��,��������СΪ(2*n+1)*(2*n+1)
//
/****************************************/
void Neighborhood(IMP_U08 *curH, IMP_S32 curW, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat);

/****************************************/
//      NearMatofSearchPoint������������������( ��ǰ������������ڵ����е㣨������ǰ�㱾����Ϊ������ )
//
//      curH         -I      ��ǰ�������е��׵�ַָ��
//      curW         -I      ��ǰ��������
//      sRadius      -I      ��ǰ��������뾶�������뾶�Ĵ�СΪ (2*m+1)*(2*m+1)
//      nRadius     - I      ��ǰ�������뾶���������Ĵ�СΪ (2*n+1)*(2*n+1)
//      step         -I      ͼ������������룩
//      nearMat      -O      ָ�����������������ָ��, �þ����СΪ [(2*m+1)*(2*m+1)] * [(2*n+1)*(2*n+1)]
//
/****************************************/
void NearMatofSearchPoint(IMP_U08 *curH, IMP_S32 curW, IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 step, IMP_U08 *nearMat);

/****************************************/
//      weigth�������ǰ�����������������Ȩ��
//
//      nearOfCurP     -I     ��ǰ����������
//      nearOfSeaP     -I     ��������������
//      pgaussian      -I     ��˹��Ȩ����
//      h              -I     ����
//      sRadius        -I     ��������뾶
//      nRadius        -I     ��������뾶
//      pweight        -O     ������Ȩ�ؾ���
//      sumweight      -O     ������Ȩ��֮��
//
/****************************************/
void weigth(IMP_U08 *nearOfCurP, IMP_U08 *nearOfSeaP, IMP_F32 *pgaussian, IMP_F32 h,
	        IMP_S32 sRadius,     IMP_S32 nRadius,     IMP_F32 *pweight,   IMP_F32 *sumweight);

/****************************************/
//      searchPoint����ǰ���������
//
//      p         -I      ��ǰ�������е��׵�ַָ��
//      w         -I      ��ǰ��������
//      n         -I      ��ǰ��������뾶����СΪ(2*n+1)*(2*n+1)
//      width     -I      ͼ�������
//      nearMat   -O      ָ��ǰ������������ָ��,��СΪ(2*n+1)*(2*n+1)
//
/****************************************/
void searchPoint(IMP_F32 *p, IMP_S32 w, IMP_S32 n, IMP_S32 width, IMP_F32 *nearMat);

/****************************************/
//      gaussianKernal�����˹��
//
//      kernelSize    -I     ��˹�˴�С
//      sigma         -I     ��˹�˱�׼��
//      kernel        -O     ��˹��
//
/****************************************/
void gaussianKernal(IMP_S32 kernelSize, IMP_F32 sigma, IMP_F32 *kernel);

/****************************************/
//      NltvDenoisingPerpare��nltvȥ��׼������
//
//      srcImg        -I     ����ͼ��
//      nltvHandle    -I     nltv�ṹ��
//      param         -I     nltv����
//      workBuf       -I/O   �����ڴ�
/****************************************/
void NltvDenoisingPerpare(IMP_U08* srcImg,NLTVHandle* nltvHandle, NLTVParam* param, IMP_U08* workBuf);


/****************************************/
//      NltvDenoisingProcess��nltvȥ�봦��
//
//      nltvHandle    -I     nltv�ṹ��
//      param         -I     nltv����
//      inPut         -I     ����ͼ��
//      outPut        -O     ���ͼ��
//      step          -I     �ڴ�����
/****************************************/
void NltvDenoisingProcess(NLTVHandle* nltvHandle, NLTVParam* param, IMP_U08* inPut, IMP_U08* outPut, IMP_S32 step);


/****************************************/
//      NltvGetMemSize�����������ڴ��С
//
//      sRadius         -I      �����뾶
//      nRadius         -I      ����뾶
//		width			-I      ͼ���
//		height          -I      ͼ���
//		bufSize         -O      �����ڴ��С
/****************************************/
void NltvGetMemSize(IMP_S32 sRadius, IMP_S32 nRadius, IMP_S32 width, IMP_S32 height, IMP_S32* bufSize);


/****************************************/
//      NltvAllocMem�������ڴ�
//
//      workBuf         -I      �������ڴ�
//      nltvHandle      -O      nltv�ṹ��
//		sRadius			-I      �����뾶
//		nRadius         -I      ����뾶
//		width           -I      ͼ���
//      height          -I      ͼ���
/****************************************/
void NltvAllocMem(IMP_U08* workBuf, NLTVHandle* nltvHandle, IMP_S32 sRadius, IMP_S32 nRadius,IMP_S32 width, IMP_S32 height);


/****************************************/
//      BoundaryExpansion��ͼ���Ե������չ
//
//      input            -I      ����ͼ��
//      output           -0      ���ͼ��
//		ExpansionSize	 -I      ��չ��С
//		heigth           -I      ����ͼ���
//		width            -I      ����ͼ���
//      step             -I      �ڴ�����Ŀ�
/****************************************/
void BoundaryExpansion(IMP_U08 *input, IMP_U08 *output, IMP_S32 ExpansionSize, IMP_S32 heigth, IMP_S32 width, IMP_S32 step);


/****************************************/
//      NltvDenoising��nltvȥ��
//
//      srcImg          -I      ����ͼ��
//      dstImg          -0      ���ͼ��
//		nltvHandle	    -I      nltv�ṹ��
//		param           -I      nltv����
/****************************************/
void NltvDenoising(IMP_U08* srcImg, IMP_U08* dstImg, NLTVHandle* nltvHandle, NLTVParam* param);


/****************************************/
//      psnr����������ͼ��֮���psnr����������ΪIMP_F32
//
//      I1         -I      ͼ��1
//      I2         -I      ͼ��2
/****************************************/
IMP_F32 psnr(IMP_MAT I1, IMP_MAT I2);


/****************************************************
//ssim: ����ͼ1��ͼ2��SSIM
//
//����
i1           -I      ͼ��1
i2           -I      ͼ��2
//����ֵ��ssim
//
*****************************************************/
float ssim(Mat i1, Mat  i2);


/*******************************************
//computNewValue���������µ�����ֵ
//������
//		pweightMat   -I     weight
//		psearchMat   -I     ��������
//		src          -I     ����ͼ��
//		lamda		 -I     lamda
//		theta	     -I     theta
//		sRadius      -I     �����뾶
//		nRadius      -I     ����뾶
//		dst			 -0     ���ͼ��
//		colIdx       -I     ���������������
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
