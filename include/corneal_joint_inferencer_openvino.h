#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

// OpenVINO Runtime 头文件
#include <openvino/openvino.hpp>

/**
 * 瞳孔检测结果结构体
 */
struct PupilDetectionResult {
    cv::Point2f center;         // 瞳孔中心坐标（原图坐标系）
    float half_width;           // 检测框半宽
    cv::Mat cropped_image;      // 以瞳孔中心裁剪的 640x640 图像
    float confidence;           // 检测置信度
    cv::Rect box;               // 原始检测框
};

/**
 * 光斑检测结果结构体
 */
struct SpotDetectionResult {
    cv::Rect box;               // 检测框 [x, y, w, h]
    float score;                // 置信度
    cv::Point2f keypoint;       // 光斑中心关键点
    float kpt_conf;             // 关键点置信度
    float aspect_ratio;         // 长宽比 (w/h)
};

/**
 * 椭圆拟合结果结构体
 */
struct EllipseFitResult {
    cv::Point2f center;         // 椭圆中心
    float major_axis;           // 长轴长度
    float minor_axis;           // 短轴长度
    float angle;                // 旋转角度（度）
    bool valid;                 // 是否拟合成功
};

/**
 * 光斑分析结果结构体
 */
struct CornealSpotAnalysisResult {
    int num_spots;              // 筛选后的光斑数量
    cv::Point2f geometric_center; // 所有光斑的几何中心
    float avg_aspect_ratio;     // 平均长宽比
    
    // 内环椭圆拟合结果
    EllipseFitResult inner_ellipse;
    std::vector<SpotDetectionResult> inner_spots;
    
    // 外环椭圆拟合结果
    EllipseFitResult outer_ellipse;
    std::vector<SpotDetectionResult> outer_spots;
    
    // 所有有效光斑
    std::vector<SpotDetectionResult> all_spots;
    
    bool valid;                 // 整体分析是否有效
    std::string error_message;  // 错误信息
};

/**
 * 瞳孔检测器（单例模式，基于 OpenVINO）
 * 用于检测瞳孔位置并裁剪出 640x640 的区域
 */
class PupilDetector {
public:
    // 获取单例实例
    static PupilDetector* GetInstance();
    
    // 初始化模型（model_path 应指向 .xml 文件）
    bool Initialize(const std::string& model_path, int input_size = 640);
    
    // 禁用拷贝和赋值
    PupilDetector(const PupilDetector&);
    PupilDetector& operator=(const PupilDetector&);

    /**
     * 检测瞳孔并裁剪图像
     * @param image 输入图像（BGR格式，uint8）
     * @param result 输出瞳孔检测结果
     * @param conf_threshold 置信度阈值
     * @param nms_threshold NMS阈值
     * @return 是否检测成功
     */
    bool DetectPupil(const cv::Mat& image,
                     PupilDetectionResult& result,
                     float conf_threshold = 0.25f,
                     float nms_threshold = 0.45f);

    /**
     * 绘制可视化结果
     * @param image 原始图像
     * @param result 检测结果
     * @return 绘制了检测框的图像
     */
    cv::Mat DrawResult(const cv::Mat& image, const PupilDetectionResult& result);

private:
    // 私有构造函数
    PupilDetector();
    ~PupilDetector();
    
    // 预处理：Letterbox Resize（返回缩放参数）
    cv::Mat Preprocess(const cv::Mat& img, float& scale, int& left, int& top);
    
    // 后处理：解析输出，NMS
    std::vector<cv::Rect> Postprocess(const float* output,
                                      const cv::Size& img_size,
                                      float scale, int left, int top,
                                      float conf_threshold,
                                      float nms_threshold,
                                      std::vector<float>& scores);
    
    // 以检测框中心裁剪 640x640 图像
    cv::Mat CropCenterRegion(const cv::Mat& image, const cv::Point2f& center);

private:
    static PupilDetector* instance_;
    
    // OpenVINO 核心组件
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    
    std::string model_path_;
    int input_size_;
    bool initialized_;
    
    std::string input_name_;
    std::string output_name_;
};

/**
 * 角膜光斑检测器（单例模式，基于 OpenVINO）
 * 用于检测光斑位置、关键点，并进行椭圆拟合分析
 */
class CornealSpotDetector {
public:
    // 获取单例实例
    static CornealSpotDetector* GetInstance();
    
    // 初始化模型（model_path 应指向 .xml 文件）
    bool Initialize(const std::string& model_path, int input_size = 640);
    
    // 禁用拷贝和赋值
    CornealSpotDetector(const CornealSpotDetector&);
    CornealSpotDetector& operator=(const CornealSpotDetector&);

    /**
     * 检测光斑并进行分析
     * @param image 输入图像（640x640，从瞳孔中心裁剪）
     * @param result 输出光斑分析结果
     * @param conf_threshold 置信度阈值
     * @param nms_threshold NMS阈值
     * @param min_spots 最小光斑数量（默认 20）
     * @param max_spots 最大光斑数量（默认 35）
     * @param min_aspect_ratio 最小长宽比（默认 0.5）
     * @param max_aspect_ratio 最大长宽比（默认 2.0）
     * @param enable_ellipse_fit 是否进行椭圆拟合（默认 0，不拟合；1 时进行拟合）
     * @return 是否检测成功
     */
    bool DetectAndAnalyze(const cv::Mat& image,
                         CornealSpotAnalysisResult& result,
                         float conf_threshold = 0.25f,
                         float nms_threshold = 0.45f,
                         int min_spots = 20,
                         int max_spots = 35,
                         float min_aspect_ratio = 0.5f,
                         float max_aspect_ratio = 2.0f,
                         int enable_ellipse_fit = 0);

    /**
     * 绘制完整可视化结果（包含椭圆拟合）
     * @param image 输入图像
     * @param result 分析结果
     * @return 绘制了检测框、关键点和椭圆的图像
     */
    cv::Mat DrawResult(const cv::Mat& image, const CornealSpotAnalysisResult& result);
    
    /**
     * 绘制简化可视化结果（仅光斑和几何中心）
     * @param image 输入图像
     * @param result 分析结果
     * @return 绘制了光斑框、长宽比标注和几何中心的图像
     */
    cv::Mat DrawSimpleResult(const cv::Mat& image, const CornealSpotAnalysisResult& result);

private:
    // 私有构造函数
    CornealSpotDetector();
    ~CornealSpotDetector();
    
    // 预处理
    cv::Mat Preprocess(const cv::Mat& img, float& scale, int& left, int& top);
    
    // 后处理：解析 YOLOv8-pose 输出
    std::vector<SpotDetectionResult> Postprocess(const float* output,
                                                  const cv::Size& img_size,
                                                  float scale, int left, int top,
                                                  float conf_threshold,
                                                  float nms_threshold);
    
    // 筛选光斑：长宽比、离群值
    std::vector<SpotDetectionResult> FilterSpots(const std::vector<SpotDetectionResult>& spots,
                                                  float min_aspect_ratio,
                                                  float max_aspect_ratio);
    
    // 离群值检测（基于距离）
    std::vector<SpotDetectionResult> RemoveOutliers(const std::vector<SpotDetectionResult>& spots);
    
    // 计算几何中心
    cv::Point2f ComputeGeometricCenter(const std::vector<SpotDetectionResult>& spots);
    
    // 计算平均长宽比
    float ComputeAverageAspectRatio(const std::vector<SpotDetectionResult>& spots);
    
    // 分离内外环光斑
    void SeparateInnerOuter(const std::vector<SpotDetectionResult>& spots,
                           const cv::Point2f& center,
                           std::vector<SpotDetectionResult>& inner,
                           std::vector<SpotDetectionResult>& outer);
    
    // 椭圆拟合
    EllipseFitResult FitEllipse(const std::vector<SpotDetectionResult>& spots);

private:
    static CornealSpotDetector* instance_;
    
    // OpenVINO 核心组件
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    
    std::string model_path_;
    int input_size_;
    bool initialized_;
    
    std::string input_name_;
    std::string output_name_;
};

