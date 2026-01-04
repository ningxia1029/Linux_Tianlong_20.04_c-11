#include "corneal_joint_inferencer_openvino.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// ============================================================================
// 辅助函数：将相对路径转换为绝对路径
// ============================================================================
static std::string resolve_path(const std::string& path) {
    if (path.empty()) {
        return path;
    }
    
    // 如果已经是绝对路径，直接返回
    if (path[0] == '/') {
        return path;
    }
    
    // 获取当前工作目录
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        // 如果获取失败，返回原路径
        return path;
    }
    
    // 构建绝对路径
    std::string abs_path = std::string(cwd) + "/" + path;
    
    // 简化路径（处理 .. 和 .）
    char resolved[PATH_MAX];
    if (realpath(abs_path.c_str(), resolved) != nullptr) {
        return std::string(resolved);
    }
    
    // 如果 realpath 失败，返回构建的绝对路径
    return abs_path;
}

// ============================================================================
// PupilDetector 实现（单例模式）
// ============================================================================

PupilDetector* PupilDetector::instance_ = NULL;

PupilDetector::PupilDetector()
    : input_size_(640),
      initialized_(false) {
}

PupilDetector::~PupilDetector() {
}

PupilDetector* PupilDetector::GetInstance() {
    if (instance_ == NULL) {
        instance_ = new PupilDetector();
    }
    return instance_;
}

bool PupilDetector::Initialize(const std::string& model_path, int input_size) {
    if (initialized_) {
        std::cout << "[PupilDetector] 已初始化，跳过重复初始化" << std::endl;
        return true;
    }
    
    // 将路径转换为绝对路径
    model_path_ = resolve_path(model_path);
    input_size_ = input_size;
    
    try {
        std::cout << "[PupilDetector] 正在加载 OpenVINO 模型..." << std::endl;
        std::cout << "[PupilDetector] 模型路径: " << model_path_ << std::endl;
        
        // 1. 读取模型
        std::shared_ptr<ov::Model> model = core_.read_model(model_path_);
        
        // 2. 配置预处理（使用 PrePostProcessor API）
        ov::preprocess::PrePostProcessor ppp(model);
        
        // 获取输入输出名称
        input_name_ = model->input().get_any_name();
        output_name_ = model->output().get_any_name();
        
        // 配置输入
        // 输入格式：cv::Mat (HWC, BGR, uint8) -> 模型期望 (NCHW, RGB, float32)
        ppp.input().tensor()
            .set_element_type(ov::element::u8)           // 输入数据类型：uint8
            .set_layout("NHWC")                          // 输入布局：NHWC (批次=1, H, W, 通道)
            .set_color_format(ov::preprocess::ColorFormat::BGR);  // 输入颜色格式：BGR
        
        // 模型期望的输入格式
        ppp.input().model()
            .set_layout("NCHW");                         // 模型布局：NCHW
        
        // 预处理步骤
        ppp.input().preprocess()
            .convert_color(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB 转换
            .convert_element_type(ov::element::f32)              // uint8 -> float32
            .scale(255.0f);                                      // 归一化：除以 255
        
        // 应用预处理配置，构建新模型
        model = ppp.build();
        
        // 3. 编译模型到 CPU 设备
        compiled_model_ = core_.compile_model(model, "CPU");
        
        // 4. 创建推理请求
        infer_request_ = compiled_model_.create_infer_request();
        
        initialized_ = true;
        
        std::cout << "[PupilDetector] 模型已加载: " << model_path_ << std::endl;
        std::cout << "  输入名: " << input_name_ << std::endl;
        std::cout << "  输出名: " << output_name_ << std::endl;
        std::cout << "  输入尺寸: [1, " << input_size_ << ", " << input_size_ << ", 3]" << std::endl;
        std::cout << "  预处理已集成到模型图中（BGR->RGB, Normalize, Layout转换）" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PupilDetector] 初始化失败: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
}

cv::Mat PupilDetector::Preprocess(const cv::Mat& img, float& scale, int& left, int& top) {
    // 固定输入尺寸：2048*1200，直接缩放处理，不用自适应
    const int fixed_input_width = 2048;
    const int fixed_input_height = 1200;
    
    int h = img.rows;
    int w = img.cols;

    // 如果输入不是2048*1200，先缩放到这个固定尺寸
    cv::Mat img_fixed;
    float scale_to_fixed = 1.0f;
    if (w != fixed_input_width || h != fixed_input_height) {
        // 计算缩放到固定尺寸的比例（保持宽高比）
        scale_to_fixed = std::min(static_cast<float>(fixed_input_width) / w,
                                 static_cast<float>(fixed_input_height) / h);
        cv::resize(img, img_fixed, cv::Size(fixed_input_width, fixed_input_height), 0, 0, cv::INTER_LINEAR);
    } else {
        img_fixed = img;
    }

    // 从固定尺寸2048*1200缩放到模型输入640x640（保持宽高比，letterbox方式）
    // 计算从2048*1200到640x640的缩放比例
    float scale_fixed_to_model = std::min(static_cast<float>(input_size_) / fixed_input_height,
                                         static_cast<float>(input_size_) / fixed_input_width);

    // 计算从原始图像到640x640的总缩放比例（用于后处理坐标还原）
    scale = scale_fixed_to_model * scale_to_fixed;

    int new_h = static_cast<int>(fixed_input_height * scale_fixed_to_model);
    int new_w = static_cast<int>(fixed_input_width * scale_fixed_to_model);

    // 缩放图像
    cv::Mat img_resized;
    cv::resize(img_fixed, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 计算填充（基于从2048*1200到640x640的缩放）
    top = (input_size_ - new_h) / 2;
    int bottom = input_size_ - new_h - top;
    left = (input_size_ - new_w) / 2;
    int right = input_size_ - new_w - left;

    // 填充灰边 (114, 114, 114)
    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 注意：不再需要手动进行 BGR->RGB 和归一化，这些已经集成到 OpenVINO 预处理中
    // 直接返回 BGR uint8 格式的图像
    return img_padded;
}

std::vector<cv::Rect> PupilDetector::Postprocess(
    const float* output,
    const cv::Size& img_size,
    float scale,
    int left,
    int top,
    float conf_threshold,
    float nms_threshold,
    std::vector<float>& scores) {

    // YOLOv8 detect 输出: [1, 5, 8400]
    // 5个通道：[cx, cy, w, h, conf]
    const int num_proposals = 8400;

    std::vector<cv::Rect> boxes;
    scores.clear();

    for (int i = 0; i < num_proposals; ++i) {
        float conf = output[4 * num_proposals + i];

        if (conf > conf_threshold) {
            float cx = output[0 * num_proposals + i];
            float cy = output[1 * num_proposals + i];
            float w = output[2 * num_proposals + i];
            float h = output[3 * num_proposals + i];

            // 坐标还原到原图
            float x1 = (cx - w / 2.0f - left) / scale;
            float y1 = (cy - h / 2.0f - top) / scale;
            float w_orig = w / scale;
            float h_orig = h / scale;

            // 限制在图像范围内
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_size.width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_size.height)));
            w_orig = std::max(0.0f, std::min(w_orig, static_cast<float>(img_size.width) - x1));
            h_orig = std::max(0.0f, std::min(h_orig, static_cast<float>(img_size.height) - y1));

            boxes.push_back(cv::Rect(static_cast<int>(x1),
                                    static_cast<int>(y1),
                                    static_cast<int>(w_orig),
                                    static_cast<int>(h_orig)));
            scores.push_back(conf);
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    for (size_t idx_i = 0; idx_i < indices.size(); ++idx_i) {
        int idx = indices[idx_i];
        nms_boxes.push_back(boxes[idx]);
        nms_scores.push_back(scores[idx]);
    }

    scores = nms_scores;
    return nms_boxes;
}

cv::Mat PupilDetector::CropCenterRegion(const cv::Mat& image, const cv::Point2f& center) {
    int h = image.rows;
    int w = image.cols;

    int target_size = 640;
    int half = target_size / 2;

    // 计算裁剪区域
    int cx = static_cast<int>(center.x);
    int cy = static_cast<int>(center.y);

    // 先对原图做必要的 padding
    int pad_left = std::max(0, half - cx);
    int pad_right = std::max(0, half - (w - cx));
    int pad_top = std::max(0, half - cy);
    int pad_bottom = std::max(0, half - (h - cy));

    cv::Mat img_padded;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0) {
        cv::copyMakeBorder(image, img_padded, pad_top, pad_bottom, pad_left, pad_right,
                          cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cx += pad_left;
        cy += pad_top;
    } else {
        img_padded = image;
    }

    // 裁剪
    int x1 = cx - half;
    int y1 = cy - half;

    // 边界修正
    x1 = std::max(0, std::min(x1, img_padded.cols - target_size));
    y1 = std::max(0, std::min(y1, img_padded.rows - target_size));

    cv::Rect crop_rect(x1, y1, std::min(target_size, img_padded.cols - x1),
                       std::min(target_size, img_padded.rows - y1));
    cv::Mat cropped = img_padded(crop_rect).clone();

    // 确保是 640x640
    if (cropped.cols != target_size || cropped.rows != target_size) {
        cv::resize(cropped, cropped, cv::Size(target_size, target_size));
    }

    return cropped;
}

bool PupilDetector::DetectPupil(const cv::Mat& image,
                                PupilDetectionResult& result,
                                float conf_threshold,
                                float nms_threshold) {

    if (image.empty()) {
        std::cerr << "[PupilDetector] 输入图像为空" << std::endl;
        return false;
    }

    try {
        // 1. 预处理（Letterbox，保持 BGR uint8 格式）
        float scale;
        int left, top;
        cv::Mat preprocessed = Preprocess(image, scale, left, top);

        // 2. 创建 OpenVINO Tensor（零拷贝方式）
        // 输入形状：[1, H, W, C] (NHWC, BGR, uint8)
        ov::Shape input_shape = {1, static_cast<size_t>(input_size_), 
                                 static_cast<size_t>(input_size_), 3};
        
        // 使用 cv::Mat 的数据指针创建 Tensor（零拷贝）
        ov::Tensor input_tensor(ov::element::u8, input_shape, preprocessed.data);

        // 3. 设置输入 Tensor
        infer_request_.set_input_tensor(input_tensor);

        // 4. 运行推理
        infer_request_.infer();

        // 5. 获取输出 Tensor
        ov::Tensor output_tensor = infer_request_.get_output_tensor();
        const float* output_data = output_tensor.data<float>();

        // 6. 后处理
        std::vector<float> scores;
        std::vector<cv::Rect> boxes = Postprocess(
            output_data, image.size(), scale, left, top,
            conf_threshold, nms_threshold, scores);

        if (boxes.empty()) {
            std::cerr << "[PupilDetector] 未检测到瞳孔" << std::endl;
            return false;
        }

        // 7. 选择置信度最高的框
        int best_idx = 0;
        float best_score = scores[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_idx = static_cast<int>(i);
            }
        }

        cv::Rect best_box = boxes[best_idx];

        // 8. 计算瞳孔中心和半宽
        result.center.x = best_box.x + best_box.width / 2.0f;
        result.center.y = best_box.y + best_box.height / 2.0f;
        result.half_width = (best_box.width + best_box.height) / 4.0f;  // 平均半径
        result.confidence = best_score;
        result.box = best_box;

        // 9. 裁剪 640x640 图像
        result.cropped_image = CropCenterRegion(image, result.center);

        std::cout << "[PupilDetector] 检测成功: center=(" << result.center.x
                  << ", " << result.center.y << "), half_width=" << result.half_width
                  << ", conf=" << result.confidence << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[PupilDetector] 推理失败: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat PupilDetector::DrawResult(const cv::Mat& image, const PupilDetectionResult& result) {
    cv::Mat vis = image.clone();

    // 绘制检测框
    cv::rectangle(vis, result.box, cv::Scalar(0, 255, 0), 2);

    // 绘制中心点
    cv::circle(vis, result.center, 3, cv::Scalar(0, 0, 255), -1);

    // 绘制标签
    std::string label = "pupil " + std::to_string(result.confidence).substr(0, 4);
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    int y = std::max(result.box.y, label_size.height + 10);

    cv::rectangle(vis,
                 cv::Point(result.box.x, y - label_size.height - 5),
                 cv::Point(result.box.x + label_size.width, y + 5),
                 cv::Scalar(0, 255, 0), -1);

    cv::putText(vis, label, cv::Point(result.box.x, y),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    return vis;
}

// ============================================================================
// CornealSpotDetector 实现（单例模式）
// ============================================================================

CornealSpotDetector* CornealSpotDetector::instance_ = NULL;

CornealSpotDetector::CornealSpotDetector()
    : input_size_(640),
      initialized_(false) {
}

CornealSpotDetector::~CornealSpotDetector() {
}

CornealSpotDetector* CornealSpotDetector::GetInstance() {
    if (instance_ == NULL) {
        instance_ = new CornealSpotDetector();
    }
    return instance_;
}

bool CornealSpotDetector::Initialize(const std::string& model_path, int input_size) {
    if (initialized_) {
        std::cout << "[CornealSpotDetector] 已初始化，跳过重复初始化" << std::endl;
        return true;
    }
    
    // 将路径转换为绝对路径
    model_path_ = resolve_path(model_path);
    input_size_ = input_size;
    
    try {
        std::cout << "[CornealSpotDetector] 正在加载 OpenVINO 模型..." << std::endl;
        std::cout << "[CornealSpotDetector] 模型路径: " << model_path_ << std::endl;
        
        // 1. 读取模型
        std::shared_ptr<ov::Model> model = core_.read_model(model_path_);
        
        // 2. 配置预处理（使用 PrePostProcessor API）
        ov::preprocess::PrePostProcessor ppp(model);
        
        // 获取输入输出名称
        input_name_ = model->input().get_any_name();
        output_name_ = model->output().get_any_name();
        
        // 配置输入
        // 输入格式：cv::Mat (HWC, BGR, uint8) -> 模型期望 (NCHW, RGB, float32)
        ppp.input().tensor()
            .set_element_type(ov::element::u8)           // 输入数据类型：uint8
            .set_layout("NHWC")                          // 输入布局：NHWC
            .set_color_format(ov::preprocess::ColorFormat::BGR);  // 输入颜色格式：BGR
        
        // 模型期望的输入格式
        ppp.input().model()
            .set_layout("NCHW");                         // 模型布局：NCHW
        
        // 预处理步骤
        ppp.input().preprocess()
            .convert_color(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB 转换
            .convert_element_type(ov::element::f32)              // uint8 -> float32
            .scale(255.0f);                                      // 归一化：除以 255
        
        // 应用预处理配置，构建新模型
        model = ppp.build();
        
        // 3. 编译模型到 CPU 设备
        compiled_model_ = core_.compile_model(model, "CPU");
        
        // 4. 创建推理请求
        infer_request_ = compiled_model_.create_infer_request();
        
        initialized_ = true;
        
        std::cout << "[CornealSpotDetector] 模型已加载: " << model_path_ << std::endl;
        std::cout << "  输入名: " << input_name_ << std::endl;
        std::cout << "  输出名: " << output_name_ << std::endl;
        std::cout << "  预处理已集成到模型图中（BGR->RGB, Normalize, Layout转换）" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[CornealSpotDetector] 初始化失败: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
}

cv::Mat CornealSpotDetector::Preprocess(const cv::Mat& img, float& scale, int& left, int& top) {
    int h = img.rows;
    int w = img.cols;

    scale = std::min(static_cast<float>(input_size_) / h,
                    static_cast<float>(input_size_) / w);

    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(new_w, new_h));

    top = (input_size_ - new_h) / 2;
    int bottom = input_size_ - new_h - top;
    left = (input_size_ - new_w) / 2;
    int right = input_size_ - new_w - left;

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 注意：不再需要手动进行 BGR->RGB 和归一化，这些已经集成到 OpenVINO 预处理中
    // 直接返回 BGR uint8 格式的图像
    return img_padded;
}

std::vector<SpotDetectionResult> CornealSpotDetector::Postprocess(
    const float* output,
    const cv::Size& img_size,
    float scale,
    int left,
    int top,
    float conf_threshold,
    float nms_threshold) {

    // YOLOv8 pose 输出: [1, 8, 8400]
    // 8个通道：[cx, cy, w, h, conf, kpt_x, kpt_y, kpt_conf]
    const int num_proposals = 8400;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kpt_confs;

    for (int i = 0; i < num_proposals; ++i) {
        float conf = output[4 * num_proposals + i];

        if (conf > conf_threshold) {
            float cx = output[0 * num_proposals + i];
            float cy = output[1 * num_proposals + i];
            float w = output[2 * num_proposals + i];
            float h = output[3 * num_proposals + i];
            float kpt_x = output[5 * num_proposals + i];
            float kpt_y = output[6 * num_proposals + i];
            float kpt_conf = output[7 * num_proposals + i];

            // 坐标还原
            float x1 = (cx - w / 2.0f - left) / scale;
            float y1 = (cy - h / 2.0f - top) / scale;
            float w_orig = w / scale;
            float h_orig = h / scale;
            float kpt_x_orig = (kpt_x - left) / scale;
            float kpt_y_orig = (kpt_y - top) / scale;

            boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                    static_cast<int>(w_orig), static_cast<int>(h_orig)));
            scores.push_back(conf);
            keypoints.push_back(cv::Point2f(kpt_x_orig, kpt_y_orig));
            kpt_confs.push_back(kpt_conf);
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    std::vector<SpotDetectionResult> results;
    for (size_t idx_i = 0; idx_i < indices.size(); ++idx_i) {
        int idx = indices[idx_i];
        SpotDetectionResult result;
        result.box = boxes[idx];
        result.score = scores[idx];
        result.keypoint = keypoints[idx];
        result.kpt_conf = kpt_confs[idx];

        // 计算长宽比
        float w = static_cast<float>(result.box.width);
        float h = static_cast<float>(result.box.height);
        result.aspect_ratio = (h > 0.0f) ? (w / h) : 0.0f;

        results.push_back(result);
    }

    return results;
}

std::vector<SpotDetectionResult> CornealSpotDetector::FilterSpots(
    const std::vector<SpotDetectionResult>& spots,
    float min_aspect_ratio,
    float max_aspect_ratio) {

    std::vector<SpotDetectionResult> filtered;
    for (size_t i = 0; i < spots.size(); ++i) {
        const SpotDetectionResult& spot = spots[i];
        if (spot.aspect_ratio >= min_aspect_ratio && spot.aspect_ratio <= max_aspect_ratio) {
            filtered.push_back(spot);
        }
    }
    return filtered;
}

std::vector<SpotDetectionResult> CornealSpotDetector::RemoveOutliers(
    const std::vector<SpotDetectionResult>& spots) {

    if (spots.size() < 3) {
        return spots;
    }

    // 计算中心
    cv::Point2f center = ComputeGeometricCenter(spots);

    // 计算距离
    std::vector<float> distances;
    for (size_t i = 0; i < spots.size(); ++i) {
        const SpotDetectionResult& spot = spots[i];
        float dx = spot.keypoint.x - center.x;
        float dy = spot.keypoint.y - center.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        distances.push_back(dist);
    }

    // 计算均值和标准差
    float mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    float sq_sum = 0.0f;
    for (size_t i = 0; i < distances.size(); ++i) {
        float d = distances[i];
        sq_sum += (d - mean_dist) * (d - mean_dist);
    }
    float std_dev = std::sqrt(sq_sum / distances.size());

    // 去除距离超过 mean + 2*std 的点
    float threshold = mean_dist + 2.0f * std_dev;
    std::vector<SpotDetectionResult> filtered;
    for (size_t i = 0; i < spots.size(); ++i) {
        if (distances[i] <= threshold) {
            filtered.push_back(spots[i]);
        }
    }

    return filtered;
}

std::vector<SpotDetectionResult> CornealSpotDetector::RemoveCenterSpots(
    const std::vector<SpotDetectionResult>& spots,
    const cv::Point2f& center,
    float center_threshold_ratio) {

    if (spots.size() < 3) {
        return spots;
    }

    // 计算所有点到中心的距离
    std::vector<float> distances;
    for (size_t i = 0; i < spots.size(); ++i) {
        const SpotDetectionResult& spot = spots[i];
        float dx = spot.keypoint.x - center.x;
        float dy = spot.keypoint.y - center.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        distances.push_back(dist);
    }

    // 计算平均距离
    float mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();

    // 中心光斑阈值：距离中心小于平均距离的 center_threshold_ratio 倍
    float center_threshold = mean_dist * center_threshold_ratio;

    // 剔除中心光斑
    std::vector<SpotDetectionResult> filtered;
    int removed_count = 0;
    for (size_t i = 0; i < spots.size(); ++i) {
        if (distances[i] > center_threshold) {
            filtered.push_back(spots[i]);
        } else {
            removed_count++;
        }
    }

    if (removed_count > 0) {
        std::cout << "[CornealSpotDetector] 剔除中心光斑: " << removed_count
                  << " 个 (阈值=" << center_threshold << ", 平均距离=" << mean_dist << ")" << std::endl;
    }

    return filtered;
}

cv::Point2f CornealSpotDetector::ComputeGeometricCenter(
    const std::vector<SpotDetectionResult>& spots) {

    if (spots.empty()) {
        return cv::Point2f(0, 0);
    }

    float sum_x = 0.0f, sum_y = 0.0f;
    for (size_t i = 0; i < spots.size(); ++i) {
        const SpotDetectionResult& spot = spots[i];
        sum_x += spot.keypoint.x;
        sum_y += spot.keypoint.y;
    }

    return cv::Point2f(sum_x / spots.size(), sum_y / spots.size());
}

float CornealSpotDetector::ComputeAverageAspectRatio(
    const std::vector<SpotDetectionResult>& spots) {

    if (spots.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < spots.size(); ++i) {
        const SpotDetectionResult& spot = spots[i];
        sum += spot.aspect_ratio;
    }

    return sum / spots.size();
}

void CornealSpotDetector::SeparateInnerOuter(
    const std::vector<SpotDetectionResult>& spots,
    const cv::Point2f& center,
    std::vector<SpotDetectionResult>& inner,
    std::vector<SpotDetectionResult>& outer) {

    if (spots.size() < 2) {
        return;
    }

    // 计算所有点到中心的距离
    std::vector<float> distances;
    for (size_t i = 0; i < spots.size(); ++i) {
        const SpotDetectionResult& spot = spots[i];
        float dx = spot.keypoint.x - center.x;
        float dy = spot.keypoint.y - center.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        distances.push_back(dist);
    }

    // 使用中位数作为分界
    std::vector<float> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    float median_dist = sorted_distances[sorted_distances.size() / 2];

    // 分离
    for (size_t i = 0; i < spots.size(); ++i) {
        if (distances[i] < median_dist) {
            inner.push_back(spots[i]);
        } else {
            outer.push_back(spots[i]);
        }
    }
}

EllipseFitResult CornealSpotDetector::FitEllipse(
    const std::vector<SpotDetectionResult>& spots) {

    EllipseFitResult result;
    result.valid = false;

    if (spots.size() < 5) {
        return result;
    }

    try {
        // 提取关键点
        std::vector<cv::Point2f> points;
        for (size_t i = 0; i < spots.size(); ++i) {
            const SpotDetectionResult& spot = spots[i];
            points.push_back(spot.keypoint);
        }

        // 使用 OpenCV 拟合椭圆
        cv::RotatedRect ellipse = cv::fitEllipse(points);

        result.center = ellipse.center;
        result.major_axis = std::max(ellipse.size.width, ellipse.size.height);
        result.minor_axis = std::min(ellipse.size.width, ellipse.size.height);
        result.angle = ellipse.angle;
        result.valid = true;

    } catch (const std::exception& e) {
        std::cerr << "[CornealSpotDetector] 椭圆拟合失败: " << e.what() << std::endl;
    }

    return result;
}

bool CornealSpotDetector::DetectAndAnalyze(
    const cv::Mat& image,
    CornealSpotAnalysisResult& result,
    float conf_threshold,
    float nms_threshold,
    int min_spots,
    int max_spots,
    float min_aspect_ratio,
    float max_aspect_ratio,
    int enable_ellipse_fit) {

    result.valid = false;
    result.error_message = "";

    if (image.empty()) {
        result.error_message = "输入图像为空";
        std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
        return false;
    }

    try {
        // 1. 预处理（Letterbox，保持 BGR uint8 格式）
        float scale;
        int left, top;
        cv::Mat preprocessed = Preprocess(image, scale, left, top);

        // 2. 创建 OpenVINO Tensor（零拷贝方式）
        // 输入形状：[1, H, W, C] (NHWC, BGR, uint8)
        ov::Shape input_shape = {1, static_cast<size_t>(input_size_), 
                                 static_cast<size_t>(input_size_), 3};
        
        // 使用 cv::Mat 的数据指针创建 Tensor（零拷贝）
        ov::Tensor input_tensor(ov::element::u8, input_shape, preprocessed.data);

        // 3. 设置输入 Tensor
        infer_request_.set_input_tensor(input_tensor);

        // 4. 运行推理
        infer_request_.infer();

        // 5. 获取输出 Tensor
        ov::Tensor output_tensor = infer_request_.get_output_tensor();
        const float* output_data = output_tensor.data<float>();

        // 6. 后处理
        std::vector<SpotDetectionResult> spots = Postprocess(
            output_data, image.size(), scale, left, top,
            conf_threshold, nms_threshold);

        std::cout << "[CornealSpotDetector] 初始检测数量: " << spots.size() << std::endl;

        // 7. 长宽比筛选
        spots = FilterSpots(spots, min_aspect_ratio, max_aspect_ratio);
        std::cout << "[CornealSpotDetector] 长宽比筛选后: " << spots.size() << std::endl;

        // 8. 离群值去除
        spots = RemoveOutliers(spots);
        std::cout << "[CornealSpotDetector] 离群值去除后: " << spots.size() << std::endl;

        // 8.5. 计算几何中心（用于剔除中心光斑）
        cv::Point2f geometric_center = ComputeGeometricCenter(spots);

        // 8.6. 剔除中心光斑（在内外环分离之前，避免中心光斑影响椭圆拟合）
        spots = RemoveCenterSpots(spots, geometric_center, 0.3f);
        std::cout << "[CornealSpotDetector] 中心光斑剔除后: " << spots.size() << std::endl;

        // 9. 数量检查
        if (static_cast<int>(spots.size()) < min_spots) {
            result.error_message = "光斑数量过少 (" + std::to_string(spots.size()) +
                                  " < " + std::to_string(min_spots) + ")";
            std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
            return false;
        }

        if (static_cast<int>(spots.size()) > max_spots) {
            result.error_message = "光斑数量过多 (" + std::to_string(spots.size()) +
                                  " > " + std::to_string(max_spots) + ")";
            std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
            return false;
        }

        // 10. 计算统计信息（使用剔除中心光斑后的数据重新计算几何中心）
        result.num_spots = static_cast<int>(spots.size());
        result.geometric_center = ComputeGeometricCenter(spots);  // 重新计算，排除中心光斑的影响
        result.avg_aspect_ratio = ComputeAverageAspectRatio(spots);
        result.all_spots = spots;

        std::cout << "[CornealSpotDetector] 分析成功: " << std::endl;
        std::cout << "  光斑数量: " << result.num_spots << std::endl;
        std::cout << "  几何中心: (" << result.geometric_center.x
                  << ", " << result.geometric_center.y << ")" << std::endl;
        std::cout << "  平均长宽比: " << result.avg_aspect_ratio << std::endl;

        // 11. 内外环分离和椭圆拟合（仅在enable_ellipse_fit=1时执行）
        if (enable_ellipse_fit == 1) {
            SeparateInnerOuter(spots, result.geometric_center,
                              result.inner_spots, result.outer_spots);

            std::cout << "[CornealSpotDetector] 内环: " << result.inner_spots.size()
                      << ", 外环: " << result.outer_spots.size() << std::endl;

            // 12. 椭圆拟合
            result.inner_ellipse = FitEllipse(result.inner_spots);
            result.outer_ellipse = FitEllipse(result.outer_spots);

            if (result.inner_ellipse.valid) {
                std::cout << "  内环椭圆: 中心(" << result.inner_ellipse.center.x
                          << ", " << result.inner_ellipse.center.y
                          << "), 长轴=" << result.inner_ellipse.major_axis
                          << ", 短轴=" << result.inner_ellipse.minor_axis << std::endl;
            }

            if (result.outer_ellipse.valid) {
                std::cout << "  外环椭圆: 中心(" << result.outer_ellipse.center.x
                          << ", " << result.outer_ellipse.center.y
                          << "), 长轴=" << result.outer_ellipse.major_axis
                          << ", 短轴=" << result.outer_ellipse.minor_axis << std::endl;
            }
        } else {
            // 不进行椭圆拟合时，清空相关数据
            result.inner_spots.clear();
            result.outer_spots.clear();
            result.inner_ellipse.valid = false;
            result.outer_ellipse.valid = false;
        }

        result.valid = true;

        return true;

    } catch (const std::exception& e) {
        result.error_message = std::string("推理失败: ") + e.what();
        std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
        return false;
    }
}

cv::Mat CornealSpotDetector::DrawResult(const cv::Mat& image,
                                        const CornealSpotAnalysisResult& result) {
    cv::Mat vis = image.clone();

    if (!result.valid) {
        return vis;
    }

    // 绘制所有光斑
    for (size_t i = 0; i < result.all_spots.size(); ++i) {
        const SpotDetectionResult& spot = result.all_spots[i];
        // 绘制检测框（蓝色）
        cv::rectangle(vis, spot.box, cv::Scalar(255, 0, 0), 2);

        // 绘制关键点（红色）
        cv::circle(vis, spot.keypoint, 3, cv::Scalar(0, 0, 255), -1);

        // 绘制标签
        std::string label = std::to_string(spot.score).substr(0, 4) + " r=" +
                           std::to_string(spot.aspect_ratio).substr(0, 4);
        cv::putText(vis, label, cv::Point(spot.box.x, std::max(0, spot.box.y - 5)),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }

    // 绘制几何中心（绿色大圆）
    cv::circle(vis, result.geometric_center, 5, cv::Scalar(0, 255, 0), -1);
    cv::putText(vis, "Center",
               cv::Point(static_cast<int>(result.geometric_center.x + 10), 
                         static_cast<int>(result.geometric_center.y)),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    // 绘制内环椭圆（黄色）
    if (result.inner_ellipse.valid) {
        cv::RotatedRect inner_rect(result.inner_ellipse.center,
                                   cv::Size2f(result.inner_ellipse.major_axis,
                                             result.inner_ellipse.minor_axis),
                                   result.inner_ellipse.angle);
        cv::ellipse(vis, inner_rect, cv::Scalar(0, 255, 255), 2);
    }

    // 绘制外环椭圆（青色）
    if (result.outer_ellipse.valid) {
        cv::RotatedRect outer_rect(result.outer_ellipse.center,
                                   cv::Size2f(result.outer_ellipse.major_axis,
                                             result.outer_ellipse.minor_axis),
                                   result.outer_ellipse.angle);
        cv::ellipse(vis, outer_rect, cv::Scalar(255, 255, 0), 2);
    }

    // 绘制统计信息
    int y_offset = 20;
    cv::putText(vis, "Spots: " + std::to_string(result.num_spots),
               cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6,
               cv::Scalar(255, 255, 255), 2);
    y_offset += 25;

    std::string ratio_text = "Avg Ratio: " + std::to_string(result.avg_aspect_ratio).substr(0, 5);
    cv::putText(vis, ratio_text, cv::Point(10, y_offset),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    return vis;
}

cv::Mat CornealSpotDetector::DrawSimpleResult(const cv::Mat& image,
                                               const CornealSpotAnalysisResult& result) {
    cv::Mat vis = image.clone();
    
    if (!result.valid) {
        return vis;
    }
    
    // 绘制每个光斑框和长宽比
    for (size_t i = 0; i < result.all_spots.size(); ++i) {
        const SpotDetectionResult& spot = result.all_spots[i];
        // 绘制检测框（蓝色）
        cv::rectangle(vis, spot.box, cv::Scalar(255, 0, 0), 2);
        
        // 绘制长宽比标注
        std::string ratio_label = "r=" + std::to_string(spot.aspect_ratio).substr(0, 4);
        cv::putText(vis, ratio_label, 
                   cv::Point(spot.box.x, spot.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }
    
    // 绘制几何中心（绿色大圆+标签）
    cv::circle(vis, result.geometric_center, 8, cv::Scalar(0, 255, 0), -1);
    cv::circle(vis, result.geometric_center, 10, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis, "Geometric Center", 
               cv::Point(static_cast<int>(result.geometric_center.x + 15), 
                         static_cast<int>(result.geometric_center.y - 5)),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    // 在左上角显示统计信息
    int y_offset = 25;
    cv::putText(vis, "Filtered Spots: " + std::to_string(result.num_spots),
               cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.7,
               cv::Scalar(255, 255, 255), 2);
    y_offset += 30;
    
    std::string avg_ratio = "Avg Ratio: " + std::to_string(result.avg_aspect_ratio).substr(0, 5);
    cv::putText(vis, avg_ratio, cv::Point(10, y_offset),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return vis;
}

