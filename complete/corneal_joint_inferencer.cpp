#include "corneal_joint_inferencer.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <cstring>  // for std::memcpy
#include <thread>
#include <chrono>
#include <map>
#include <set>
#include <sstream>  // 替代 QDebug

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
std::mutex PupilDetector::instance_mutex_;
std::mutex PupilDetector::global_inference_mutex_;
PupilDetector::PupilDetector()
    : input_size_(640),
      initialized_(false) {
}

PupilDetector::~PupilDetector() {
}

PupilDetector* PupilDetector::GetInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (instance_ == NULL) {
        instance_ = new PupilDetector();
    }
    return instance_;
}

bool PupilDetector::Initialize(const std::string& model_path, int input_size, int num_infer_requests) {
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
        std::cout << "[PupilDetector] 推理请求数量: " << num_infer_requests << " (用于多线程并发)" << std::endl;

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

        // 4. 创建多个推理请求（用于多线程并发）
        infer_requests_.clear();
        infer_request_available_.clear();

        if (num_infer_requests <= 0) {
            num_infer_requests = 1;
        }

        infer_requests_.reserve(num_infer_requests);
        infer_request_available_.reserve(num_infer_requests);

        for (int i = 0; i < num_infer_requests; ++i) {
            infer_requests_.push_back(compiled_model_.create_infer_request());
            infer_request_available_.push_back(true);  // 初始状态为可用
        }

        initialized_ = true;

        std::cout << "[PupilDetector] 模型已加载: " << model_path_ << std::endl;
        std::cout << "  输入名: " << input_name_ << std::endl;
        std::cout << "  输出名: " << output_name_ << std::endl;
        std::cout << "  输入尺寸: [1, " << input_size_ << ", " << input_size_ << ", 3]" << std::endl;
        std::cout << "  InferRequest对象池大小: " << infer_requests_.size() << std::endl;
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

// 获取可用的 InferRequest（线程安全）
int PupilDetector::acquireInferRequest() {
    std::unique_lock<std::mutex> lock(infer_requests_mutex_);

    // 添加调试信息
    static std::atomic<int> acquire_counter{0};
    int call_id = ++acquire_counter;

    if (infer_request_available_.empty()) {
        std::cerr << "[PupilDetector][" << call_id
                  << "] ERROR: InferRequest对象池为空，可能未成功初始化模型" << std::endl;
        return -1;
    }

    int available_count = std::count(infer_request_available_.begin(),
                                     infer_request_available_.end(), true);
    std::cout << "[PupilDetector][" << call_id
              << "] acquireInferRequest: entering, available: "
              << available_count
              << "/" << infer_request_available_.size() << std::endl;

    // 使用条件变量等待，带超时机制
    auto timeout = std::chrono::seconds(5);  // 5秒超时
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        // 检查是否有可用的InferRequest
        for (size_t i = 0; i < infer_request_available_.size(); ++i) {
            if (infer_request_available_[i]) {
                infer_request_available_[i] = false;  // 标记为占用
                std::cout << "[PupilDetector][" << call_id
                          << "] acquireInferRequest: acquired request "
                          << i << std::endl;
                return static_cast<int>(i);
            }
        }

        // 检查超时
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

        if (elapsed >= timeout) {
            std::cerr << "[PupilDetector][" << call_id
                      << "] ERROR: acquireInferRequest timeout after "
                      << timeout.count() << " seconds! 所有 "
                      << infer_request_available_.size()
                      << " 个 InferRequest 均被占用" << std::endl;
            return -1;
        }

        // 等待通知，带超时
        std::cout << "[PupilDetector][" << call_id
                  << "] acquireInferRequest: waiting..." << std::endl;
        infer_request_cv_.wait_for(lock, std::chrono::milliseconds(100));
    }
}

// 释放 InferRequest（线程安全）
void PupilDetector::releaseInferRequest(int index) {
    if (index < 0 || index >= static_cast<int>(infer_request_available_.size())) {
        std::cerr << "[PupilDetector] ERROR: releaseInferRequest invalid index: "
                  << index << std::endl;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(infer_requests_mutex_);
        infer_request_available_[index] = true;  // 标记为可用

        std::cout << "[PupilDetector] releaseInferRequest: released " << index
                  << ", available now: "
                  << std::count(infer_request_available_.begin(),
                               infer_request_available_.end(), true)
                  << "/" << infer_request_available_.size() << std::endl;
    }

    // 通知等待的线程
    infer_request_cv_.notify_all();
}

std::vector<cv::Rect> PupilDetector::Postprocess(
    const float* output,
    const ov::Shape& output_shape,
    const cv::Size& img_size,
    float scale,
    int left,
    int top,
    float conf_threshold,
    float nms_threshold,
    std::vector<float>& scores) {

    // YOLOv8 detect 输出格式: [1, 5, num_proposals] 或 [1, num_proposals, 5]
    // 5个通道：[cx, cy, w, h, conf]
    // 根据输出tensor形状动态计算proposal数量
    int num_proposals = 0;
    bool is_channel_first = false;  // true: [1, 5, N], false: [1, N, 5]

    if (output_shape.size() == 3) {
        if (output_shape[1] == 5) {
            // 格式: [1, 5, num_proposals]
            num_proposals = static_cast<int>(output_shape[2]);
            is_channel_first = true;
        } else if (output_shape[2] == 5) {
            // 格式: [1, num_proposals, 5]
            num_proposals = static_cast<int>(output_shape[1]);
            is_channel_first = false;
        } else {
            std::cerr << "[PupilDetector] ERROR: 不支持的输出格式，shape=["
                     << output_shape[0] << "," << output_shape[1] << "," << output_shape[2] << "]" << std::endl;
            return std::vector<cv::Rect>();
        }
    } else {
        std::cerr << "[PupilDetector] ERROR: 输出tensor维度不是3，实际维度=" << output_shape.size() << std::endl;
        return std::vector<cv::Rect>();
    }

    std::cout << "[PupilDetector] Postprocess: num_proposals=" << num_proposals
             << ", is_channel_first=" << is_channel_first << std::endl;

    std::vector<cv::Rect> boxes;
    scores.clear();

    for (int i = 0; i < num_proposals; ++i) {
        float cx, cy, w, h, conf;

        if (is_channel_first) {
            // 格式: [1, 5, num_proposals] - 通道优先
            cx = output[0 * num_proposals + i];
            cy = output[1 * num_proposals + i];
            w = output[2 * num_proposals + i];
            h = output[3 * num_proposals + i];
            conf = output[4 * num_proposals + i];
        } else {
            // 格式: [1, num_proposals, 5] - proposal优先
            cx = output[i * 5 + 0];
            cy = output[i * 5 + 1];
            w = output[i * 5 + 2];
            h = output[i * 5 + 3];
            conf = output[i * 5 + 4];
        }

        // 调试：检查前几个proposal的值
        if (i < 3) {
            std::cout << "[PupilDetector] Proposal" << i
                     << ": cx=" << cx << ", cy=" << cy
                     << ", w=" << w << ", h=" << h << ", conf=" << conf << std::endl;
        }

        if (conf > conf_threshold) {

            // 坐标还原到原图
            float x1 = (cx - w / 2.0f - left) / scale;
            float y1 = (cy - h / 2.0f - top) / scale;
            float w_orig = w / scale;
            float h_orig = h / scale;

            // 调试：检查坐标还原后的值
            if (boxes.size() < 3) {
                std::cout << "[PupilDetector] 坐标还原: x1=" << x1 << ", y1=" << y1
                         << ", w_orig=" << w_orig << ", h_orig=" << h_orig
                         << ", img_size=" << img_size.width << "x" << img_size.height << std::endl;
            }

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

bool PupilDetector::DetectPupil(const cv::Mat& image,
                                PupilDetectionResult& result,
                                float conf_threshold,
                                float nms_threshold) {

    // 调试信息：基础状态
    std::cout << "[PupilDetector] DetectPupil called. initialized=" << (initialized_ ? 1 : 0)
              << ", image_size=" << image.cols << "x" << image.rows
              << ", channels=" << image.channels()
              << ", conf_threshold=" << conf_threshold
              << ", nms_threshold=" << nms_threshold
              << std::endl;

    if (image.empty()) {
        std::cerr << "[PupilDetector] 输入图像为空" << std::endl;
        return false;
    }

    // 从对象池获取可用的 InferRequest（线程安全）
    int request_index = acquireInferRequest();

    // 检查是否成功获取InferRequest
    if (request_index < 0) {
        std::cerr << "[PupilDetector] 无法获取可用的InferRequest，请求可能超时" << std::endl;
        return false;
    }

    try {
        // 1. 预处理（Letterbox，保持 BGR uint8 格式）
        float scale;
        int left, top;
        cv::Mat preprocessed = Preprocess(image, scale, left, top);

        // 2. 创建 OpenVINO Tensor（使用深拷贝确保数据安全）
        // 输入形状：[1, H, W, C] (NHWC, BGR, uint8)
        ov::Shape input_shape = {1, static_cast<size_t>(input_size_),
                                 static_cast<size_t>(input_size_), 3};

        // 使用深拷贝方式创建Tensor，避免cv::Mat生命周期问题
        size_t data_size = input_size_ * input_size_ * 3 * sizeof(uint8_t);
        std::vector<uint8_t> input_data(data_size);

        // 深拷贝图像数据
        std::memcpy(input_data.data(), preprocessed.data, data_size);

        // 使用深拷贝的数据创建Tensor
        ov::Tensor input_tensor(ov::element::u8, input_shape, input_data.data());

        // 3. 获取当前线程专用的 InferRequest
        ov::InferRequest& infer_request = infer_requests_[request_index];

        // 4. 设置输入 Tensor
        infer_request.set_input_tensor(input_tensor);

        // 5. 运行推理（使用全局互斥锁保护，防止OpenVINO内部并发冲突）
        // 注意：虽然降低了并发性能，但这是为了避免OpenVINO内部的竞态条件导致崩溃
        {
            std::lock_guard<std::mutex> global_lock(PupilDetector::global_inference_mutex_);
            std::cout << "[PupilDetector] 开始推理 (request_index=" << request_index << ")" << std::endl;
            infer_request.infer();
            std::cout << "[PupilDetector] 推理完成 (request_index=" << request_index << ")" << std::endl;
        }

        // 6. 获取输出 Tensor
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<float>();
        auto output_shape = output_tensor.get_shape();

        // 7. 后处理
        std::vector<float> scores;
        std::vector<cv::Rect> boxes = Postprocess(
            output_data, output_shape, image.size(), scale, left, top,
            conf_threshold, nms_threshold, scores);

        // 推理完成，立即释放 InferRequest 供其他线程使用
        releaseInferRequest(request_index);

        if (boxes.empty()) {
            std::cerr << "[PupilDetector] 未检测到瞳孔，boxes 为空。"
                      << " image_size=" << image.cols << "x" << image.rows
                      << ", conf_threshold=" << conf_threshold
                      << ", nms_threshold=" << nms_threshold
                      << std::endl;
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
        // 异常情况下也要释放 InferRequest
        releaseInferRequest(request_index);

        // 使用 std::cerr 输出完整异常信息
        std::cerr << "[PupilDetector] 推理失败: " << e.what() << std::endl;
        return false;
    }
}

// 自适应预处理：支持任意输入尺寸，直接缩放到模型输入尺寸（320×320）
cv::Mat PupilDetector::PreprocessAdaptive(const cv::Mat& img, float& scale, int& left, int& top) {
    int h = img.rows;
    int w = img.cols;

    // 直接计算从输入图像到模型输入尺寸的缩放比例（保持宽高比，letterbox方式）
    scale = std::min(static_cast<float>(input_size_) / h,
                    static_cast<float>(input_size_) / w);

    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    // 缩放图像
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 计算填充（letterbox方式）
    top = (input_size_ - new_h) / 2;
    int bottom = input_size_ - new_h - top;
    left = (input_size_ - new_w) / 2;
    int right = input_size_ - new_w - left;

    // 填充灰边 (114, 114, 114)
    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return img_padded;
}

// 自适应检测函数：支持任意输入尺寸
bool PupilDetector::DetectPupilAdaptive(const cv::Mat& image,
                                        PupilDetectionResult& result,
                                        float conf_threshold,
                                        float nms_threshold) {

    // 调试信息：基础状态
    std::cout << "[PupilDetector] DetectPupilAdaptive called. initialized="
             << (initialized_ ? 1 : 0)
             << " image_size=" << image.cols << "x" << image.rows
             << " channels=" << image.channels()
             << " conf_threshold=" << conf_threshold
             << " nms_threshold=" << nms_threshold << std::endl;

    if (image.empty()) {
        std::cerr << "[PupilDetector] 输入图像为空" << std::endl;
        return false;
    }

    // 从对象池获取可用的 InferRequest（线程安全）
    int request_index = acquireInferRequest();

    // 检查是否成功获取InferRequest
    if (request_index < 0) {
        std::cerr << "[PupilDetector] 无法获取可用的InferRequest，请求可能超时" << std::endl;
        return false;
    }

    try {
        // 1. 自适应预处理（支持任意输入尺寸）
        float scale;
        int left, top;
        cv::Mat preprocessed = PreprocessAdaptive(image, scale, left, top);

        // 2. 创建 OpenVINO Tensor
        ov::Shape input_shape = {1, static_cast<size_t>(input_size_),
                                 static_cast<size_t>(input_size_), 3};

        size_t data_size = input_size_ * input_size_ * 3 * sizeof(uint8_t);
        std::vector<uint8_t> input_data(data_size);
        std::memcpy(input_data.data(), preprocessed.data, data_size);

        ov::Tensor input_tensor(ov::element::u8, input_shape, input_data.data());

        // 3. 获取当前线程专用的 InferRequest
        ov::InferRequest& infer_request = infer_requests_[request_index];

        // 4. 设置输入 Tensor
        infer_request.set_input_tensor(input_tensor);

        // 5. 运行推理
        {
            std::lock_guard<std::mutex> global_lock(PupilDetector::global_inference_mutex_);
            std::cout << "[PupilDetector] 开始推理 (request_index=" << request_index << ")" << std::endl;
            infer_request.infer();
            std::cout << "[PupilDetector] 推理完成 (request_index=" << request_index << ")" << std::endl;
        }

        // 6. 获取输出 Tensor
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<float>();

        // 调试：输出tensor的形状和内容
        auto output_shape = output_tensor.get_shape();
        std::cout << "[PupilDetector] 输出tensor形状: ["
                 << output_shape[0] << "," << output_shape[1] << ","
                 << output_shape[2] << "]" << std::endl;
        std::cout << "[PupilDetector] 预处理参数: scale=" << scale
                 << ", left=" << left << ", top=" << top
                 << ", input_size_=" << input_size_ << std::endl;

        // 7. 后处理（坐标还原到原始输入图像尺寸）
        std::vector<float> scores;
        std::vector<cv::Rect> boxes = Postprocess(
            output_data, output_shape, image.size(), scale, left, top,
            conf_threshold, nms_threshold, scores);

        std::cout << "[PupilDetector] Postprocess返回: boxes数量=" << boxes.size()
                 << ", scores数量=" << scores.size() << std::endl;

        // 推理完成，立即释放 InferRequest
        releaseInferRequest(request_index);

        if (boxes.empty()) {
            std::cerr << "[PupilDetector] 未检测到瞳孔，boxes 为空。"
                     << " image_size=" << image.cols << "x" << image.rows
                     << " conf_threshold=" << conf_threshold
                     << " nms_threshold=" << nms_threshold << std::endl;
            return false;
        }

        // 8. 选择置信度最高的框
        int best_idx = 0;
        float best_score = scores[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_idx = static_cast<int>(i);
            }
        }

        cv::Rect best_box = boxes[best_idx];

        // 9. 计算瞳孔中心和半宽（坐标已在Postprocess中还原到原始输入图像）
        result.center.x = best_box.x + best_box.width / 2.0f;
        result.center.y = best_box.y + best_box.height / 2.0f;
        result.half_width = (best_box.width + best_box.height) / 4.0f;
        result.confidence = best_score;
        result.box = best_box;

        std::cout << "[PupilDetector] 检测成功: center=(" << result.center.x
                 << "," << result.center.y << "), half_width=" << result.half_width
                 << ", conf=" << result.confidence
                 << ", box=(" << best_box.x << "," << best_box.y
                 << "," << best_box.width << "," << best_box.height << ")" << std::endl;

        return true;

    } catch (const std::exception& e) {
        releaseInferRequest(request_index);
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
std::mutex CornealSpotDetector::instance_mutex_;
std::mutex CornealSpotDetector::global_inference_mutex_;

CornealSpotDetector::CornealSpotDetector()
    : input_size_(640),
      initialized_(false) {
}

CornealSpotDetector::~CornealSpotDetector() {
}

CornealSpotDetector* CornealSpotDetector::GetInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (instance_ == NULL) {
        instance_ = new CornealSpotDetector();
    }
    return instance_;
}

bool CornealSpotDetector::Initialize(const std::string& model_path, int input_size, int num_infer_requests) {
    if (initialized_) {
        std::cout << "[CornealSpotDetector] 已初始化，跳过重复初始化" << std::endl;
        return true;
    }

    // 将路径转换为绝对路径
    model_path_ = resolve_path(model_path);

    try {
        std::cout << "[CornealSpotDetector] 正在加载 OpenVINO UNet 模型..." << std::endl;
        std::cout << "[CornealSpotDetector] 模型路径: " << model_path_ << std::endl;
        std::cout << "[CornealSpotDetector] 推理请求数量: " << num_infer_requests << " (用于多线程并发)" << std::endl;

        // 1. 读取模型
        std::shared_ptr<ov::Model> model = core_.read_model(model_path_);

        // 1.5. 从模型自动获取输入尺寸（如果未指定或为0）
        if (input_size <= 0) {
            auto input_shape = model->input().get_shape();
            if (input_shape.size() == 4) {  // [N, C, H, W]
                input_size_ = static_cast<int>(input_shape[2]);  // 假设H=W（正方形输入）
                std::cout << "[CornealSpotDetector] 从模型自动检测输入尺寸: "
                          << input_size_ << "x" << input_size_ << std::endl;
            } else {
                std::cerr << "[CornealSpotDetector] 无法从模型获取输入尺寸，使用默认320" << std::endl;
                input_size_ = 320;
            }
        } else {
            input_size_ = input_size;
            std::cout << "[CornealSpotDetector] 使用指定的输入尺寸: "
                      << input_size_ << "x" << input_size_ << std::endl;
        }

        // 2. 配置预处理（UNet使用ImageNet标准化）
        ov::preprocess::PrePostProcessor ppp(model);

        // 获取输入输出名称
        input_name_ = model->input().get_any_name();
        output_name_ = model->output().get_any_name();

        // 配置输入：cv::Mat (HWC, BGR, uint8) -> 模型期望 (NCHW, RGB, float32, ImageNet标准化)
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

        ppp.input().model()
            .set_layout("NCHW");

        // UNet预处理步骤：BGR->RGB, uint8->float32, /255.0, ImageNet标准化
        ppp.input().preprocess()
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .convert_element_type(ov::element::f32)
            .scale(255.0f)  // 先 /255.0
            .mean({0.485f, 0.456f, 0.406f})  // ImageNet mean
            .scale({0.229f, 0.224f, 0.225f});  // ImageNet std (这里scale实际是除以std)

        // 应用预处理配置，构建新模型
        model = ppp.build();

        // 3. 编译模型到 CPU 设备
        compiled_model_ = core_.compile_model(model, "CPU");

        // 4. 创建多个推理请求（用于多线程并发）
        infer_requests_.clear();
        infer_request_available_.clear();

        if (num_infer_requests <= 0) {
            num_infer_requests = 1;
        }

        infer_requests_.reserve(num_infer_requests);
        infer_request_available_.reserve(num_infer_requests);

        for (int i = 0; i < num_infer_requests; ++i) {
            infer_requests_.push_back(compiled_model_.create_infer_request());
            infer_request_available_.push_back(true);  // 初始状态为可用
        }

        initialized_ = true;

        std::cout << "[CornealSpotDetector] UNet模型已加载: " << model_path_ << std::endl;
        std::cout << "  输入名: " << input_name_ << std::endl;
        std::cout << "  输出名: " << output_name_ << std::endl;
        std::cout << "  InferRequest对象池大小: " << infer_requests_.size() << std::endl;
        std::cout << "  预处理已集成到模型图中（BGR->RGB, ImageNet标准化, Layout转换）" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[CornealSpotDetector] 初始化失败: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
}

cv::Mat CornealSpotDetector::Preprocess(const cv::Mat& img, float& scale, int& left, int& top) {
    // 自适应处理任意输入尺寸，统一resize到UNet模型输入尺寸（320x320）
    // 支持：640x640, 任意尺寸输入

    int h = img.rows;
    int w = img.cols;

    // 如果输入已经是目标尺寸，直接返回（避免不必要的resize）
    if (h == input_size_ && w == input_size_) {
        scale = 1.0f;
        left = 0;
        top = 0;
        return img.clone();
    }

    // 计算缩放比例：保持宽高比，以长边为准缩放到input_size
    // 例如：640x640 -> 320x320 (scale=0.5)
    //      800x600 -> 320x240 (scale=0.4, 然后填充到320x320)
    scale = static_cast<float>(input_size_) / std::max(h, w);

    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    // Resize图像（保持宽高比）
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 计算填充参数（居中填充到input_size x input_size）
    top = (input_size_ - new_h) / 2;
    int bottom = input_size_ - new_h - top;
    left = (input_size_ - new_w) / 2;
    int right = input_size_ - new_w - left;

    // 填充到目标尺寸（黑色填充）
    cv::Mat img_padded;
    if (top > 0 || bottom > 0 || left > 0 || right > 0) {
        cv::copyMakeBorder(img_resized, img_padded, top, bottom, left, right,
                          cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        // 如果不需要填充（已经是目标尺寸），直接使用resized图像
        img_padded = img_resized;
    }

    // 返回 BGR uint8 格式的图像，ImageNet标准化已集成到OpenVINO预处理中
    return img_padded;
}

// 获取可用的 InferRequest（线程安全）
int CornealSpotDetector::acquireInferRequest() {
    std::unique_lock<std::mutex> lock(infer_requests_mutex_);

    // 添加调试信息
    static std::atomic<int> acquire_counter{0};
    int call_id = ++acquire_counter;

    std::cout << "[CornealSpotDetector][" << call_id
             << "] acquireInferRequest: entering, available: "
             << std::count(infer_request_available_.begin(),
                          infer_request_available_.end(), true)
             << "/" << infer_request_available_.size() << std::endl;

    // 方法1：使用 wait_for 添加超时（推荐）
    auto timeout = std::chrono::seconds(5);  // 5秒超时

    bool acquired = false;
    int acquired_index = -1;
    auto start_time = std::chrono::steady_clock::now();

    while (!acquired) {
        // 检查是否有可用的
        for (size_t i = 0; i < infer_request_available_.size(); ++i) {
            if (infer_request_available_[i]) {
                infer_request_available_[i] = false;
                acquired_index = static_cast<int>(i);
                acquired = true;

                std::cout << "[CornealSpotDetector] acquireInferRequest: acquired request " << acquired_index << std::endl;
                break;
            }
        }

        if (acquired) {
            break;
        }

        // 检查超时
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

        if (elapsed >= timeout) {
            std::cerr << "[CornealSpotDetector] ERROR: acquireInferRequest timeout after "
                     << timeout.count() << " seconds!" << std::endl;
            return -1;
        }

        // 等待通知，带超时
        std::cout << "[CornealSpotDetector] acquireInferRequest: waiting..." << std::endl;

        infer_request_cv_.wait_for(lock, std::chrono::milliseconds(100));
    }

    return acquired_index;

    // 方法2：使用你的原始版本，但添加超时
    /*
    auto result = infer_request_cv_.wait_for(lock, std::chrono::seconds(30), [this]() {
        for (bool available : infer_request_available_) {
            if (available) return true;
        }
        return false;
    });

    if (!result) {
        std::cerr << "[CornealSpotDetector] acquireInferRequest timeout!" << std::endl;
        return -1;
    }

    for (size_t i = 0; i < infer_request_available_.size(); ++i) {
        if (infer_request_available_[i]) {
            infer_request_available_[i] = false;
            return static_cast<int>(i);
        }
    }

    return -1;
    */
}

// 释放 InferRequest（线程安全）
void CornealSpotDetector::releaseInferRequest(int index) {
    if (index < 0 || index >= static_cast<int>(infer_request_available_.size())) {
        std::cerr << "[CornealSpotDetector] ERROR: releaseInferRequest invalid index: " << index << std::endl;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(infer_requests_mutex_);
        infer_request_available_[index] = true;

        std::cout << "[CornealSpotDetector] releaseInferRequest: released " << index
                 << ", available now: "
                 << std::count(infer_request_available_.begin(),
                              infer_request_available_.end(), true)
                 << "/" << infer_request_available_.size() << std::endl;
    }

    // 重要：使用 notify_all 而不是 notify_one
    // 因为可能有多个线程在等待
    infer_request_cv_.notify_all();
}

std::vector<SpotDetectionResult> CornealSpotDetector::PostprocessUNet(
    const float* output,
    const cv::Size& img_size,
    float scale,
    int left,
    int top,
    cv::Mat& mask,
    float threshold) {

    // UNet输出: [1, 1, H, W]，logits值（未经sigmoid）
    int output_h = input_size_;
    int output_w = input_size_;

    // 1. Sigmoid激活 + 二值化
    mask = cv::Mat(output_h, output_w, CV_8UC1);
    for (int i = 0; i < output_h * output_w; ++i) {
        float logit = output[i];
        // Sigmoid
        float prob = 1.0f / (1.0f + std::exp(-logit));
        // 二值化
        mask.data[i] = (prob > threshold) ? 255 : 0;
    }

    // 2. 形态学操作：去除噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 3. 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "[CornealSpotDetector] UNet检测到 " << contours.size() << " 个轮廓" << std::endl;

    // 4. 从轮廓提取光斑信息
    std::vector<SpotDetectionResult> results;

    for (size_t i = 0; i < contours.size(); ++i) {
        const std::vector<cv::Point>& contour = contours[i];

        // 过滤太小的轮廓
        double area = cv::contourArea(contour);
        if (area < 5.0) {  // 最小面积阈值（像素）
            continue;
        }

        // 计算轮廓的外接矩形
        cv::Rect bbox = cv::boundingRect(contour);

        // 计算轮廓的重心（矩）
        cv::Moments m = cv::moments(contour);
        cv::Point2f centroid;
        if (m.m00 > 0) {
            centroid.x = static_cast<float>(m.m10 / m.m00);
            centroid.y = static_cast<float>(m.m01 / m.m00);
        } else {
            centroid.x = bbox.x + bbox.width / 2.0f;
            centroid.y = bbox.y + bbox.height / 2.0f;
        }

        // 坐标还原：从模型输出坐标系还原到原图坐标系
        // 步骤：去除padding -> 缩放还原
        float centroid_x_orig = (centroid.x - left) / scale;
        float centroid_y_orig = (centroid.y - top) / scale;

        float bbox_x_orig = (bbox.x - left) / scale;
        float bbox_y_orig = (bbox.y - top) / scale;
        float bbox_w_orig = bbox.width / scale;
        float bbox_h_orig = bbox.height / scale;

        // 创建SpotDetectionResult
        SpotDetectionResult result;
        result.box = cv::Rect(static_cast<int>(bbox_x_orig),
                             static_cast<int>(bbox_y_orig),
                             static_cast<int>(bbox_w_orig),
                             static_cast<int>(bbox_h_orig));
        result.keypoint = cv::Point2f(centroid_x_orig, centroid_y_orig);  // 使用重心作为关键点
        result.score = 1.0f;  // UNet没有置信度，设为1.0
        result.kpt_conf = 1.0f;

        // 计算长宽比
        result.aspect_ratio = (bbox.height > 0) ?
            (static_cast<float>(bbox.width) / static_cast<float>(bbox.height)) : 0.0f;

        results.push_back(result);
    }

    std::cout << "[CornealSpotDetector] 过滤后有效轮廓: " << results.size() << std::endl;

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

// DBSCAN辅助函数：计算两点之间的欧氏距离
static float euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// DBSCAN辅助函数：找到eps邻域内的所有点
static std::vector<int> regionQuery(const std::vector<cv::Point2f>& points,
                                    int pointIdx, float eps) {
    std::vector<int> neighbors;
    const cv::Point2f& point = points[pointIdx];

    for (size_t i = 0; i < points.size(); ++i) {
        if (euclideanDistance(point, points[i]) <= eps) {
            neighbors.push_back(static_cast<int>(i));
        }
    }

    return neighbors;
}

// DBSCAN聚类算法实现
static std::vector<int> dbscan(const std::vector<cv::Point2f>& points,
                               float eps, int min_samples) {
    std::vector<int> labels(points.size(), -1);  // -1表示未分类（噪声）
    int clusterId = 0;
    std::vector<bool> visited(points.size(), false);

    for (size_t i = 0; i < points.size(); ++i) {
        if (visited[i]) {
            continue;
        }

        visited[i] = true;
        std::vector<int> neighbors = regionQuery(points, static_cast<int>(i), eps);

        if (neighbors.size() < static_cast<size_t>(min_samples)) {
            // 标记为噪声（保持-1）
            continue;
        }

        // 创建新cluster
        labels[i] = clusterId;

        // 扩展cluster
        for (size_t j = 0; j < neighbors.size(); ++j) {
            int neighborIdx = neighbors[j];

            if (!visited[neighborIdx]) {
                visited[neighborIdx] = true;
                std::vector<int> neighborNeighbors = regionQuery(points, neighborIdx, eps);

                if (neighborNeighbors.size() >= static_cast<size_t>(min_samples)) {
                    // 将新邻居添加到neighbors列表
                    neighbors.insert(neighbors.end(), neighborNeighbors.begin(), neighborNeighbors.end());
                }
            }

            if (labels[neighborIdx] == -1) {
                labels[neighborIdx] = clusterId;
            }
        }

        clusterId++;
    }

    return labels;
}

std::vector<SpotDetectionResult> CornealSpotDetector::RemoveOutliers(
    const std::vector<SpotDetectionResult>& spots) {

    if (spots.size() < 3) {
        return spots;
    }

    // 提取光斑坐标点
    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < spots.size(); ++i) {
        points.push_back(spots[i].keypoint);
    }

    // 自动计算eps（基于平均距离的百分比）
    cv::Point2f center = ComputeGeometricCenter(spots);
    std::vector<float> distances;
    for (size_t i = 0; i < spots.size(); ++i) {
        float dx = spots[i].keypoint.x - center.x;
        float dy = spots[i].keypoint.y - center.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        distances.push_back(dist);
    }

    float mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    float sq_sum = 0.0f;
    for (size_t i = 0; i < distances.size(); ++i) {
        float d = distances[i];
        sq_sum += (d - mean_dist) * (d - mean_dist);
    }
    float std_dist = std::sqrt(sq_sum / distances.size());

    // eps设为平均距离的28-32%，适合环状结构
    float eps = mean_dist * 0.28f;
    // 如果标准差较大，说明有内外环，可以适当增大eps
    if (std_dist > mean_dist * 0.3f) {
        eps = mean_dist * 0.32f;
    }

    int min_samples = 3;  // DBSCAN最小样本数

    // 执行DBSCAN聚类
    std::vector<int> labels = dbscan(points, eps, min_samples);

    // 统计每个cluster的大小（排除噪声点 label=-1）
    std::map<int, int> cluster_sizes;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != -1) {
            cluster_sizes[labels[i]]++;
        }
    }

    if (cluster_sizes.empty()) {
        // 如果所有点都是噪声，返回原列表
        std::cout << "[CornealSpotDetector] [DBSCAN] 警告：所有点都被标记为噪声，返回所有点" << std::endl;
        return spots;
    }

    // 找到最大的两个cluster
    std::vector<std::pair<int, int>> sorted_clusters(cluster_sizes.begin(), cluster_sizes.end());
    std::sort(sorted_clusters.begin(), sorted_clusters.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.second > b.second;  // 按大小降序排序
              });

    std::set<int> top2_clusters;
    for (size_t i = 0; i < std::min(size_t(2), sorted_clusters.size()); ++i) {
        top2_clusters.insert(sorted_clusters[i].first);
    }

    // 只保留属于最大两个cluster的光斑
    std::vector<SpotDetectionResult> filtered;
    int noise_count = 0;
    int removed_cluster_count = 0;

    for (size_t i = 0; i < spots.size(); ++i) {
        if (labels[i] == -1) {
            noise_count++;
        } else if (top2_clusters.find(labels[i]) != top2_clusters.end()) {
            filtered.push_back(spots[i]);
        } else {
            removed_cluster_count++;
        }
    }

    std::cout << "[CornealSpotDetector] [DBSCAN] eps=" << eps
              << ", min_samples=" << min_samples << std::endl;
    std::cout << "[CornealSpotDetector] [DBSCAN] 发现 " << cluster_sizes.size()
              << " 个cluster，保留最大的2个" << std::endl;
    std::cout << "[CornealSpotDetector] [DBSCAN] 去除噪声点: " << noise_count
              << " 个, 去除其他cluster: " << removed_cluster_count << " 个" << std::endl;
    std::cout << "[CornealSpotDetector] [DBSCAN] 保留光斑: " << filtered.size() << " 个" << std::endl;

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

std::vector<SpotDetectionResult> CornealSpotDetector::RemoveDistantSpots(
    const std::vector<SpotDetectionResult>& spots,
    const cv::Point2f& center,
    float max_distance_ratio) {

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

    // 计算平均距离和标准差
    float mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    float sq_sum = 0.0f;
    for (size_t i = 0; i < distances.size(); ++i) {
        float d = distances[i];
        sq_sum += (d - mean_dist) * (d - mean_dist);
    }
    float std_dev = std::sqrt(sq_sum / distances.size());

    // 计算最大允许距离：平均距离 + max_distance_ratio * 标准差
    // 这样可以去除那些明显偏离内外环的光斑
    float max_allowed_distance = mean_dist + max_distance_ratio * std_dev;

    // 过滤掉距离过远的光斑
    std::vector<SpotDetectionResult> filtered;
    int removed_count = 0;
    for (size_t i = 0; i < spots.size(); ++i) {
        if (distances[i] <= max_allowed_distance) {
            filtered.push_back(spots[i]);
        } else {
            removed_count++;
        }
    }

    if (removed_count > 0) {
        std::cout << "[CornealSpotDetector] 去除离内外环很远的光斑: " << removed_count
                  << " 个 (平均距离=" << mean_dist << ", 最大允许距离="
                  << max_allowed_distance << ")" << std::endl;
    }

    return filtered;
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
        result.ellipse = ellipse;

        // 验证椭圆长宽比：如果长宽比不在合理范围内，标记为无效
        // 长宽比 = major_axis / minor_axis
        float aspect_ratio = (result.minor_axis > 0.0f) ?
                            (result.major_axis / result.minor_axis) : 0.0f;

        const float min_ellipse_aspect_ratio = 0.6f;  // 最小长宽比
        const float max_ellipse_aspect_ratio = 1.4f;  // 最大长宽比

        if (aspect_ratio < min_ellipse_aspect_ratio || aspect_ratio > max_ellipse_aspect_ratio) {
            std::cerr << "[CornealSpotDetector] 椭圆拟合长宽比异常: " << aspect_ratio
                      << " (范围: [" << min_ellipse_aspect_ratio << ", "
                      << max_ellipse_aspect_ratio << "])，标记为无效" << std::endl;
            result.valid = false;
        } else {
            result.valid = true;
            std::cout << "[CornealSpotDetector] 椭圆拟合成功: 长宽比=" << aspect_ratio << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[CornealSpotDetector] 椭圆拟合失败: " << e.what() << std::endl;
        result.valid = false;
    }

    return result;
}


#include <thread>
#include <atomic>
#include <chrono>

bool CornealSpotDetector::test_concurrent_inference(
        const cv::Mat& image,
        CornealSpotAnalysisResult& result,
        float conf_threshold,
        float nms_threshold,
        int min_spots,
        int max_spots,
        float min_aspect_ratio,
        float max_aspect_ratio,
        int enable_ellipse_fit) {

    if (image.empty()) {
        result.error_message = "输入图像为空";
        std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
        return false;
    }

    std::cout << "=== 开始并发推理测试 ===" << std::endl;
    std::cout << "使用 InferRequest 0 和 1 进行并发测试" << std::endl;
    std::cout << "图像大小: " << image.cols << "x" << image.rows << std::endl;

    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 用于同步的变量
    std::atomic<int> completed_threads{0};
    std::atomic<int> failed_threads{0};
    std::vector<std::string> thread_errors(2);
    std::vector<std::thread::id> thread_ids(2);

    // 线程函数
    auto inference_task = [&](int request_index, int thread_id) {
        std::stringstream ss;
        ss << "线程" << thread_id << " (TID: " << std::this_thread::get_id()
           << ", Request: " << request_index << ")";
        std::string thread_info = ss.str();

        thread_ids[thread_id] = std::this_thread::get_id();

        try {
            std::cout << thread_info << " 开始执行" << std::endl;

            // 获取对应的 InferRequest
            ov::InferRequest& infer_request = infer_requests_[request_index];

            // 预处理（每个线程独立进行）
            float scale;
            int left, top;
            cv::Mat preprocessed = Preprocess(image, scale, left, top);

            // 创建独立的 Tensor 数据
            ov::Shape input_shape = {1, static_cast<size_t>(input_size_),
                                     static_cast<size_t>(input_size_), 3};
            size_t data_size = input_size_ * input_size_ * 3 * sizeof(uint8_t);

            std::vector<uint8_t> input_data(data_size);
            std::memcpy(input_data.data(), preprocessed.data, data_size);

            ov::Tensor input_tensor(ov::element::u8, input_shape, input_data.data());

            // 设置输入
            infer_request.set_input_tensor(input_tensor);

            // 记录推理开始时间
            auto thread_start = std::chrono::high_resolution_clock::now();

            std::cout << thread_info << " 调用 infer()..." << std::endl;

            // 执行推理
            infer_request.infer();

            // 记录推理结束时间
            auto thread_end = std::chrono::high_resolution_clock::now();
            auto thread_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                thread_end - thread_start);

            std::cout << thread_info << " infer() 完成，耗时: "
                      << thread_duration.count() << "ms" << std::endl;

            // 获取输出结果（验证推理成功）
            auto output_tensor = infer_request.get_output_tensor();
            const float* output_data = output_tensor.data<float>();

            if (output_data) {
                std::cout << thread_info << " 推理成功，输出数据有效" << std::endl;
                completed_threads++;
            } else {
                thread_errors[thread_id] = thread_info + " 输出数据为空";
                failed_threads++;
            }

        } catch (const ov::Exception& e) {
            thread_errors[thread_id] = thread_info + " OpenVINO异常: " + e.what();
            failed_threads++;
            std::cerr << thread_errors[thread_id] << std::endl;
        } catch (const std::exception& e) {
            thread_errors[thread_id] = thread_info + " 异常: " + e.what();
            failed_threads++;
            std::cerr << thread_errors[thread_id] << std::endl;
        } catch (...) {
            thread_errors[thread_id] = thread_info + " 未知异常";
            failed_threads++;
            std::cerr << thread_errors[thread_id] << std::endl;
        }
    };

    // 创建并启动两个线程
    std::cout << "\n创建两个并发线程..." << std::endl;

    std::thread thread0([&]() { inference_task(0, 0); });
    std::thread thread1([&]() { inference_task(1, 1); });

    // 等待线程完成
    std::cout << "等待线程完成..." << std::endl;

    thread0.join();
    thread1.join();

    // 计算总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // 输出测试结果
    std::cout << "\n=== 并发测试结果 ===" << std::endl;
    std::cout << "总耗时: " << total_duration.count() << "ms" << std::endl;
    std::cout << "完成的线程: " << completed_threads << "/2" << std::endl;
    std::cout << "失败的线程: " << failed_threads << "/2" << std::endl;

    // 输出线程ID信息
    std::cout << "线程ID: " << std::endl;
    for (int i = 0; i < 2; ++i) {
        std::cout << "  线程" << i << ": " << thread_ids[i] << std::endl;
    }

    // 如果有错误，输出详细信息
    if (failed_threads > 0) {
        std::cout << "\n错误详情:" << std::endl;
        for (int i = 0; i < 2; ++i) {
            if (!thread_errors[i].empty()) {
                std::cout << "  " << thread_errors[i] << std::endl;
            }
        }

        result.error_message = "并发测试失败: " + std::to_string(failed_threads) + "个线程失败";
        std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
        return false;
    }

    std::cout << "✅ 并发测试成功！两个线程都完成了推理" << std::endl;
    return true;
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

    // 从对象池获取可用的 InferRequest（线程安全）
    int request_index = acquireInferRequest();
    std::cout << "[CornealSpotDetector] request index: " << request_index << std::endl;
    if(request_index < 0)
    {
        std::cerr << "[CornealSpotDetector] Error Request Index : " << request_index << std::endl;
        return false;
    }
    try {
        // 1. 预处理（Letterbox，保持 BGR uint8 格式）
        float scale;
        int left, top;

        cv::Mat preprocessed = Preprocess(image, scale, left, top);

        // 2. 创建 OpenVINO Tensor（使用深拷贝确保数据安全）
        // 输入形状：[1, H, W, C] (NHWC, BGR, uint8)
        ov::Shape input_shape = {1, static_cast<size_t>(input_size_),
                                 static_cast<size_t>(input_size_), 3};

        // 使用深拷贝方式创建Tensor，避免cv::Mat生命周期问题
        // 计算所需内存大小
        size_t data_size = input_size_ * input_size_ * 3 * sizeof(uint8_t);
        std::vector<uint8_t> input_data(data_size);

        // 深拷贝图像数据
        std::memcpy(input_data.data(), preprocessed.data, data_size);

        // 使用深拷贝的数据创建Tensor
        ov::Tensor input_tensor(ov::element::u8, input_shape, input_data.data());

        // 3. 获取当前线程专用的 InferRequest
        ov::InferRequest& infer_request = infer_requests_[request_index];

        // 4. 设置输入 Tensor
        infer_request.set_input_tensor(input_tensor);

        // 5. 运行推理（使用全局互斥锁保护，防止OpenVINO内部并发冲突）
        // 注意：虽然降低了并发性能，但这是为了避免OpenVINO内部的竞态条件导致崩溃
        {
            std::lock_guard<std::mutex> global_lock(global_inference_mutex_);
            std::cout << "[CornealSpotDetector] 开始推理 (request_index=" << request_index << ")" << std::endl;
            infer_request.infer();
            std::cout << "[CornealSpotDetector] 推理完成 (request_index=" << request_index << ")" << std::endl;
        }

        // 6. 获取输出 Tensor
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<float>();

        // 7. UNet后处理：从分割mask中提取轮廓重心（同时保存mask）
        cv::Mat segmentation_mask;
        std::vector<SpotDetectionResult> spots = PostprocessUNet(
            output_data, image.size(), scale, left, top, segmentation_mask, 0.5f);
        
        // 保存mask到结果中
        result.segmentation_mask = segmentation_mask;

        // 推理完成，立即释放 InferRequest 供其他线程使用
        releaseInferRequest(request_index);

        std::cout << "[CornealSpotDetector] UNet初始检测数量: " << spots.size() << std::endl;

        // 7. 长宽比筛选（UNet轮廓可能不规则，适当放宽长宽比限制）
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

        // 9. 数量检查和处理
        if (static_cast<int>(spots.size()) < min_spots) {
            result.error_message = "光斑数量过少 (" + std::to_string(spots.size()) +
                                  " < " + std::to_string(min_spots) + ")";
            std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
            return false;
        }

        // 如果光斑数量超过最大值，按距离几何中心的距离排序并取前max_spots个
        if (static_cast<int>(spots.size()) > max_spots) {
            std::cout << "[CornealSpotDetector] 光斑数量过多 (" << spots.size()
                      << " > " << max_spots << ")，按距离几何中心排序并取前" << max_spots << "个" << std::endl;

            // 计算几何中心
            cv::Point2f center = ComputeGeometricCenter(spots);

            // 按距离几何中心的距离排序（距离近的优先保留，更符合内外环分布）
            std::sort(spots.begin(), spots.end(),
                     [&center](const SpotDetectionResult& a, const SpotDetectionResult& b) {
                         float dist_a = std::sqrt((a.keypoint.x - center.x) * (a.keypoint.x - center.x) +
                                                  (a.keypoint.y - center.y) * (a.keypoint.y - center.y));
                         float dist_b = std::sqrt((b.keypoint.x - center.x) * (b.keypoint.x - center.x) +
                                                  (b.keypoint.y - center.y) * (b.keypoint.y - center.y));
                         return dist_a < dist_b;  // 升序：距离近的在前
                     });

            // 只保留前max_spots个
            spots.resize(max_spots);
            std::cout << "[CornealSpotDetector] 排序后保留: " << spots.size() << " 个光斑" << std::endl;
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

            // 11.5. 去除离内外环很远的光斑（在椭圆拟合之前）
            // 对内环和外环分别进行过滤（max_distance_ratio改为1.5）
            result.inner_spots = RemoveDistantSpots(result.inner_spots, result.geometric_center, 1.5f);
            result.outer_spots = RemoveDistantSpots(result.outer_spots, result.geometric_center, 1.5f);

            std::cout << "[CornealSpotDetector] 距离过滤后 - 内环: " << result.inner_spots.size()
                      << ", 外环: " << result.outer_spots.size() << std::endl;

            // 12. 椭圆拟合（内部会进行长宽比验证）
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
        // 异常情况下也要释放 InferRequest
        releaseInferRequest(request_index);

        result.error_message = std::string("推理失败: ") + e.what();
        std::cerr << "[CornealSpotDetector] " << result.error_message << std::endl;
        return false;
    }
}

// 辅助函数：绘制虚线椭圆（参考Python matplotlib的虚线效果）
static void drawDashedEllipse(cv::Mat& img, const cv::RotatedRect& ellipse,
                              const cv::Scalar& color, int thickness = 2,
                              int dashLength = 10, int gapLength = 5) {
    // 计算椭圆参数
    cv::Point2f center = ellipse.center;
    cv::Size2f size = ellipse.size;
    const float PI = 3.14159265358979323846f;
    float angle = ellipse.angle * PI / 180.0f;  // 转换为弧度

    // 生成椭圆上的点（足够密集以绘制平滑的虚线）
    int numPoints = 360;  // 360个点，每度一个点
    std::vector<cv::Point2f> points;
    points.reserve(numPoints);

    float a = size.width / 2.0f;   // 半长轴
    float b = size.height / 2.0f;  // 半短轴

    for (int i = 0; i < numPoints; ++i) {
        float theta = i * 2.0f * PI / numPoints;

        // 椭圆参数方程
        float x = a * std::cos(theta);
        float y = b * std::sin(theta);

        // 旋转
        float x_rot = x * std::cos(angle) - y * std::sin(angle);
        float y_rot = x * std::sin(angle) + y * std::cos(angle);

        // 平移
        points.push_back(cv::Point2f(center.x + x_rot, center.y + y_rot));
    }

    // 绘制虚线：间隔绘制线段
    int segmentLength = dashLength + gapLength;
    for (int i = 0; i < numPoints; i += segmentLength) {
        int endIdx = std::min(i + dashLength, numPoints - 1);
        for (int j = i; j < endIdx; ++j) {
            int nextIdx = (j + 1) % numPoints;
            cv::line(img, points[j], points[nextIdx], color, thickness);
        }
    }
}

cv::Mat CornealSpotDetector::DrawResult(const cv::Mat& image,
                                        const CornealSpotAnalysisResult& result) {
    cv::Mat vis = image.clone();

    if (!result.valid) {
        return vis;
    }

    // 1. 绘制内环光斑（绿色点）
    for (size_t i = 0; i < result.inner_spots.size(); ++i) {
        const SpotDetectionResult& spot = result.inner_spots[i];
        // 只绘制重心点，不绘制矩形框和标签
        cv::circle(vis, spot.keypoint, 3, cv::Scalar(0, 255, 0), -1);  // 绿色 (BGR)
    }

    // 2. 绘制外环光斑（洋红色点）
    for (size_t i = 0; i < result.outer_spots.size(); ++i) {
        const SpotDetectionResult& spot = result.outer_spots[i];
        // 只绘制重心点，不绘制矩形框和标签
        cv::circle(vis, spot.keypoint, 3, cv::Scalar(255, 0, 255), -1);  // 洋红色 (BGR)
    }

    // 3. 绘制内环椭圆（绿色虚线）
    if (result.inner_ellipse.valid) {
        cv::RotatedRect inner_rect(result.inner_ellipse.center,
                                   cv::Size2f(result.inner_ellipse.major_axis,
                                             result.inner_ellipse.minor_axis),
                                   result.inner_ellipse.angle);
        drawDashedEllipse(vis, inner_rect, cv::Scalar(0, 255, 0), 2);  // 绿色虚线
    }

    // 4. 绘制外环椭圆（洋红色虚线）
    if (result.outer_ellipse.valid) {
        cv::RotatedRect outer_rect(result.outer_ellipse.center,
                                   cv::Size2f(result.outer_ellipse.major_axis,
                                             result.outer_ellipse.minor_axis),
                                   result.outer_ellipse.angle);
        drawDashedEllipse(vis, outer_rect, cv::Scalar(255, 0, 255), 2);  // 洋红色虚线
    }

    // 5. 绘制几何中心（蓝色加号，参考Python的'b+'）
    cv::Point2f center = result.geometric_center;
    int crossSize = 15;  // 加号大小
    int thickness = 3;   // 线条粗细

    // 绘制水平线
    cv::line(vis,
             cv::Point(static_cast<int>(center.x - crossSize), static_cast<int>(center.y)),
             cv::Point(static_cast<int>(center.x + crossSize), static_cast<int>(center.y)),
             cv::Scalar(255, 0, 0), thickness);  // 蓝色 (BGR)

    // 绘制垂直线
    cv::line(vis,
             cv::Point(static_cast<int>(center.x), static_cast<int>(center.y - crossSize)),
             cv::Point(static_cast<int>(center.x), static_cast<int>(center.y + crossSize)),
             cv::Scalar(255, 0, 0), thickness);  // 蓝色 (BGR)

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

cv::Mat CornealSpotDetector::DrawMaskResult(const cv::Mat& image,
                                           const CornealSpotAnalysisResult& result) {
    cv::Mat vis = image.clone();
    
    if (!result.valid || result.segmentation_mask.empty()) {
        return vis;
    }
    
    // 将mask从模型输入尺寸还原到原图尺寸
    cv::Mat mask_resized;
    if (result.segmentation_mask.size() != image.size()) {
        cv::resize(result.segmentation_mask, mask_resized, image.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        mask_resized = result.segmentation_mask;
    }
    
    // 将mask转换为3通道，用于叠加显示
    cv::Mat mask_colored;
    cv::cvtColor(mask_resized, mask_colored, cv::COLOR_GRAY2BGR);
    
    // 创建彩色mask（绿色半透明）
    cv::Mat colored_mask = cv::Mat::zeros(mask_colored.size(), CV_8UC3);
    colored_mask.setTo(cv::Scalar(0, 255, 0), mask_resized);  // 绿色
    
    // 将mask叠加到原图上（半透明效果）
    cv::Mat overlay;
    vis.copyTo(overlay);
    overlay.setTo(cv::Scalar(0, 255, 0), mask_resized);  // 绿色mask区域
    cv::addWeighted(vis, 0.7, overlay, 0.3, 0, vis);  // 70%原图 + 30%mask
    
    // 在左上角添加说明文字
    cv::putText(vis, "UNet Segmentation Mask",
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
               cv::Scalar(0, 255, 0), 2);
    
    return vis;
}
