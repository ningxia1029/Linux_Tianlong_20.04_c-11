#include "yolo_pose_inferencer.h"
#include <iostream>
#include <algorithm>
#include <mutex>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

YoloPoseInferencer* YoloPoseInferencer::m_pInstance = nullptr;

YoloPoseInferencer* YoloPoseInferencer::getInstance() {
    if (m_pInstance == nullptr) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> locker(mutex);
        if (m_pInstance == nullptr) {
            m_pInstance = new YoloPoseInferencer();
        }
    }
    return m_pInstance;
}

YoloPoseInferencer::YoloPoseInferencer()
    : model_path_(),
      input_size_(640),
      initialized_(false) {}

bool YoloPoseInferencer::Initialize(const std::string& model_path, int input_size) {
    if (initialized_) {
        std::cout << "[YoloPoseInferencer] 已初始化，跳过重复初始化" << std::endl;
        return true;
    }
    
    model_path_ = model_path;
    input_size_ = input_size;
    
    try {
        std::cout << "[YoloPoseInferencer] 正在加载 OpenVINO 模型..." << std::endl;
        std::cout << "[YoloPoseInferencer] 模型路径: " << model_path_ << std::endl;
        
        // 检查文件是否存在
        std::ifstream file_check(model_path_);
        if (!file_check.good()) {
            std::cerr << "[YoloPoseInferencer] 错误: 模型文件不存在或无法访问: " << model_path_ << std::endl;
            std::cerr << "[YoloPoseInferencer] 提示: OpenVINO 需要 .xml 格式的模型文件" << std::endl;
            initialized_ = false;
            return false;
        }
        file_check.close();
        
        // 检查文件扩展名
        if (model_path_.substr(model_path_.find_last_of(".") + 1) != "xml") {
            std::cerr << "[YoloPoseInferencer] 警告: 模型文件扩展名不是 .xml，OpenVINO 需要 IR 格式 (.xml + .bin)" << std::endl;
        }
        
        // 1. 读取模型
        std::cout << "[YoloPoseInferencer] 步骤 1/4: 读取模型文件..." << std::endl;
        std::shared_ptr<ov::Model> model = core_.read_model(model_path_);
        std::cout << "[YoloPoseInferencer] 模型读取成功" << std::endl;
        
        // 2. 配置预处理（使用 PrePostProcessor API）
        std::cout << "[YoloPoseInferencer] 步骤 2/4: 配置预处理..." << std::endl;
        ov::preprocess::PrePostProcessor ppp(model);
        
        // 获取输入输出名称
        input_name_ = model->input().get_any_name();
        output_name_ = model->output().get_any_name();
        std::cout << "[YoloPoseInferencer] 输入名: " << input_name_ << ", 输出名: " << output_name_ << std::endl;
        
        // 配置输入
        // 输入格式：cv::Mat (HWC, BGR, uint8) -> 模型期望 (NCHW, RGB, float32)
        ppp.input().tensor()
            .set_element_type(ov::element::u8)           // 输入数据类型：uint8
            .set_layout("NHWC")                          // 输入布局：NHWC (批次, H, W, 通道)
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
        std::cout << "[YoloPoseInferencer] 构建预处理模型..." << std::endl;
        model = ppp.build();
        std::cout << "[YoloPoseInferencer] 预处理配置完成" << std::endl;
        
        // 3. 编译模型到 CPU 设备
        std::cout << "[YoloPoseInferencer] 步骤 3/4: 编译模型到 CPU..." << std::endl;
        compiled_model_ = core_.compile_model(model, "CPU");
        std::cout << "[YoloPoseInferencer] 模型编译成功" << std::endl;
        
        // 4. 创建推理请求
        std::cout << "[YoloPoseInferencer] 步骤 4/4: 创建推理请求..." << std::endl;
        infer_request_ = compiled_model_.create_infer_request();
        std::cout << "[YoloPoseInferencer] 推理请求创建成功" << std::endl;
        
        initialized_ = true;
        
        std::cout << "[YoloPoseInferencer] 模型已加载: " << model_path_ << std::endl;
        std::cout << "  输入名: " << input_name_ << std::endl;
        std::cout << "  输出名: " << output_name_ << std::endl;
        std::cout << "  输入尺寸: [1, " << input_size_ << ", " << input_size_ << ", 3]" << std::endl;
        std::cout << "  预处理已集成到模型图中（BGR->RGB, Normalize, Layout转换）" << std::endl;

        return true;
    } catch (const ov::Exception& e) {
        std::cerr << "[YoloPoseInferencer] OpenVINO 异常: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[YoloPoseInferencer] 初始化失败: " << e.what() << std::endl;
        std::cerr << "[YoloPoseInferencer] 可能的原因:" << std::endl;
        std::cerr << "  1. 模型文件路径错误或文件不存在" << std::endl;
        std::cerr << "  2. 模型文件格式错误（需要 .xml 格式，不是 .onnx）" << std::endl;
        std::cerr << "  3. OpenVINO 库未正确链接" << std::endl;
        std::cerr << "  4. 模型文件损坏或不完整" << std::endl;
        initialized_ = false;
        return false;
    }
}

cv::Mat YoloPoseInferencer::Preprocess(const cv::Mat& img,
                                       float& scale,
                                       int& left,
                                       int& top) const {
    // Letterbox Resize：保持宽高比缩放并填充灰边
    int h = img.rows;
    int w = img.cols;

    // 计算缩放比例
    scale = std::min(static_cast<float>(input_size_) / h,
                    static_cast<float>(input_size_) / w);
    
    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    // 缩放图像
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(new_w, new_h));

    // 计算填充
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

std::vector<DetectionResult> YoloPoseInferencer::Postprocess(
    const float* output,
    const cv::Size& img_size,
    float scale,
    int left,
    int top,
    float conf_threshold,
    float nms_threshold) const {
    
    // 输出形状：[1, 8, 8400]
    // 8个通道：[cx, cy, w, h, conf, kpt_x, kpt_y, kpt_conf]
    const int num_proposals = 8400;
    const int num_channels = 8;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kpt_confs;

    // 遍历所有预测（转置后的数据）
    for (int i = 0; i < num_proposals; ++i) {
        // output 是 [1, 8, 8400]，按列优先存储
        // 第 i 个proposal的数据在：output[0*8*8400 + channel*8400 + i]
        float conf = output[4 * num_proposals + i];  // 第5个通道（索引4）
        
        if (conf > conf_threshold) {
            float cx = output[0 * num_proposals + i];
            float cy = output[1 * num_proposals + i];
            float w = output[2 * num_proposals + i];
            float h = output[3 * num_proposals + i];
            float kpt_x = output[5 * num_proposals + i];
            float kpt_y = output[6 * num_proposals + i];
            float kpt_conf = output[7 * num_proposals + i];

            // 坐标还原到原图
            float x1 = (cx - w / 2.0f - left) / scale;
            float y1 = (cy - h / 2.0f - top) / scale;
            float w_orig = w / scale;
            float h_orig = h / scale;
            float kpt_x_orig = (kpt_x - left) / scale;
            float kpt_y_orig = (kpt_y - top) / scale;

            boxes.push_back(cv::Rect(static_cast<int>(x1),
                                    static_cast<int>(y1),
                                    static_cast<int>(w_orig),
                                    static_cast<int>(h_orig)));
            scores.push_back(conf);
            keypoints.push_back(cv::Point2f(kpt_x_orig, kpt_y_orig));
            kpt_confs.push_back(kpt_conf);
        }
    }

    // NMS（非极大值抑制）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    // 构建结果
    std::vector<DetectionResult> results;
    for (int idx : indices) {
        DetectionResult result;
        result.box = boxes[idx];
        result.score = scores[idx];
        result.keypoint = keypoints[idx];
        result.kpt_conf = kpt_confs[idx];
        results.push_back(result);
    }

    return results;
}

bool YoloPoseInferencer::SelectPupil(
    const std::vector<DetectionResult>& detections,
    int is_right_eye,
    cv::Point2f& pupil_point) const {
    
    if (detections.empty()) {
        return false;
    }

    if (detections.size() == 1) {
        // 只检测到1个瞳孔，直接返回
        pupil_point = detections[0].keypoint;
        return true;
    }

    // 检测到多个瞳孔，按规则选择
    if (is_right_eye == 1) {
        // 检测右眼：选择 x 坐标最小的
        auto it = std::min_element(detections.begin(), detections.end(),
            [](const DetectionResult& a, const DetectionResult& b) {
                return a.keypoint.x < b.keypoint.x;
            });
        pupil_point = it->keypoint;
    } else {
        // 检测左眼：选择 x 坐标最大的
        auto it = std::max_element(detections.begin(), detections.end(),
            [](const DetectionResult& a, const DetectionResult& b) {
                return a.keypoint.x < b.keypoint.x;
            });
        pupil_point = it->keypoint;
    }

    return true;
}

bool YoloPoseInferencer::InferPupil(
    const cv::Mat& img,
    int is_right_eye,
    cv::Point2f& pupil_point,
    float conf_threshold,
    float nms_threshold) {
    
    if (img.empty()) {
        std::cerr << "[YoloPoseInferencer] 错误：输入图像为空" << std::endl;
        return false;
    }

    try {
        // 1. 预处理（Letterbox，保持 BGR uint8 格式）
        float scale;
        int left, top;
        cv::Mat preprocessed = Preprocess(img, scale, left, top);

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
        std::vector<DetectionResult> detections = Postprocess(
            output_data, img.size(), scale, left, top,
            conf_threshold, nms_threshold);

        // 7. 选择瞳孔
        return SelectPupil(detections, is_right_eye, pupil_point);

    } catch (const std::exception& e) {
        std::cerr << "[YoloPoseInferencer] 推理失败: " << e.what() << std::endl;
        return false;
    }
}



