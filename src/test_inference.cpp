#include "yolo_pose_inferencer.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
    // 配置参数
    std::string model_path = "../../binocular_pupil_detect.xml";  // OpenVINO IR 格式 (.xml)
    std::string image_path = "../../test/0.png";  // 拼接的双目图像
    bool use_stereo = true;  // 是否使用双目推理（分别推理左右图）

    // 支持命令行参数
    if (argc >= 2) {
        image_path = argv[1];
    }
    if (argc >= 3) {
        model_path = argv[2];
    }
    if (argc >= 4) {
        use_stereo = (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
    }

    std::cout << "=== YOLOv8 Pose 瞳孔检测 C++ 测试 (OpenVINO) ===" << std::endl;
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "图像路径: " << image_path << std::endl;
    std::cout << "使用双目推理: " << (use_stereo ? "是" : "否") << std::endl;

    // 1. 初始化推理引擎
    YoloPoseInferencer* inferencer = YoloPoseInferencer::getInstance();
    if (!inferencer->Initialize(model_path, 640)) {
        std::cerr << "模型初始化失败！" << std::endl;
        return -1;
    }

    // 2. 读取双目图像
    cv::Mat full_img = cv::imread(image_path);
    if (full_img.empty()) {
        std::cerr << "错误：无法读取图像 " << image_path << std::endl;
        return -1;
    }

    int height = full_img.rows;
    int width = full_img.cols;
    int mid_point = width / 2;

    std::cout << "图像尺寸: " << width << " x " << height << std::endl;
    std::cout << "拆分中点: " << mid_point << std::endl;

    // 3. 拆分左右视图
    cv::Mat img_left = full_img(cv::Rect(0, 0, mid_point, height)).clone();
    cv::Mat img_right = full_img(cv::Rect(mid_point, 0, width - mid_point, height)).clone();

    cv::Point2f left_pupil, right_pupil;
    bool left_success = false, right_success = false;


    // 分别推理左右眼
    // 4. 推理左眼（is_right_eye = 0）
    std::cout << "\n正在推理左眼..." << std::endl;
    left_success = inferencer->InferPupil(img_left, 0, left_pupil);

    // 5. 推理右眼（is_right_eye = 1）
    std::cout << "\n正在推理右眼..." << std::endl;
    right_success = inferencer->InferPupil(img_right, 1, right_pupil);
    

    // 6. 显示结果并绘制
    if (left_success) {
        std::cout << "左眼瞳孔坐标: (" << left_pupil.x << ", " << left_pupil.y << ")" << std::endl;
        
        // 在原图上绘制（左视图坐标）
        cv::circle(full_img, 
                  cv::Point(static_cast<int>(left_pupil.x), 
                           static_cast<int>(left_pupil.y)),
                  5, cv::Scalar(0, 0, 255), -1);  // 红色实心圆
        cv::putText(full_img, "Left", 
                   cv::Point(static_cast<int>(left_pupil.x) + 10,
                            static_cast<int>(left_pupil.y) - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    } else {
        std::cout << "左眼未检测到瞳孔" << std::endl;
    }

    if (right_success) {
        std::cout << "右眼瞳孔坐标: (" << right_pupil.x << ", " << right_pupil.y << ")" << std::endl;
        
        // 在原图上绘制（需要加上偏移量）
        int right_x = static_cast<int>(right_pupil.x) + mid_point;
        int right_y = static_cast<int>(right_pupil.y);
        cv::circle(full_img, cv::Point(right_x, right_y),
                  5, cv::Scalar(0, 0, 255), -1);  // 红色实心圆
        cv::putText(full_img, "Right",
                   cv::Point(right_x + 10, right_y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    } else {
        std::cout << "右眼未检测到瞳孔" << std::endl;
    }

    // 7. 保存结果（无 GUI 环境，不使用 imshow/waitKey）
    std::cout << "\n=== 检测完成 ===" << std::endl;
    std::cout << "左眼检测: " << (left_success ? "成功" : "失败") << std::endl;
    std::cout << "右眼检测: " << (right_success ? "成功" : "失败") << std::endl;
    
    // 保存结果
    std::string output_path = "output_result.png";
    cv::imwrite(output_path, full_img);
    std::cout << "结果已保存到: " << output_path << std::endl;

    return (left_success && right_success) ? 0 : 1;
}

