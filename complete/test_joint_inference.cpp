#include "corneal_joint_inferencer.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <limits.h>
#include <string>

int main(int argc, char** argv) {
    // 配置参数
    std::string pupil_model_path = "pupil_detect.xml";  // 瞳孔检测模型
    std::string corneal_model_path = "corneal_curvature.xml";  // 角膜光斑检测模型
    std::string image_path = "OD/cc17.bmp";  // 测试图像
    
    // 支持命令行参数
    if (argc >= 2) {
        image_path = argv[1];
    }
    if (argc >= 3) {
        pupil_model_path = argv[2];
    }
    if (argc >= 4) {
        corneal_model_path = argv[3];
    }
    
    std::cout << "=== 角膜联合推理测试 (OpenVINO) ===" << std::endl;
    std::cout << "瞳孔检测模型: " << pupil_model_path << std::endl;
    std::cout << "角膜光斑模型: " << corneal_model_path << std::endl;
    std::cout << "测试图像: " << image_path << std::endl;
    
    // 1. 初始化瞳孔检测器
    std::cout << "\n[1/3] 初始化瞳孔检测器..." << std::endl;
    PupilDetector* pupil_detector = PupilDetector::GetInstance();
    if (!pupil_detector->Initialize(pupil_model_path, 640, 4)) {
        std::cerr << "瞳孔检测器初始化失败！" << std::endl;
        return -1;
    }
    std::cout << "瞳孔检测器初始化成功" << std::endl;
    
    // 2. 初始化角膜光斑检测器（输入尺寸会自动从模型检测）
    std::cout << "\n[2/3] 初始化角膜光斑检测器..." << std::endl;
    CornealSpotDetector* spot_detector = CornealSpotDetector::GetInstance();
    // 传入0表示自动检测，或者传入任意值都会被模型实际尺寸覆盖
    if (!spot_detector->Initialize(corneal_model_path, 0, 4)) {
        std::cerr << "角膜光斑检测器初始化失败！" << std::endl;
        return -1;
    }
    std::cout << "角膜光斑检测器初始化成功" << std::endl;
    
    // 3. 读取测试图像
    std::cout << "\n[3/3] 读取测试图像..." << std::endl;
    // 解析图像路径（使用与模型路径相同的解析逻辑）
    std::string resolved_image_path = image_path;
    if (image_path[0] != '/') {
        // 相对路径：尝试从可执行文件位置找到项目根目录
        char exe_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len != -1) {
            exe_path[len] = '\0';
            std::string exe_dir = std::string(exe_path);
            size_t last_slash = exe_dir.find_last_of('/');
            if (last_slash != std::string::npos) {
                exe_dir = exe_dir.substr(0, last_slash);
                if (exe_dir.find("/build/bin") != std::string::npos || 
                    exe_dir.find("/build/") != std::string::npos) {
                    size_t build_pos = exe_dir.find("/build");
                    if (build_pos != std::string::npos) {
                        std::string project_root = exe_dir.substr(0, build_pos);
                        resolved_image_path = project_root + "/" + image_path;
                    }
                }
            }
        }
        // 如果还是找不到，尝试当前目录的上一级
        if (!cv::imread(resolved_image_path).data) {
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                std::string parent_path = std::string(cwd);
                size_t last_slash = parent_path.find_last_of('/');
                if (last_slash != std::string::npos) {
                    parent_path = parent_path.substr(0, last_slash);
                    resolved_image_path = parent_path + "/" + image_path;
                }
            }
        }
    }
    
    cv::Mat image = cv::imread(resolved_image_path);
    if (image.empty()) {
        std::cerr << "错误：无法读取图像 " << resolved_image_path << std::endl;
        std::cerr << "尝试的路径: " << resolved_image_path << std::endl;
        return -1;
    }
    std::cout << "图像尺寸: " << image.cols << " x " << image.rows << std::endl;
    
    PupilDetectionResult pupil_result;
    // 4. 瞳孔检测
    /*
    std::cout << "\n=== 步骤 1: 瞳孔检测 ===" << std::endl;
    PupilDetectionResult pupil_result;
    bool pupil_success = pupil_detector->DetectPupilAdaptive(image, pupil_result, 0.25f, 0.45f);
    
    if (!pupil_success) {
        std::cerr << "瞳孔检测失败！" << std::endl;
        return -1;
    }
    
    std::cout << "瞳孔检测成功:" << std::endl;
    std::cout << "  中心: (" << pupil_result.center.x << ", " << pupil_result.center.y << ")" << std::endl;
    std::cout << "  半宽: " << pupil_result.half_width << std::endl;
    std::cout << "  置信度: " << pupil_result.confidence << std::endl;
    */
    // 5. 角膜光斑检测与分析
    std::cout << "\n=== 步骤 2: 角膜光斑检测 ===" << std::endl;
    CornealSpotAnalysisResult spot_result;
    
    // 检测参数
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int min_spots = 29;  // 最少光斑数（调整为更宽松的值）
    int max_spots = 32;  // 最多光斑数
    float min_aspect_ratio = 0.5f;  // 最小长宽比（放宽限制）
    float max_aspect_ratio = 2.0f;  // 最大长宽比（放宽限制）
    int enable_ellipse_fit = 1;     // 启用椭圆拟合
    
    pupil_result.cropped_image = image;
    bool spot_success = spot_detector->DetectAndAnalyze(
        pupil_result.cropped_image,  // 使用裁剪后的640x640图像
        spot_result,
        conf_threshold,
        nms_threshold,
        min_spots,
        max_spots,
        min_aspect_ratio,
        max_aspect_ratio,
        enable_ellipse_fit
    );
    
    if (!spot_success) {
        std::cerr << "角膜光斑检测失败: " << spot_result.error_message << std::endl;
        return -1;
    }
    
    std::cout << "角膜光斑检测成功:" << std::endl;
    std::cout << "  光斑数量: " << spot_result.num_spots << std::endl;
    std::cout << "  几何中心: (" << spot_result.geometric_center.x << ", " 
              << spot_result.geometric_center.y << ")" << std::endl;
    std::cout << "  平均长宽比: " << spot_result.avg_aspect_ratio << std::endl;
    
    if (spot_result.inner_ellipse.valid) {
        std::cout << "  内环椭圆: 长轴=" << spot_result.inner_ellipse.major_axis
                  << ", 短轴=" << spot_result.inner_ellipse.minor_axis << std::endl;
    }
    
    if (spot_result.outer_ellipse.valid) {
        std::cout << "  外环椭圆: 长轴=" << spot_result.outer_ellipse.major_axis
                  << ", 短轴=" << spot_result.outer_ellipse.minor_axis << std::endl;
    }
    
    // 6. 可视化结果
    std::cout << "\n=== 保存可视化结果 ===" << std::endl;
    
    // 绘制瞳孔检测结果
    /*
    cv::Mat pupil_vis = pupil_detector->DrawResult(image, pupil_result);
    cv::imwrite("output_pupil.png", pupil_vis);
    std::cout << "瞳孔检测结果已保存到: output_pupil.png" << std::endl;
    */
    // 绘制角膜光斑检测结果（完整版）
    cv::Mat spot_vis = spot_detector->DrawResult(pupil_result.cropped_image, spot_result);
    cv::imwrite("output_spots_full.png", spot_vis);
    std::cout << "角膜光斑检测结果（完整）已保存到: output_spots_full.png" << std::endl;
    
    // 绘制角膜光斑检测结果（简化版）
    cv::Mat spot_vis_simple = spot_detector->DrawSimpleResult(pupil_result.cropped_image, spot_result);
    cv::imwrite("output_spots_simple.png", spot_vis_simple);
    std::cout << "角膜光斑检测结果（简化）已保存到: output_spots_simple.png" << std::endl;
    
    // 绘制语义分割mask结果
    cv::Mat mask_vis = spot_detector->DrawMaskResult(pupil_result.cropped_image, spot_result);
    cv::imwrite("output_segmentation_mask.png", mask_vis);
    std::cout << "语义分割mask结果已保存到: output_segmentation_mask.png" << std::endl;
    
    std::cout << "\n=== 联合推理测试完成 ===" << std::endl;
    std::cout << "瞳孔检测: 成功" << std::endl;
    std::cout << "光斑检测: 成功" << std::endl;
    
    return 0;
}
