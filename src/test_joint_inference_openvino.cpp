#include "corneal_joint_inferencer_openvino.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "角膜光斑联合推理测试程序 (OpenVINO)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ========== 配置参数 ==========
    // 模型路径（OpenVINO .xml 文件）
    // Linux 路径示例
    std::string pupil_model_path = "../../pupil_detect.xml";
    std::string spot_model_path = "../../corneal_curvature.xml";
    
    // Windows 路径示例（如果在 Windows 上运行，请取消注释并修改路径）
    // std::string pupil_model_path = "E:\\deeplearning_tianlong\\Tianlong_main_pupil\\runs\\detect\\train7\\weights\\best_int8_openvino_model\\best.xml";
    // std::string spot_model_path = "E:\\deeplearning_tianlong\\Tianlong_corneal_curvature\\runs\\pose\\train11\\weights\\best_int8_openvino_model\\best.xml";
    
    // 测试图片路径
    std::string test_image_path = "../../test/images/output_153820.jpg";
    
    // Windows 路径示例（如果在 Windows 上运行，请取消注释并修改路径）
    // std::string test_image_path = "E:\\deeplearning_tianlong\\Tianlong_main_pupil\\data\\test\\images\\output_153820.jpg";
    
    // 输出目录
    std::string output_dir = "../runs/cpp_inference";
    
    // 检测参数
    float pupil_conf = 0.25f;
    float pupil_nms = 0.45f;
    
    float spot_conf = 0.25f;
    float spot_nms = 0.45f;
    int min_spots = 20;
    int max_spots = 35;
    float min_aspect = 0.5f;
    float max_aspect = 2.0f;
    int enable_ellipse_fit = 1;  // 是否进行椭圆拟合（0=不拟合，1=拟合）
    // =============================
    
    // 解析命令行参数（可选）
    if (argc > 1) {
        test_image_path = argv[1];
    }
    if (argc > 2) {
        pupil_model_path = argv[2];
    }
    if (argc > 3) {
        spot_model_path = argv[3];
    }
    
    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  瞳孔模型: " << pupil_model_path << std::endl;
    std::cout << "  光斑模型: " << spot_model_path << std::endl;
    std::cout << "  测试图片: " << test_image_path << std::endl;
    std::cout << "  输出目录: " << output_dir << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. 初始化瞳孔检测器（单例模式）
        std::cout << "========== 步骤 1: 初始化瞳孔检测器 ==========" << std::endl;
        PupilDetector* pupil_detector = PupilDetector::GetInstance();
        if (!pupil_detector->Initialize(pupil_model_path)) {
            std::cerr << "错误：瞳孔检测器初始化失败！" << std::endl;
            return -1;
        }
        
        // 2. 初始化光斑检测器（单例模式）
        std::cout << "\n========== 步骤 2: 初始化光斑检测器 ==========" << std::endl;
        CornealSpotDetector* spot_detector = CornealSpotDetector::GetInstance();
        if (!spot_detector->Initialize(spot_model_path)) {
            std::cerr << "错误：光斑检测器初始化失败！" << std::endl;
            return -1;
        }
        
        // 3. 读取测试图片
        std::cout << "\n========== 步骤 3: 读取测试图片 ==========" << std::endl;
        
        cv::Mat test_image;
        std::string image_name = "test";
        
        // 简单判断：如果路径包含 .jpg/.png 等后缀，视为文件；否则视为目录
        if (test_image_path.find(".jpg") != std::string::npos ||
            test_image_path.find(".png") != std::string::npos ||
            test_image_path.find(".bmp") != std::string::npos) {
            // 直接读取文件
            test_image = cv::imread(test_image_path);
            size_t last_slash = test_image_path.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                image_name = test_image_path.substr(last_slash + 1);
                size_t dot = image_name.find_last_of('.');
                if (dot != std::string::npos) {
                    image_name = image_name.substr(0, dot);
                }
            }
        } else {
            // 视为目录，尝试读取第一张图片
            std::string img_path = test_image_path + "/output_153820.jpg";
            test_image = cv::imread(img_path);
            if (test_image.empty()) {
                // 尝试其他文件
                img_path = test_image_path + "/output_153929.jpg";
                test_image = cv::imread(img_path);
            }
            image_name = "output_153820";
        }
        
        if (test_image.empty()) {
            std::cerr << "错误：无法读取测试图片！" << std::endl;
            std::cerr << "请检查路径: " << test_image_path << std::endl;
            return -1;
        }
        
        std::cout << "成功读取图片: " << test_image_path << std::endl;
        std::cout << "图片尺寸: " << test_image.cols << "x" << test_image.rows << std::endl;
        
        // 4. 瞳孔检测
        std::cout << "\n========== 步骤 4: 瞳孔检测 ==========" << std::endl;
        PupilDetectionResult pupil_result;
        bool pupil_ok = pupil_detector->DetectPupil(test_image, pupil_result, 
                                                   pupil_conf, pupil_nms);
        
        if (!pupil_ok) {
            std::cerr << "错误：瞳孔检测失败！" << std::endl;
            return -1;
        }
        
        // 保存瞳孔检测可视化结果
        cv::Mat pupil_vis = pupil_detector->DrawResult(test_image, pupil_result);
        std::string pupil_output = output_dir + "/" + image_name + "_pupil.jpg";
        cv::imwrite(pupil_output, pupil_vis);
        std::cout << "瞳孔检测可视化已保存: " << pupil_output << std::endl;
        
        // 5. 光斑检测与分析
        std::cout << "\n========== 步骤 5: 光斑检测与分析 ==========" << std::endl;
        CornealSpotAnalysisResult spot_result;
        bool spot_ok = spot_detector->DetectAndAnalyze(
            pupil_result.cropped_image, spot_result,
            spot_conf, spot_nms, min_spots, max_spots, 
            min_aspect, max_aspect, enable_ellipse_fit);
        
        if (!spot_ok) {
            std::cerr << "错误：光斑检测失败！" << std::endl;
            std::cerr << "原因: " << spot_result.error_message << std::endl;
            return -1;
        }
        
        // 保存光斑检测可视化结果
        cv::Mat spot_vis = spot_detector->DrawResult(pupil_result.cropped_image, spot_result);
        std::string spot_output = output_dir + "/" + image_name + "_spots.jpg";
        cv::imwrite(spot_output, spot_vis);
        std::cout << "光斑检测可视化已保存: " << spot_output << std::endl;
        
        // 6. 输出最终结果
        std::cout << "\n========== 最终结果 ==========" << std::endl;
        std::cout << "【瞳孔检测】" << std::endl;
        std::cout << "  中心坐标: (" << pupil_result.center.x << ", " 
                  << pupil_result.center.y << ")" << std::endl;
        std::cout << "  半宽: " << pupil_result.half_width << std::endl;
        std::cout << "  置信度: " << pupil_result.confidence << std::endl;
        
        std::cout << "\n【光斑分析】" << std::endl;
        std::cout << "  有效光斑数量: " << spot_result.num_spots << std::endl;
        std::cout << "  几何中心: (" << spot_result.geometric_center.x << ", " 
                  << spot_result.geometric_center.y << ")" << std::endl;
        std::cout << "  平均长宽比: " << spot_result.avg_aspect_ratio << std::endl;
        
        std::cout << "\n  内环光斑数量: " << spot_result.inner_spots.size() << std::endl;
        if (spot_result.inner_ellipse.valid) {
            std::cout << "  内环椭圆:" << std::endl;
            std::cout << "    中心: (" << spot_result.inner_ellipse.center.x << ", " 
                      << spot_result.inner_ellipse.center.y << ")" << std::endl;
            std::cout << "    长轴: " << spot_result.inner_ellipse.major_axis << std::endl;
            std::cout << "    短轴: " << spot_result.inner_ellipse.minor_axis << std::endl;
            std::cout << "    角度: " << spot_result.inner_ellipse.angle << "°" << std::endl;
        }
        
        std::cout << "\n  外环光斑数量: " << spot_result.outer_spots.size() << std::endl;
        if (spot_result.outer_ellipse.valid) {
            std::cout << "  外环椭圆:" << std::endl;
            std::cout << "    中心: (" << spot_result.outer_ellipse.center.x << ", " 
                      << spot_result.outer_ellipse.center.y << ")" << std::endl;
            std::cout << "    长轴: " << spot_result.outer_ellipse.major_axis << std::endl;
            std::cout << "    短轴: " << spot_result.outer_ellipse.minor_axis << std::endl;
            std::cout << "    角度: " << spot_result.outer_ellipse.angle << "°" << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "测试完成！" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n错误：程序异常退出" << std::endl;
        std::cerr << "异常信息: " << e.what() << std::endl;
        return -1;
    }
}

