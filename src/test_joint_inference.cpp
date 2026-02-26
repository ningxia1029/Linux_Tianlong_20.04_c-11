#include "corneal_joint_inferencer.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <limits.h>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>

// 辅助函数：解析图像路径（使用与模型路径相同的解析逻辑）
static std::string resolve_image_path(const std::string& image_path) {
    std::string resolved_path = image_path;
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
                        resolved_path = project_root + "/" + image_path;
                    }
                }
            }
        }
        // 如果还是找不到，尝试当前目录的上一级
        if (!cv::imread(resolved_path).data) {
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                std::string parent_path = std::string(cwd);
                size_t last_slash = parent_path.find_last_of('/');
                if (last_slash != std::string::npos) {
                    parent_path = parent_path.substr(0, last_slash);
                    resolved_path = parent_path + "/" + image_path;
                }
            }
        }
    }
    return resolved_path;
}

// 辅助函数：获取目录中的所有图片文件
static std::vector<std::string> get_image_files(const std::string& dir_path) {
    std::vector<std::string> image_files;
    std::vector<std::string> extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"};

    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) {
        std::cerr << "无法打开目录: " << dir_path << std::endl;
        return image_files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename == "." || filename == "..") {
            continue;
        }

        std::string full_path = dir_path + "/" + filename;

        // 检查是否为文件
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            // 检查扩展名
            std::string lower_filename = filename;
            std::transform(lower_filename.begin(), lower_filename.end(), lower_filename.begin(), ::tolower);
            for (const auto& ext : extensions) {
                if (lower_filename.length() >= ext.length() &&
                    lower_filename.substr(lower_filename.length() - ext.length()) == ext) {
                    image_files.push_back(full_path);
                    break;
                }
            }
        }
    }
    closedir(dir);

    // 排序文件名
    std::sort(image_files.begin(), image_files.end());

    return image_files;
}

// 辅助函数：创建目录（如果不存在）
static bool create_directory(const std::string& dir_path) {
    struct stat st;
    if (stat(dir_path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return true;  // 目录已存在
        } else {
            std::cerr << "路径已存在但不是目录: " << dir_path << std::endl;
            return false;
        }
    }

    // 创建目录
    if (mkdir(dir_path.c_str(), 0755) != 0) {
        std::cerr << "无法创建目录: " << dir_path << std::endl;
        return false;
    }

    return true;
}

// 辅助函数：从文件路径提取文件名（不含扩展名）
static std::string get_filename_without_ext(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of('/');
    size_t last_dot = filepath.find_last_of('.');

    // 提取完整文件名（含扩展名）
    std::string filename;
    if (last_slash != std::string::npos) {
        filename = filepath.substr(last_slash + 1);
    } else {
        filename = filepath;
    }

    // 去除扩展名
    if (last_dot != std::string::npos &&
        (last_slash == std::string::npos || last_dot > last_slash)) {
        // 计算在filename中的相对位置
        size_t dot_pos_in_filename = last_dot - (last_slash == std::string::npos ? 0 : last_slash + 1);
        filename = filename.substr(0, dot_pos_in_filename);
    }

    return filename;
}

int main(int argc, char** argv) {
    // 配置参数
    std::string pupil_model_path = "pupil_detect.xml";  // 瞳孔检测模型
    std::string corneal_model_path = "corneal_curvature.xml";  // 角膜光斑检测模型
    std::string image_input = "test";  // 测试图像或目录

    // 支持命令行参数
    // 用法: ./test_joint_inference [图像路径或目录] [瞳孔模型路径] [角膜模型路径]
    if (argc >= 2) {
        image_input = argv[1];
    }
    if (argc >= 3) {
        pupil_model_path = argv[2];
    }
    if (argc >= 4) {
        corneal_model_path = argv[3];
    }

    std::cout << "=== 角膜联合推理测试 (OpenVINO) - 批量处理模式 ===" << std::endl;
    std::cout << "瞳孔检测模型: " << pupil_model_path << std::endl;
    std::cout << "角膜光斑模型: " << corneal_model_path << std::endl;
    std::cout << "输入路径: " << image_input << std::endl;

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

    // 3. 确定输入是文件还是目录，收集所有图片文件
    std::cout << "\n[3/3] 收集图片文件..." << std::endl;
    std::vector<std::string> image_files;

    std::string resolved_input = resolve_image_path(image_input);

    // 检查是文件还是目录
    struct stat st;
    if (stat(resolved_input.c_str(), &st) != 0) {
        std::cerr << "错误：无法访问路径: " << resolved_input << std::endl;
        return -1;
    }

    if (S_ISDIR(st.st_mode)) {
        // 是目录，获取所有图片文件
        std::cout << "检测到输入为目录，扫描图片文件..." << std::endl;
        image_files = get_image_files(resolved_input);
        std::cout << "找到 " << image_files.size() << " 张图片" << std::endl;
    } else if (S_ISREG(st.st_mode)) {
        // 是单个文件
        image_files.push_back(resolved_input);
        std::cout << "检测到输入为单个文件" << std::endl;
    } else {
        std::cerr << "错误：输入路径既不是文件也不是目录: " << resolved_input << std::endl;
        return -1;
    }

    if (image_files.empty()) {
        std::cerr << "错误：未找到任何图片文件" << std::endl;
        return -1;
    }

    // 4. 创建输出目录及子目录
    std::string output_dir = "output";
    if (!create_directory(output_dir)) {
        std::cerr << "警告：无法创建输出目录，将使用当前目录" << std::endl;
        output_dir = ".";
    }
    std::string output_spots_dir = output_dir + "/spots_full";
    std::string output_mask_dir  = output_dir + "/segmentation_mask";
    create_directory(output_spots_dir);
    create_directory(output_mask_dir);
    std::cout << "输出目录: " << output_dir << std::endl;
    std::cout << "  spots_full       -> " << output_spots_dir << std::endl;
    std::cout << "  segmentation_mask-> " << output_mask_dir  << std::endl;

    // 5. 批量处理图片
    std::cout << "\n=== 开始批量处理 " << image_files.size() << " 张图片 ===" << std::endl;

    int success_count = 0;
    int fail_count = 0;

    for (size_t img_idx = 0; img_idx < image_files.size(); ++img_idx) {
        const std::string& image_path = image_files[img_idx];
        std::cout << "\n[" << (img_idx + 1) << "/" << image_files.size() << "] 处理: " << image_path << std::endl;

        // 读取图像
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "  错误：无法读取图像 " << image_path << std::endl;
            fail_count++;
            continue;
        }
        std::cout << "  图像尺寸: " << image.cols << " x " << image.rows << std::endl;

        // 4. 瞳孔检测

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
        // 调试：查看裁剪后的图像是否有效
        std::cout << "  cropped_image empty: " << (pupil_result.cropped_image.empty() ? 1 : 0)
                  << ", size: " << pupil_result.cropped_image.cols
                  << " x " << pupil_result.cropped_image.rows << std::endl;

        // 角膜光斑检测与分析
        CornealSpotAnalysisResult spot_result;

        // 检测参数
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;
        int min_spots = 25;
        int max_spots = 32;
        float min_aspect_ratio = 0.5f;
        float max_aspect_ratio = 2.0f;
        int enable_ellipse_fit = 1;

        // 如果裁剪图像为空，则回退使用原图，避免后续“输入图像为空”错误
        const cv::Mat& corneal_input = pupil_result.cropped_image.empty() ? image : pupil_result.cropped_image;
        if (pupil_result.cropped_image.empty()) {
            std::cout << "  警告：pupil_result.cropped_image 为空，改用原图进行角膜光斑检测" << std::endl;
        }

        bool spot_success = spot_detector->DetectAndAnalyze(
            corneal_input,  // 使用裁剪后的640x640图像（若为空则用原图）
            spot_result,
            conf_threshold,
            nms_threshold,
            min_spots,
            max_spots,
            min_aspect_ratio,
            max_aspect_ratio,
            enable_ellipse_fit
        );

        // 生成输出文件名（基于原始文件名）
        std::string base_filename = get_filename_without_ext(image_path);

        // 可视化时使用与检测输入一致的图像（裁剪图或原图）
        const cv::Mat& vis_image = corneal_input;

        // 无论成功还是失败，都保存 segmentation_mask
        cv::Mat mask_vis = spot_detector->DrawMaskResult(vis_image, spot_result);
        std::string output_mask = output_mask_dir + "/" + base_filename + "_segmentation_mask.png";
        cv::imwrite(output_mask, mask_vis);
        std::cout << "  已保存mask: " << output_mask << std::endl;

        if (!spot_success) {
            std::cerr << "  角膜光斑检测失败: " << spot_result.error_message << std::endl;
            fail_count++;
            continue;
        }

        std::cout << "  检测成功 - 光斑数量: " << spot_result.num_spots << std::endl;
        std::cout << "  几何中心(鲁棒RANSAC): (" << spot_result.geometric_center.x
                  << ", " << spot_result.geometric_center.y << ")" << std::endl;
        std::cout << "  平均长宽比: " << spot_result.avg_aspect_ratio << std::endl;

        if (spot_result.inner_ellipse.valid) {
            std::cout << "  内环椭圆: 长轴=" << spot_result.inner_ellipse.major_axis
                      << ", 短轴=" << spot_result.inner_ellipse.minor_axis << std::endl;
            std::cout << "  内环光斑数量: " << spot_result.inner_spots.size() << std::endl;
        } else {
            std::cout << "  内环椭圆: 无效" << std::endl;
        }

        if (spot_result.outer_ellipse.valid) {
            std::cout << "  外环椭圆: 长轴=" << spot_result.outer_ellipse.major_axis
                      << ", 短轴=" << spot_result.outer_ellipse.minor_axis << std::endl;
            std::cout << "  外环光斑数量: " << spot_result.outer_spots.size() << std::endl;
        } else {
            std::cout << "  外环椭圆: 无效" << std::endl;
        }

        // 保存 spots_full 结果
        cv::Mat spot_vis = spot_detector->DrawResult(vis_image, spot_result);
        std::string output_full = output_spots_dir + "/" + base_filename + "_spots_full.png";
        cv::imwrite(output_full, spot_vis);
        std::cout << "  已保存spots: " << output_full << std::endl;

        success_count++;
    }

    std::cout << "\n=== 批量处理完成 ===" << std::endl;
    std::cout << "成功: " << success_count << " 张" << std::endl;
    std::cout << "失败: " << fail_count << " 张" << std::endl;
    std::cout << "输出目录: " << output_dir << std::endl;

    return (fail_count == 0) ? 0 : -1;
}
