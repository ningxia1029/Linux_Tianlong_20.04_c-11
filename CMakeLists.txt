cmake_minimum_required(VERSION 3.14)  # 建议提高到3.14，对C++11+支持更好
project(Tianlong)       # 项目名称

# 设置C++11标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置输出目录（支持多配置生成器，如Visual Studio）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)     # 可执行文件输出到build/bin/
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)     # 动态库(.so)输出到 build/lib/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)     # 静态库(.a)输出到 build/lib/

# 为多配置生成器（如Visual Studio）设置子目录
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# 明确列出源文件（最佳实践）
set(SOURCES
    src/test_joint_inference_openvino.cpp
    src/corneal_joint_inferencer_openvino.cpp
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 添加头文件目录（使用现代CMake语法，作用域仅限于目标）
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

find_package(OpenCV REQUIRED) # 查找OpenCV 4.x版本

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION} at ${OpenCV_DIR}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    
    # 链接 OpenCV
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV 3.x or 4.x development packages.")
endif()  


# 专业方式：查找并链接OpenVINO
# 优先尝试使用系统的OpenVINO安装
find_package(OpenVINO QUIET)

if(OpenVINO_FOUND)
    message(STATUS "Found system OpenVINO: ${OpenVINO_VERSION}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenVINO_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenVINO_INCLUDE_DIRS})
else()
    # 备选方案：使用项目自带的OpenVINO
    message(STATUS "System OpenVINO not found, trying project-local version")
    
    # 跨平台路径处理
    if(WIN32)
        set(OPENVINO_LOCAL_DIR "${CMAKE_SOURCE_DIR}/lib/openvino/windows")
    elseif(UNIX AND NOT APPLE)
        set(OPENVINO_LOCAL_DIR "${CMAKE_SOURCE_DIR}/lib/openvino/linux")
    elseif(APPLE)
        set(OPENVINO_LOCAL_DIR "${CMAKE_SOURCE_DIR}/lib/openvino/macos")
    endif()
    
    if(EXISTS "${OPENVINO_LOCAL_DIR}")
        message(STATUS "Using local OpenVINO from ${OPENVINO_LOCAL_DIR}")
        
        # 跨平台库名处理
        if(WIN32)
            set(OPENVINO_LIB_NAME openvino)
            set(OPENVINO_LIB_PATH "${OPENVINO_LOCAL_DIR}/bin")
            set(OPENVINO_INCLUDE_PATH "${OPENVINO_LOCAL_DIR}/include")
        else()
            set(OPENVINO_LIB_NAME openvino)
            set(OPENVINO_LIB_PATH "${OPENVINO_LOCAL_DIR}/lib")
            set(OPENVINO_INCLUDE_PATH "${OPENVINO_LOCAL_DIR}/include")
        endif()
        
        target_include_directories(${PROJECT_NAME} PRIVATE ${OPENVINO_INCLUDE_PATH})
        target_link_directories(${PROJECT_NAME} PRIVATE ${OPENVINO_LIB_PATH})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENVINO_LIB_NAME})
    else()
        message(WARNING "OpenVINO not found in system or project directory. Some features may be disabled.")
        # 这里可以添加回退逻辑或设置编译选项
    endif()
endif()

# 添加项目特定的库目录（如果有其他依赖）
if(EXISTS "${CMAKE_SOURCE_DIR}/lib")
    target_link_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/lib")
endif()

# 为调试构建添加调试符号
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES MATCHES "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -g)
endif()