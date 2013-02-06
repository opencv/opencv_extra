cmake_minimum_required(VERSION 2.8.8)

project(gpu_demos_pack)

set(README_FILE "${CMAKE_SOURCE_DIR}/README.txt")
set(LICENSE_FILE "${CMAKE_SOURCE_DIR}/LICENSE.txt")

############ Find packages #############

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include_directories("${CMAKE_SOURCE_DIR}/utility")

find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

if(WIN32)
    find_package(CUDA REQUIRED)
    find_package(Qt4 REQUIRED)
endif()

############ targets ############

file(GLOB sources "${CMAKE_SOURCE_DIR}/utility/*.h" "${CMAKE_SOURCE_DIR}/utility/*.cpp")
add_library(utility STATIC ${sources})
target_link_libraries(utility ${OpenCV_LIBS} ${TBB_LIBRARIES})

file(GLOB demos RELATIVE "${CMAKE_SOURCE_DIR}/demos" "${CMAKE_SOURCE_DIR}/demos/*")
message(STATUS "Found demos:")
foreach(demo ${demos})
    if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/demos/${demo}")
        option(BUILD_demo_${demo} "Build ${demo} demo" ON)

        if(BUILD_demo_${demo})
            set(target "demo_${demo}")

            file(GLOB sources "${CMAKE_SOURCE_DIR}/demos/${demo}/*.h" "${CMAKE_SOURCE_DIR}/demos/${demo}/*.cpp")
            add_executable(${target} ${sources})
            add_dependencies(${target} utility)
            target_link_libraries(${target} ${OpenCV_LIBS} ${TBB_LIBRARIES} utility)

            install(TARGETS ${target} RUNTIME DESTINATION ".")

            string(REPLACE "_" ";" DEMO_WORDS ${demo})
            set(DEMO_NAME "")
            foreach(word ${DEMO_WORDS})
                string(SUBSTRING ${word} 0 1 first_letter)
                string(SUBSTRING ${word} 1 -1 last_part)
                string(TOUPPER ${first_letter} first_letter)
                set(DEMO_NAME "${DEMO_NAME} ${first_letter}${last_part}")
            endforeach()
            string(STRIP ${DEMO_NAME} DEMO_NAME)
            message(STATUS "  ${DEMO_NAME}")

            if(WIN32)
                list(APPEND CPACK_PACKAGE_EXECUTABLES "${target};${DEMO_NAME}")
            endif()
        endif()
    endif()
endforeach()

############### Install ################

install(DIRECTORY "data" DESTINATION ".")
install(DIRECTORY "demos" "utility" "cmake" DESTINATION ".")
install(FILES "CMakeLists.txt" "opencv.ico" ${README_FILE} ${LICENSE_FILE} DESTINATION ".")
if(WIN32)
    install(DIRECTORY "codecs" DESTINATION ".")
endif()

if(WIN32)
    string(REPLACE "lib" "bin" OpenCV_BIN_DIR ${OpenCV_LIB_DIR_OPT})
    file(GLOB ALL_OpenCV_DLLS "${OpenCV_BIN_DIR}/*.dll")
    install(FILES ${ALL_OpenCV_DLLS} DESTINATION ".")

    string(REPLACE "lib" "bin" TBB_BINARY_DIR ${TBB_LIBRARY_DIR})
    set(ALL_TBB_DLLS "${TBB_BINARY_DIR}/tbb.dll" "${TBB_BINARY_DIR}/tbbmalloc.dll")
    install(FILES ${ALL_TBB_DLLS} DESTINATION ".")

    set(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        file(GLOB CUBLAS_DLL "${CUDA_BIN_DIR}/cublas64*.dll")
        file(GLOB CUDART_DLL "${CUDA_BIN_DIR}/cudart64*.dll")
        file(GLOB CUFFT_DLL "${CUDA_BIN_DIR}/cufft64*.dll")
        file(GLOB NPP_DLL "${CUDA_BIN_DIR}/npp64*.dll")
    else()
        file(GLOB CUBLAS_DLL "${CUDA_BIN_DIR}/cublas32*.dll")
        file(GLOB CUDART_DLL "${CUDA_BIN_DIR}/cudart32*.dll")
        file(GLOB CUFFT_DLL "${CUDA_BIN_DIR}/cufft32*.dll")
        file(GLOB NPP_DLL "${CUDA_BIN_DIR}/npp32*.dll")
    endif()
    set(ALL_CUDA_DLLS ${CUBLAS_DLL} ${CUDART_DLL} ${CUFFT_DLL} ${NPP_DLL})
    install(FILES ${ALL_CUDA_DLLS} DESTINATION ".")

    set(ALL_Qt_DLLS ${QT_BINARY_DIR}/QtCore4.dll ${QT_BINARY_DIR}/QtGui4.dll ${QT_BINARY_DIR}/QtOpenGL4.dll ${QT_BINARY_DIR}/QtTest4.dll)
    install(FILES ${ALL_Qt_DLLS} DESTINATION ".")
endif()

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
include(InstallRequiredSystemLibraries)

################ CPack #################

set(CPACK_PACKAGE_NAME "OpenCV GPU Demos Pack")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenCV ${OpenCV_VERSION} GPU Demos Pack")
set(CPACK_PACKAGE_VENDOR "Itseez")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CPACK_PACKAGE_FILE_NAME "OpenCV-${OpenCV_VERSION}-GPU-demos-pack-x64")
else()
    set(CPACK_PACKAGE_FILE_NAME "OpenCV-${OpenCV_VERSION}-GPU-demos-pack-x86")
endif()
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_FILE_NAME})
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY ${CPACK_PACKAGE_FILE_NAME})

set(CPACK_PACKAGE_VERSION_MAJOR "${OpenCV_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${OpenCV_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${OpenCV_VERSION_PATCH}")

set(CPACK_PACKAGE_DESCRIPTION_FILE "${README_FILE}")
set(CPACK_RESOURCE_FILE_README "${README_FILE}")
set(CPACK_RESOURCE_FILE_LICENSE "${LICENSE_FILE}")

set(CPACK_MONOLITHIC_INSTALL TRUE)

if(WIN32)
    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}\\\\opencv.ico")
    set(CPACK_NSIS_MUI_ICON ${CPACK_PACKAGE_ICON})
    set(CPACK_NSIS_MUI_UNIICON ${CPACK_PACKAGE_ICON})

    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")

    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "ExecWait \\\"$INSTDIR\\\\codecs\\\\ffdshow_beta7_rev3154_20091209.exe\\\"")
endif()

include(CPack)
