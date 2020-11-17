find_package(OpenCV 4 REQUIRED)
#find_package(Boost 1.7 REQUIRED COMPONENTS filesystem)

#include_directories(${OpenCV_INCLUDE_DIRS} 
#                    ${Boost_INCLUDE_DIRS})

# SET(CMAKE_CXX_FLAGS_DEBUG "-O0")
# SET(CMAKE_C_FLAGS_DEBUG "-O0")

include_directories(${OpenCV_INCLUDE_DIRS}) 


add_library(features features.cpp)
add_library(serialization serialization.cpp)
add_library(codebook codebook.cpp)
add_library(histbook histbook.cpp)

add_executable(preprocess preprocess_and_serialize.cpp)
target_link_libraries(preprocess 
                    features 
                    serialization 
                    codebook
                    histbook
                    ${OpenCV_LIBS}) 
#                    Boost::filesystem)

add_executable(main main.cpp)
target_link_libraries(main 
                    features 
                    serialization 
                    codebook 
                    histbook
                    ${OpenCV_LIBS})           
