cmake_minimum_required(VERSION 2.8)
project(Haralick_Ann_Reguralization)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(FANN REQUIRED)
INCLUDE_DIRECTORIES(${FANN_INCLUDE_DIRS})

find_package(ITK_Haralick REQUIRED)
include_directories(${ITK_Haralick_INCLUDE_DIRS})

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Valgrind REQUIRED)
include_directories(${VALGRIND_INCLUDE_DIR})

add_executable(main main.cpp time_utils.cpp cli_parser.cpp common.cpp learning_classes_loader.cpp)

set_target_properties(main PROPERTIES COMPILE_DEFINITIONS FANN_NO_DLL)
 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(main ITKReview ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${FANN_STATIC_FLOAT_LIBRARY} ${Boost_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(main ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${FANN_STATIC_FLOAT_LIBRARY} ${Boost_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
