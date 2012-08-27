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
find_package(Boost COMPONENTS program_options system filesystem regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Valgrind REQUIRED)
include_directories(${VALGRIND_INCLUDE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(TULIP3 REQUIRED)
include_directories(${TULIP_INCLUDE_DIR})

add_library(image_loader STATIC image_loader.cpp)
add_library(haralick STATIC haralick.cpp)
add_library(classification STATIC classification.cpp)

add_executable(main main.cpp time_utils.cpp cli_parser.cpp common.cpp)

set_target_properties(main PROPERTIES COMPILE_DEFINITIONS FANN_NO_DLL)
 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
	target_link_libraries(image_loader ITKReview ${ITK_LIBRARIES} ${Boost_LIBRARIES})
	target_link_libraries(haralick ITKReview image_loader ${ITK_LIBRARIES})
	target_link_libraries(classification ITKReview image_loader ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${Boost_LIBRARIES})

	target_link_libraries(main ITKReview image_loader haralick classification ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${Boost_LIBRARIES} ${TULIP_LIBRARY})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
	target_link_libraries(image_loader ${ITK_LIBRARIES} ${Boost_LIBRARIES})
	target_link_libraries(haralick image_loader ${ITK_LIBRARIES})
	target_link_libraries(classification image_loader ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${Boost_LIBRARIES})

	target_link_libraries(main image_loader haralick classification ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${Boost_LIBRARIES} ${TULIP_LIBRARY})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
