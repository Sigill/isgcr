cmake_minimum_required(VERSION 2.8)
project(Haralick_Ann_Reguralization)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(ITK REQUIRED COMPONENTS ITKCommon ITKIOImageBase ITKIOMeta ITKIOPNG ITKIOBMP ITKIOJPEG)
include(${ITK_USE_FILE})

find_package(FANN REQUIRED)
INCLUDE_DIRECTORIES(${FANN_INCLUDE_DIRS})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
include_directories( ${QT_INCLUDE_DIR})

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    ON)
find_package(Boost COMPONENTS program_options system filesystem regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Valgrind REQUIRED)
include_directories(${VALGRIND_INCLUDE_DIR})

find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(TULIP REQUIRED)
include_directories(${TULIP_INCLUDE_DIR})
add_definitions(-DITK_MANUAL_INSTANTIATION)

add_executable(haralick_ann_regularization
	main.cpp
	time_utils.cpp
	cli_parser.cpp
	common.cpp
	image_loader.cpp
	NeuralNetworkPixelClassifiers.cpp
	LoggerPluginProgress.cpp
	ParseUtils.cpp
	ClassificationDataset.cpp
	FannClassificationDataset.cpp
	templates.cpp
	boost_program_options_types.cpp
	)
set_target_properties(haralick_ann_regularization PROPERTIES COMPILE_DEFINITIONS FANN_NO_DLL)
target_link_libraries(haralick_ann_regularization ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${Boost_LIBRARIES} ${TULIP_LIBRARIES} ${LOG4CXX_LIBRARIES} ${QT_LIBRARIES})

install(TARGETS haralick_ann_regularization RUNTIME DESTINATION ".")
