cmake_minimum_required(VERSION 2.8)
project(Haralick_Ann_Reguralization)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(FANN REQUIRED)
INCLUDE_DIRECTORIES(${FANN_INCLUDE_DIRS})

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

find_package(TULIP3 REQUIRED)
include_directories(${TULIP_INCLUDE_DIR})

add_executable(main main.cpp time_utils.cpp cli_parser.cpp common.cpp image_loader.cpp NeuralNetworkPixelClassifiers.cpp LoggerPluginProgress.cpp ParseUtils.cpp)
set_target_properties(main PROPERTIES COMPILE_DEFINITIONS FANN_NO_DLL)
target_link_libraries(main ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${Boost_LIBRARIES} ${TULIP_LIBRARY} ${LOG4CXX_LIBRARIES})

