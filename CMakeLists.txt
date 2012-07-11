cmake_minimum_required(VERSION 2.8)
project(Haralick_Ann_Reguralization)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(FANN REQUIRED)
INCLUDE_DIRECTORIES(${FANN_INCLUDE_DIRS})

find_package(ITK_Haralick REQUIRED)
INCLUDE_DIRECTORIES(${ITK_Haralick_INCLUDE_DIRS})

add_executable(main main.cpp)

SET_TARGET_PROPERTIES(main PROPERTIES COMPILE_DEFINITIONS FANN_NO_DLL)
 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(main ITKReview ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${FANN_STATIC_FLOAT_LIBRARY} )
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(main ${ITK_LIBRARIES} ${FANN_STATIC_DOUBLE_LIBRARY} ${FANN_STATIC_FLOAT_LIBRARY})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
