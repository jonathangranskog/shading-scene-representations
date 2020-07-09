cmake_minimum_required(VERSION 3.1)

project(optixRenderer CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

# OPTIX SETUP
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
if (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
set(BUILD_SHARED_LIBS ON)

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch sm_30)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()
  set(flag "--use_fast_math")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()

  if (CUDA_VERSION VERSION_LESS "3.0")
    set(flag "--keep")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    set(flag "-U__BLOCKS__")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  set(CUDA_NVRTC_FLAGS -arch compute_30 -use_fast_math -default-device -rdc true -D__x86_64 CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../Optix" CACHE PATH "Path to OptiX installed location.")

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  )



# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")

include_directories(${OptiX_INCLUDE} ${CUDA_INCLUDE_DIRS})

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
if (NOT CUDA_NVRTC_ENABLED)
    source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

# Separate the sources from the CMake and CUDA options fed to the macro.  This code
# comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
# code here, so that we can use our own name for the target.  target_name is used in the
# creation of the output file names, and we want this to be unique for each target in
# the SDK.
CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

if (CUDA_NVRTC_ENABLED)
  
    # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
    set(cu_obj_source_files)
    foreach(file ${source_files})
      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
      if(${_cuda_source_format} MATCHES "OBJ")
        list(APPEND cu_obj_source_files ${file})
      endif()
    endforeach()

    # Create the rules to build the OBJ from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )
else()
  
    # Create the rules to build the PTX and OBJ from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
endif()

# tinyobj
set(tiny_obj_files ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyobj/tiny_obj_loader.h ${CMAKE_CURRENT_SOURCE_DIR}/src/tinyobj/tiny_obj_loader.cc)

# filesystem
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/filesystem)

# STB
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/stb)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/stb)

# rapidjson
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL " " FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/rapidjson/rapidjson)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/rapidjson/rapidjson/include)

file(GLOB cuda_files ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu)

file(GLOB src_files
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
        #${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

cuda_compile_ptx(
  render_ptx
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/render.cu
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/kernels/render.ptx COMMAND ${CMAKE_COMMAND} -E rename ${render_ptx} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/render.ptx DEPENDS ${render_ptx}) 
add_custom_target(render_ptx ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/render.ptx 
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/render.cu SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/render.cu)

cuda_compile_ptx(
  intersect_ptx
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/intersect.cu
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/kernels/intersect.ptx COMMAND ${CMAKE_COMMAND} -E rename ${intersect_ptx} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/intersect.ptx DEPENDS ${intersect_ptx}) 
add_custom_target(intersect_ptx ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/intersect.ptx 
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/intersect.cu SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/intersect.cu)

add_subdirectory(ext/pybind11)
pybind11_add_module(rnd pybind/render.cpp ${src_files} ${cuda_files} ${tiny_obj_files} ${source_files} ${generated_files} ${cmake_options})
target_link_libraries(rnd PRIVATE m optix)
