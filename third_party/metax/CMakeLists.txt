include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} APPEND PROPERTY INCLUDE_DIRECTORIES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
add_subdirectory(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)

if(FLAGTREE_PLUGIN)
  set(PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
elseif(EDITABLE_MODE)
  set(PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set(PLUGIN_DIR "$ENV{HOME}/.flagtree/${FLAGTREE_BACKEND}")
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  if(BUILD_MCTLE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/plugin)
    add_subdirectory(plugin/mctle)
  endif()

  if(FLAGTREE_PLUGIN)
    include_directories(plugin)
    add_subdirectory(plugin)
  else()
    find_library(BackendTritonPluginLib
      NAMES
        metaxTritonPlugin.so
      PATHS
        ${PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMetax
      SHARED_LIB ${BackendTritonPluginLib}
    )
    list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" PARENT_SCOPE)
  endif()
endif()

target_sources(TritonTools PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Tools/SwizzleDumpUtils.cpp
)

if(BUILD_MCTLE)
  add_dependencies(TritonAnalysis McTleTableGen)
  target_link_libraries(TritonToTritonGPU PUBLIC McTleIR)
  target_include_directories(TritonToTritonGPU PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/metax/plugin
    ${PROJECT_BINARY_DIR}/third_party/metax/plugin
  )
endif()

if(TRITON_BUILD_DISTRIBUTED)
  target_link_libraries(TritonAnalysis PUBLIC DistributedIR)
endif()
