cmake_minimum_required (VERSION 3.31)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

add_compile_definitions(_WIN32_WINNT=0x0601)

project ("spaceapps2025")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/BuildConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/BuildConfig.h"
  @ONLY
)

add_executable (SpaceApps2025 "main/SpaceApps2025.cpp" "main/SpaceApps2025.h"   "web/web_manager.h" "web/web_manager.cpp" "data/data_manager.h" "data/data_manager.cpp" "models/DNN.h" "models/DNN.cpp" "data/dataset_split.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET SpaceApps2025 PROPERTY CXX_STANDARD 20)
endif()

target_include_directories(SpaceApps2025 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

find_package(crow CONFIG REQUIRED)
target_link_libraries(SpaceApps2025 PRIVATE Crow::Crow)