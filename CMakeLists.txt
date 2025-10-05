# CMakeList.txt : CMake project for SpaceApps2025, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.31)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("spaceapps2025")

# Generate a header with the build path
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/BuildConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/BuildConfig.h"
  @ONLY
)

# Add source to this project's executable.
add_executable (SpaceApps2025 "main/SpaceApps2025.cpp" "main/SpaceApps2025.h" "models/attention.h" "models/attention.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET SpaceApps2025 PROPERTY CXX_STANDARD 20)
endif()

target_include_directories(SpaceApps2025 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

find_package(crow CONFIG REQUIRED)
target_link_libraries(SpaceApps2025 PRIVATE Crow::Crow)