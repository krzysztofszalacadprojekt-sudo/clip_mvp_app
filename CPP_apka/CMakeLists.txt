cmake_minimum_required(VERSION 3.20)
project(TestBackend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use FetchContent to automatically download and build libcurl
include(FetchContent)
set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE) # Use Windows native SSL/TLS

FetchContent_Declare(
    curl
    URL https://github.com/curl/curl/releases/download/curl-8_7_1/curl-8.7.1.zip
)
FetchContent_MakeAvailable(curl)

add_executable(TestBackend main.cpp)
target_link_libraries(TestBackend PRIVATE libcurl)