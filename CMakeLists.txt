cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(ekf_for_multicopter C CXX ASM)
set(CMAKE_C_STNDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

include_directories(/Users/itoukouhei/pico/eigen)

add_executable(ekf_for_multicopter
        ekf_for_multicopter.cpp
)

# Pull in our pico_stdlib which pulls in commonly used features
target_link_libraries(ekf_for_multicopter pico_stdlib hardware_uart)
target_compile_options(ekf_for_multicopter PRIVATE -DEIGEN_NO_DEBUG)

# enable usb output, disable uart output
pico_enable_stdio_usb(ekf_for_multicopter 1)
pico_enable_stdio_uart(ekf_for_multicopter 0)

# create map/bin/hex file etc.
pico_add_extra_outputs(ekf_for_multicopter)
