################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow/lite/experimental/microfrontend/lib/fft.cc \
../tensorflow/lite/experimental/microfrontend/lib/fft_util.cc \
../tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.cc 

C_SRCS += \
../tensorflow/lite/experimental/microfrontend/lib/filterbank.c \
../tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c \
../tensorflow/lite/experimental/microfrontend/lib/frontend.c \
../tensorflow/lite/experimental/microfrontend/lib/frontend_util.c \
../tensorflow/lite/experimental/microfrontend/lib/log_lut.c \
../tensorflow/lite/experimental/microfrontend/lib/log_scale.c \
../tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c \
../tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c \
../tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c \
../tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c \
../tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c \
../tensorflow/lite/experimental/microfrontend/lib/window.c \
../tensorflow/lite/experimental/microfrontend/lib/window_util.c 

C_DEPS += \
./tensorflow/lite/experimental/microfrontend/lib/filterbank.d \
./tensorflow/lite/experimental/microfrontend/lib/filterbank_util.d \
./tensorflow/lite/experimental/microfrontend/lib/frontend.d \
./tensorflow/lite/experimental/microfrontend/lib/frontend_util.d \
./tensorflow/lite/experimental/microfrontend/lib/log_lut.d \
./tensorflow/lite/experimental/microfrontend/lib/log_scale.d \
./tensorflow/lite/experimental/microfrontend/lib/log_scale_util.d \
./tensorflow/lite/experimental/microfrontend/lib/noise_reduction.d \
./tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.d \
./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.d \
./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.d \
./tensorflow/lite/experimental/microfrontend/lib/window.d \
./tensorflow/lite/experimental/microfrontend/lib/window_util.d 

CC_DEPS += \
./tensorflow/lite/experimental/microfrontend/lib/fft.d \
./tensorflow/lite/experimental/microfrontend/lib/fft_util.d \
./tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.d 

OBJS += \
./tensorflow/lite/experimental/microfrontend/lib/fft.o \
./tensorflow/lite/experimental/microfrontend/lib/fft_util.o \
./tensorflow/lite/experimental/microfrontend/lib/filterbank.o \
./tensorflow/lite/experimental/microfrontend/lib/filterbank_util.o \
./tensorflow/lite/experimental/microfrontend/lib/frontend.o \
./tensorflow/lite/experimental/microfrontend/lib/frontend_util.o \
./tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.o \
./tensorflow/lite/experimental/microfrontend/lib/log_lut.o \
./tensorflow/lite/experimental/microfrontend/lib/log_scale.o \
./tensorflow/lite/experimental/microfrontend/lib/log_scale_util.o \
./tensorflow/lite/experimental/microfrontend/lib/noise_reduction.o \
./tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.o \
./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.o \
./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.o \
./tensorflow/lite/experimental/microfrontend/lib/window.o \
./tensorflow/lite/experimental/microfrontend/lib/window_util.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow/lite/experimental/microfrontend/lib/%.o tensorflow/lite/experimental/microfrontend/lib/%.su tensorflow/lite/experimental/microfrontend/lib/%.cyclo: ../tensorflow/lite/experimental/microfrontend/lib/%.cc tensorflow/lite/experimental/microfrontend/lib/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++17 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6 -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/kissfft -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/tensorflow -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/flatbuffers/include -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/gemmlowp -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/ruy -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
tensorflow/lite/experimental/microfrontend/lib/%.o tensorflow/lite/experimental/microfrontend/lib/%.su tensorflow/lite/experimental/microfrontend/lib/%.cyclo: ../tensorflow/lite/experimental/microfrontend/lib/%.c tensorflow/lite/experimental/microfrontend/lib/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6 -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/kissfft -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/tensorflow -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/flatbuffers/include -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/gemmlowp -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/ruy -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-tensorflow-2f-lite-2f-experimental-2f-microfrontend-2f-lib

clean-tensorflow-2f-lite-2f-experimental-2f-microfrontend-2f-lib:
	-$(RM) ./tensorflow/lite/experimental/microfrontend/lib/fft.cyclo ./tensorflow/lite/experimental/microfrontend/lib/fft.d ./tensorflow/lite/experimental/microfrontend/lib/fft.o ./tensorflow/lite/experimental/microfrontend/lib/fft.su ./tensorflow/lite/experimental/microfrontend/lib/fft_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/fft_util.d ./tensorflow/lite/experimental/microfrontend/lib/fft_util.o ./tensorflow/lite/experimental/microfrontend/lib/fft_util.su ./tensorflow/lite/experimental/microfrontend/lib/filterbank.cyclo ./tensorflow/lite/experimental/microfrontend/lib/filterbank.d ./tensorflow/lite/experimental/microfrontend/lib/filterbank.o ./tensorflow/lite/experimental/microfrontend/lib/filterbank.su ./tensorflow/lite/experimental/microfrontend/lib/filterbank_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/filterbank_util.d ./tensorflow/lite/experimental/microfrontend/lib/filterbank_util.o ./tensorflow/lite/experimental/microfrontend/lib/filterbank_util.su ./tensorflow/lite/experimental/microfrontend/lib/frontend.cyclo ./tensorflow/lite/experimental/microfrontend/lib/frontend.d ./tensorflow/lite/experimental/microfrontend/lib/frontend.o ./tensorflow/lite/experimental/microfrontend/lib/frontend.su ./tensorflow/lite/experimental/microfrontend/lib/frontend_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/frontend_util.d ./tensorflow/lite/experimental/microfrontend/lib/frontend_util.o ./tensorflow/lite/experimental/microfrontend/lib/frontend_util.su ./tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.cyclo ./tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.d ./tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.o ./tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.su ./tensorflow/lite/experimental/microfrontend/lib/log_lut.cyclo ./tensorflow/lite/experimental/microfrontend/lib/log_lut.d ./tensorflow/lite/experimental/microfrontend/lib/log_lut.o ./tensorflow/lite/experimental/microfrontend/lib/log_lut.su ./tensorflow/lite/experimental/microfrontend/lib/log_scale.cyclo ./tensorflow/lite/experimental/microfrontend/lib/log_scale.d ./tensorflow/lite/experimental/microfrontend/lib/log_scale.o ./tensorflow/lite/experimental/microfrontend/lib/log_scale.su ./tensorflow/lite/experimental/microfrontend/lib/log_scale_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/log_scale_util.d ./tensorflow/lite/experimental/microfrontend/lib/log_scale_util.o ./tensorflow/lite/experimental/microfrontend/lib/log_scale_util.su ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction.cyclo ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction.d ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction.o ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction.su ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.d ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.o ./tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.su ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.cyclo ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.d ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.o ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.su ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.d ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.o ./tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.su ./tensorflow/lite/experimental/microfrontend/lib/window.cyclo ./tensorflow/lite/experimental/microfrontend/lib/window.d ./tensorflow/lite/experimental/microfrontend/lib/window.o ./tensorflow/lite/experimental/microfrontend/lib/window.su ./tensorflow/lite/experimental/microfrontend/lib/window_util.cyclo ./tensorflow/lite/experimental/microfrontend/lib/window_util.d ./tensorflow/lite/experimental/microfrontend/lib/window_util.o ./tensorflow/lite/experimental/microfrontend/lib/window_util.su

.PHONY: clean-tensorflow-2f-lite-2f-experimental-2f-microfrontend-2f-lib

