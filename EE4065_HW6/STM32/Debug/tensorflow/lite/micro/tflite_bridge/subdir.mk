################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc \
../tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc 

CC_DEPS += \
./tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.d \
./tensorflow/lite/micro/tflite_bridge/micro_error_reporter.d 

OBJS += \
./tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.o \
./tensorflow/lite/micro/tflite_bridge/micro_error_reporter.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow/lite/micro/tflite_bridge/%.o tensorflow/lite/micro/tflite_bridge/%.su tensorflow/lite/micro/tflite_bridge/%.cyclo: ../tensorflow/lite/micro/tflite_bridge/%.cc tensorflow/lite/micro/tflite_bridge/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++17 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6 -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/kissfft -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/tensorflow -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/flatbuffers/include -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/gemmlowp -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/ruy -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-tensorflow-2f-lite-2f-micro-2f-tflite_bridge

clean-tensorflow-2f-lite-2f-micro-2f-tflite_bridge:
	-$(RM) ./tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cyclo ./tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.d ./tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.o ./tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.su ./tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cyclo ./tensorflow/lite/micro/tflite_bridge/micro_error_reporter.d ./tensorflow/lite/micro/tflite_bridge/micro_error_reporter.o ./tensorflow/lite/micro/tflite_bridge/micro_error_reporter.su

.PHONY: clean-tensorflow-2f-lite-2f-micro-2f-tflite_bridge

