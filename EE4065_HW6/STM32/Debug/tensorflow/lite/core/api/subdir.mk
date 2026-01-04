################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow/lite/core/api/error_reporter.cc \
../tensorflow/lite/core/api/flatbuffer_conversions.cc \
../tensorflow/lite/core/api/tensor_utils.cc 

CC_DEPS += \
./tensorflow/lite/core/api/error_reporter.d \
./tensorflow/lite/core/api/flatbuffer_conversions.d \
./tensorflow/lite/core/api/tensor_utils.d 

OBJS += \
./tensorflow/lite/core/api/error_reporter.o \
./tensorflow/lite/core/api/flatbuffer_conversions.o \
./tensorflow/lite/core/api/tensor_utils.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow/lite/core/api/%.o tensorflow/lite/core/api/%.su tensorflow/lite/core/api/%.cyclo: ../tensorflow/lite/core/api/%.cc tensorflow/lite/core/api/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++17 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6 -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/kissfft -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/tensorflow -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/flatbuffers/include -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/gemmlowp -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/ruy -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-tensorflow-2f-lite-2f-core-2f-api

clean-tensorflow-2f-lite-2f-core-2f-api:
	-$(RM) ./tensorflow/lite/core/api/error_reporter.cyclo ./tensorflow/lite/core/api/error_reporter.d ./tensorflow/lite/core/api/error_reporter.o ./tensorflow/lite/core/api/error_reporter.su ./tensorflow/lite/core/api/flatbuffer_conversions.cyclo ./tensorflow/lite/core/api/flatbuffer_conversions.d ./tensorflow/lite/core/api/flatbuffer_conversions.o ./tensorflow/lite/core/api/flatbuffer_conversions.su ./tensorflow/lite/core/api/tensor_utils.cyclo ./tensorflow/lite/core/api/tensor_utils.d ./tensorflow/lite/core/api/tensor_utils.o ./tensorflow/lite/core/api/tensor_utils.su

.PHONY: clean-tensorflow-2f-lite-2f-core-2f-api

