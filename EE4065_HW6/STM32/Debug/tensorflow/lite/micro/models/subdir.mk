################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow/lite/micro/models/person_detect_model_data.cc 

CC_DEPS += \
./tensorflow/lite/micro/models/person_detect_model_data.d 

OBJS += \
./tensorflow/lite/micro/models/person_detect_model_data.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow/lite/micro/models/%.o tensorflow/lite/micro/models/%.su tensorflow/lite/micro/models/%.cyclo: ../tensorflow/lite/micro/models/%.cc tensorflow/lite/micro/models/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++17 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6 -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/kissfft -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/tensorflow -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/flatbuffers/include -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/gemmlowp -IC:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/third_party/ruy -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-tensorflow-2f-lite-2f-micro-2f-models

clean-tensorflow-2f-lite-2f-micro-2f-models:
	-$(RM) ./tensorflow/lite/micro/models/person_detect_model_data.cyclo ./tensorflow/lite/micro/models/person_detect_model_data.d ./tensorflow/lite/micro/models/person_detect_model_data.o ./tensorflow/lite/micro/models/person_detect_model_data.su

.PHONY: clean-tensorflow-2f-lite-2f-micro-2f-models

