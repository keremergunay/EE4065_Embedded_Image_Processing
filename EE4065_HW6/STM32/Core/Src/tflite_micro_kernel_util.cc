/**
 * TFLite Micro Kernel Utility Functions
 * 
 * These functions are required by TFLite Micro kernels but may be missing
 * from your TFLite Micro build.
 * 
 * Copy this file to: C:/Users/Kerem/STM32CubeIDE/workspace_1.19.0/HW6/Core/Src/
 */

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

// Forward declare QuantizeMultiplier from quantization_util.cc
namespace tflite {
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift);
}

namespace tflite {

// ============================================================================
// Kernel Utility Functions (from kernel_util.cc)
// ============================================================================

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
    if (input1 == nullptr || input2 == nullptr) {
        return false;
    }
    if (input1->dims == nullptr || input2->dims == nullptr) {
        return input1->dims == input2->dims;
    }
    if (input1->dims->size != input2->dims->size) {
        return false;
    }
    for (int i = 0; i < input1->dims->size; ++i) {
        if (input1->dims->data[i] != input2->dims->data[i]) {
            return false;
        }
    }
    return true;
}

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                                TfLiteFusedActivation activation,
                                                TfLiteTensor* output,
                                                int32_t* act_min,
                                                int32_t* act_max) {
    int32_t qmin = 0;
    int32_t qmax = 0;
    
    if (output->type == kTfLiteUInt8) {
        qmin = 0;
        qmax = 255;
    } else if (output->type == kTfLiteInt8) {
        qmin = -128;
        qmax = 127;
    } else if (output->type == kTfLiteInt16) {
        qmin = -32768;
        qmax = 32767;
    } else {
        qmin = -128;
        qmax = 127;
    }

    *act_min = qmin;
    *act_max = qmax;
    
    const auto scale = output->params.scale;
    const auto zero_point = output->params.zero_point;

    if (scale == 0.0f) {
        return kTfLiteOk;
    }

    auto quantize = [scale, zero_point](float f) {
        return zero_point + static_cast<int32_t>(std::round(f / scale));
    };

    if (activation == kTfLiteActRelu) {
        *act_min = std::max(qmin, quantize(0.0f));
    } else if (activation == kTfLiteActRelu6) {
        *act_min = std::max(qmin, quantize(0.0f));
        *act_max = std::min(qmax, quantize(6.0f));
    } else if (activation == kTfLiteActReluN1To1) {
        *act_min = std::max(qmin, quantize(-1.0f));
        *act_max = std::min(qmax, quantize(1.0f));
    }
    
    return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                               const TfLiteTensor* input,
                                               const TfLiteTensor* filter,
                                               const TfLiteTensor* bias,
                                               TfLiteTensor* output,
                                               double* multiplier) {
    const double input_product_scale = 
        static_cast<double>(input->params.scale) * 
        static_cast<double>(filter->params.scale);
    
    const double output_scale = static_cast<double>(output->params.scale);
    
    if (output_scale == 0.0) {
        *multiplier = 0.0;
    } else {
        *multiplier = input_product_scale / output_scale;
    }
    
    return kTfLiteOk;
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* output_multiplier,
    int* output_shift, int32_t* output_activation_min,
    int32_t* output_activation_max, int32_t* per_channel_multiplier,
    int32_t* per_channel_shift, int num_channels) {
    
    // Calculate the effective scale
    double real_multiplier = 0.0;
    TfLiteStatus status = GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier);
    if (status != kTfLiteOk) {
        return status;
    }
    
    // Quantize the multiplier
    QuantizeMultiplier(real_multiplier, output_multiplier, output_shift);
    
    // For per-channel quantization, populate per-channel values
    if (per_channel_multiplier != nullptr && per_channel_shift != nullptr) {
        // If filter has per-channel quantization
        if (filter->quantization.type == kTfLiteAffineQuantization) {
            const TfLiteAffineQuantization* affine = 
                static_cast<const TfLiteAffineQuantization*>(filter->quantization.params);
            
            if (affine != nullptr && affine->scale != nullptr && 
                affine->scale->size == num_channels) {
                const double input_scale = static_cast<double>(input->params.scale);
                const double output_scale = static_cast<double>(output->params.scale);
                
                for (int i = 0; i < num_channels; ++i) {
                    const double filter_scale = static_cast<double>(affine->scale->data[i]);
                    const double channel_multiplier = 
                        (input_scale * filter_scale) / output_scale;
                    int shift_temp;
                    QuantizeMultiplier(channel_multiplier, 
                                      &per_channel_multiplier[i],
                                      &shift_temp);
                    per_channel_shift[i] = shift_temp;
                }
            } else {
                // Fall back to single multiplier for all channels
                for (int i = 0; i < num_channels; ++i) {
                    per_channel_multiplier[i] = *output_multiplier;
                    per_channel_shift[i] = *output_shift;
                }
            }
        } else {
            // No per-channel quantization, use same values for all
            for (int i = 0; i < num_channels; ++i) {
                per_channel_multiplier[i] = *output_multiplier;
                per_channel_shift[i] = *output_shift;
            }
        }
    }
    
    // Calculate activation range
    return CalculateActivationRangeQuantized(context, activation, output,
                                              output_activation_min,
                                              output_activation_max);
}

}  // namespace tflite

