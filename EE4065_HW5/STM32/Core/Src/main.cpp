/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.cpp
  * @brief          : EE4065 Homework 5 - Combined KWS and HDR on STM32 F446RE
  *
  * SIMPLIFIED VERSION - No TensorFlow Lite dependency!
  * Implements MLP inference directly using extracted weights.
  *
  * Protocol:
  *   - '1': Select KWS mode (Keyword Spotting)
  *   - '2': Select HDR mode (Handwritten Digit Recognition)
  *   - 'i': Show info
  *   - '?': Show menu
  *   - 0xAA: Start inference (send ACK, receive features, return predictions)
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* USER CODE BEGIN Includes */
// Model weights (extracted from Keras models)
#include "kws_weights.h"
#include "hdr_weights.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum {
    MODE_IDLE = 0,
    MODE_KWS,       // Q1: Keyword Spotting (26 features)
    MODE_HDR        // Q2: Handwritten Digit Recognition (7 features)
} OperationMode;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// Feature sizes
#define NUM_FEATURES_KWS    26
#define NUM_FEATURES_HDR    7
#define NUM_CLASSES         10

// Protocol bytes
#define SYNC_BYTE           0xAA
#define ACK_BYTE            0x55

// Maximum layer size (100 neurons in hidden layers)
#define MAX_LAYER_SIZE      100

/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
// NOTE: If you have usart.c from CubeIDE, use: extern UART_HandleTypeDef huart2;
// If you removed usart.c, uncomment the line below:
extern UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// Working buffers for inference
static float layer_input[MAX_LAYER_SIZE];
static float layer_output[MAX_LAYER_SIZE];

// Current mode
static OperationMode current_mode = MODE_IDLE;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN PFP */
static void RunKWSInference(void);
static void RunHDRInference(void);
static void PrintMenu(void);
static void PrintInfo(void);

// MLP inference functions
static void MatMulAdd(const float* input, const float* weights, const float* bias,
                      float* output, int input_size, int output_size);
static void ReLU(float* data, int size);
static void Softmax(float* data, int size);

/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

// Printf redirect to UART
#ifdef __GNUC__
extern "C" int __io_putchar(int ch)
{
    HAL_UART_Transmit(&huart2, (uint8_t*)&ch, 1, HAL_MAX_DELAY);
    return ch;
}
#endif

/**
  * @brief Matrix multiplication with bias addition: output = input * weights + bias
  */
static void MatMulAdd(const float* input, const float* weights, const float* bias,
                      float* output, int input_size, int output_size)
{
    for (int j = 0; j < output_size; j++) {
        float sum = bias[j];
        for (int i = 0; i < input_size; i++) {
            // weights stored in row-major: weights[i * output_size + j]
            sum += input[i] * weights[i * output_size + j];
        }
        output[j] = sum;
    }
}

/**
  * @brief ReLU activation: output = max(0, input)
  */
static void ReLU(float* data, int size)
{
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

/**
  * @brief Softmax activation: output = exp(input) / sum(exp(input))
  */
static void Softmax(float* data, int size)
{
    // Find max for numerical stability
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

/**
  * @brief Run KWS MLP inference
  *        Architecture: 26 -> 100 (ReLU) -> 100 (ReLU) -> 10 (Softmax)
  */
static void RunKWSInference(void)
{
    float features[NUM_FEATURES_KWS];
    float predictions[NUM_CLASSES];
    uint8_t ack = ACK_BYTE;

    // Send ACK
    HAL_UART_Transmit(&huart2, &ack, 1, HAL_MAX_DELAY);

    // Receive features (26 floats = 104 bytes)
    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t*)features,
                                                 NUM_FEATURES_KWS * sizeof(float), 2000);
    if (status != HAL_OK) {
        printf("ERROR: Feature receive timeout!\r\n");
        return;
    }

    uint32_t start = HAL_GetTick();

    // Layer 0: Input(26) -> Hidden1(100), ReLU
    MatMulAdd(features, kws_layer0_weights, kws_layer0_bias,
              layer_output, KWS_LAYER0_INPUT_SIZE, KWS_LAYER0_OUTPUT_SIZE);
    ReLU(layer_output, KWS_LAYER0_OUTPUT_SIZE);

    // Copy for next layer
    memcpy(layer_input, layer_output, KWS_LAYER0_OUTPUT_SIZE * sizeof(float));

    // Layer 1: Hidden1(100) -> Hidden2(100), ReLU
    MatMulAdd(layer_input, kws_layer1_weights, kws_layer1_bias,
              layer_output, KWS_LAYER1_INPUT_SIZE, KWS_LAYER1_OUTPUT_SIZE);
    ReLU(layer_output, KWS_LAYER1_OUTPUT_SIZE);

    // Copy for next layer
    memcpy(layer_input, layer_output, KWS_LAYER1_OUTPUT_SIZE * sizeof(float));

    // Layer 2: Hidden2(100) -> Output(10), Softmax
    MatMulAdd(layer_input, kws_layer2_weights, kws_layer2_bias,
              predictions, KWS_LAYER2_INPUT_SIZE, KWS_LAYER2_OUTPUT_SIZE);
    Softmax(predictions, NUM_CLASSES);

    uint32_t elapsed = HAL_GetTick() - start;

    // Send predictions back (10 floats = 40 bytes)
    HAL_UART_Transmit(&huart2, (uint8_t*)predictions, NUM_CLASSES * sizeof(float), HAL_MAX_DELAY);

    // Find argmax and print
    int predicted = 0;
    float max_prob = predictions[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            predicted = i;
        }
    }
    printf("KWS: Digit=%d (%.1f%%) Time=%lums\r\n", predicted, max_prob * 100.0f, elapsed);
}

/**
  * @brief Run HDR MLP inference
  *        Architecture: 7 -> 100 (ReLU) -> 100 (ReLU) -> 10 (Softmax)
  */
static void RunHDRInference(void)
{
    float features[NUM_FEATURES_HDR];
    float predictions[NUM_CLASSES];
    uint8_t ack = ACK_BYTE;

    // Send ACK
    HAL_UART_Transmit(&huart2, &ack, 1, HAL_MAX_DELAY);

    // Receive features (7 floats = 28 bytes)
    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t*)features,
                                                 NUM_FEATURES_HDR * sizeof(float), 2000);
    if (status != HAL_OK) {
        printf("ERROR: Feature receive timeout!\r\n");
        return;
    }

    uint32_t start = HAL_GetTick();

    // Layer 0: Input(7) -> Hidden1(100), ReLU
    MatMulAdd(features, hdr_layer0_weights, hdr_layer0_bias,
              layer_output, HDR_LAYER0_INPUT_SIZE, HDR_LAYER0_OUTPUT_SIZE);
    ReLU(layer_output, HDR_LAYER0_OUTPUT_SIZE);

    // Copy for next layer
    memcpy(layer_input, layer_output, HDR_LAYER0_OUTPUT_SIZE * sizeof(float));

    // Layer 1: Hidden1(100) -> Hidden2(100), ReLU
    MatMulAdd(layer_input, hdr_layer1_weights, hdr_layer1_bias,
              layer_output, HDR_LAYER1_INPUT_SIZE, HDR_LAYER1_OUTPUT_SIZE);
    ReLU(layer_output, HDR_LAYER1_OUTPUT_SIZE);

    // Copy for next layer
    memcpy(layer_input, layer_output, HDR_LAYER1_OUTPUT_SIZE * sizeof(float));

    // Layer 2: Hidden2(100) -> Output(10), Softmax
    MatMulAdd(layer_input, hdr_layer2_weights, hdr_layer2_bias,
              predictions, HDR_LAYER2_INPUT_SIZE, HDR_LAYER2_OUTPUT_SIZE);
    Softmax(predictions, NUM_CLASSES);

    uint32_t elapsed = HAL_GetTick() - start;

    // Send predictions back (10 floats = 40 bytes)
    HAL_UART_Transmit(&huart2, (uint8_t*)predictions, NUM_CLASSES * sizeof(float), HAL_MAX_DELAY);

    // Find argmax and print
    int predicted = 0;
    float max_prob = predictions[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            predicted = i;
        }
    }
    printf("HDR: Digit=%d (%.1f%%) Time=%lums\r\n", predicted, max_prob * 100.0f, elapsed);
}

/**
  * @brief Print menu
  */
static void PrintMenu(void)
{
    printf("\r\n");
    printf("============================================\r\n");
    printf("  EE4065 HW5 - STM32 Nucleo-F446RE\r\n");
    printf("  (Custom MLP Inference - No TFLite)\r\n");
    printf("============================================\r\n");
    printf("  Current Mode: ");
    switch(current_mode) {
        case MODE_KWS: printf("Q1 - Keyword Spotting\r\n"); break;
        case MODE_HDR: printf("Q2 - Digit Recognition\r\n"); break;
        default: printf("IDLE\r\n"); break;
    }
    printf("--------------------------------------------\r\n");
    printf("  Commands:\r\n");
    printf("    1 - Select Q1 (Keyword Spotting)\r\n");
    printf("    2 - Select Q2 (Handwritten Digit)\r\n");
    printf("    i - System info\r\n");
    printf("    ? - This menu\r\n");
    printf("============================================\r\n");
}

/**
  * @brief Print system info
  */
static void PrintInfo(void)
{
    printf("\r\n>>> System Info:\r\n");
    printf("  MCU: STM32F446RE\r\n");
    printf("  Clock: %lu MHz\r\n", HAL_RCC_GetSysClockFreq() / 1000000);
    printf("  KWS Model: 26->100->100->10 MLP\r\n");
    printf("  HDR Model: 7->100->100->10 MLP\r\n");
    printf("  Inference: Custom (No TFLite)\r\n");
    printf("  Mode: %s\r\n", current_mode == MODE_KWS ? "KWS" :
                            current_mode == MODE_HDR ? "HDR" : "IDLE");
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  */
int main(void)
{
    /* MCU Configuration */
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART2_UART_Init();

    /* USER CODE BEGIN 2 */
    HAL_Delay(100);

    // Welcome message
    printf("\r\n\r\n");
    printf("********************************************\r\n");
    printf("*  EE4065 Embedded Machine Learning HW5   *\r\n");
    printf("*  STM32 Nucleo-F446RE                    *\r\n");
    printf("*  Custom MLP Inference Engine            *\r\n");
    printf("********************************************\r\n");

    // Default to KWS mode
    current_mode = MODE_KWS;

    PrintMenu();
    printf("\r\nReady. Waiting for commands...\r\n");

    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    uint8_t rx_byte;

    while (1)
    {
        // Wait for incoming byte
        if (HAL_UART_Receive(&huart2, &rx_byte, 1, 100) == HAL_OK)
        {
            // Handle command
            switch (rx_byte)
            {
                case '1':
                    // Select KWS mode
                    current_mode = MODE_KWS;
                    printf("Mode: Q1 - Keyword Spotting (26 MFCC features)\r\n");
                    break;

                case '2':
                    // Select HDR mode
                    current_mode = MODE_HDR;
                    printf("Mode: Q2 - Handwritten Digit Recognition (7 Hu Moments)\r\n");
                    break;

                case 'i':
                case 'I':
                    PrintInfo();
                    break;

                case '?':
                    PrintMenu();
                    break;

                case SYNC_BYTE:
                    // Inference request from PC
                    if (current_mode == MODE_KWS) {
                        RunKWSInference();
                    } else if (current_mode == MODE_HDR) {
                        RunHDRInference();
                    } else {
                        printf("ERROR: No mode selected!\r\n");
                    }
                    break;

                case '\r':
                case '\n':
                    // Ignore newlines
                    break;

                default:
                    // Unknown command - ignore
                    break;
            }
        }

        // Blink LED to show we're alive
        static uint32_t last_blink = 0;
        if (HAL_GetTick() - last_blink > 1000) {
            HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
            last_blink = HAL_GetTick();
        }
    }
    /* USER CODE END WHILE */
    /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration - 180 MHz
  */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 8;
    RCC_OscInitStruct.PLL.PLLN = 180;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = 2;
    RCC_OscInitStruct.PLL.PLLR = 2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    if (HAL_PWREx_EnableOverDrive() != HAL_OK) {
        Error_Handler();
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
        Error_Handler();
    }
}

/**
  * @brief USART2 Initialization - 115200 baud
  */
static void MX_USART2_UART_Init(void)
{
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) {
        Error_Handler();
    }
}

/**
  * @brief GPIO Initialization
  */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // LED (PA5 on Nucleo-F446RE)
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Button (PC13 on Nucleo-F446RE)
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

/**
  * @brief Error Handler
  */
void Error_Handler(void)
{
    __disable_irq();
    while (1) {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
        for(volatile int i = 0; i < 100000; i++);
    }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    printf("Assert failed: %s line %lu\r\n", file, line);
}
#endif

