/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.cpp
  * @brief          : EE4065 Homework 6 - CNN Handwritten Digit Recognition
  *                   TensorFlow Lite Micro on STM32 Nucleo-F446RE
  *
  * Models tested (select ONE at a time by including the appropriate header):
  *   - SqueezeNetMini:    54.2 KB  (RECOMMENDED - Smallest)
  *   - ResNet8:           96.3 KB
  *   - MobileNetV2Mini:  106.5 KB
  *   - EfficientNetMini: 108.5 KB
  *   - ResNet14:         201.7 KB  (May need smaller arena)
  *
  * Protocol (PC -> STM32 -> PC):
  *   1. PC sends 0xAA (sync byte)
  *   2. STM32 replies 0x55 (ACK)
  *   3. PC sends 1024 bytes (32x32 grayscale image)
  *   4. STM32 runs inference
  *   5. STM32 sends 10 bytes (uint8 predictions) + 4 bytes (inference time ms)
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* USER CODE BEGIN Includes */
// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

// ============================================================================
// SELECT YOUR MODEL HERE - Change the number (1-5)
// ============================================================================
#define MODEL_SELECT 3  // 1=SqueezeNetMini, 2=ResNet8, 3=MobileNetV2Mini, 4=EfficientNetMini, 5=ResNet14

// Auto-configure based on MODEL_SELECT
#if MODEL_SELECT == 1
    #include "SqueezeNetMini.h"
    #define MODEL_NAME "SqueezeNetMini"
    #define MODEL_DATA SqueezeNetMini_tflite
    #define TENSOR_ARENA_SIZE (50 * 1024)
#elif MODEL_SELECT == 2
    #include "ResNet8.h"
    #define MODEL_NAME "ResNet8"
    #define MODEL_DATA ResNet8_tflite
    #define TENSOR_ARENA_SIZE (60 * 1024)
#elif MODEL_SELECT == 3
    #include "MobileNetV2Mini.h"
    #define MODEL_NAME "MobileNetV2Mini"
    #define MODEL_DATA MobileNetV2Mini_tflite
    #define TENSOR_ARENA_SIZE (65 * 1024)
#elif MODEL_SELECT == 4
    #include "EfficientNetMini.h"
    #define MODEL_NAME "EfficientNetMini"
    #define MODEL_DATA EfficientNetMini_tflite
    #define TENSOR_ARENA_SIZE (80 * 1024)  // Increased from 70KB to 80KB
#elif MODEL_SELECT == 5
    #include "ResNet14.h"
    #define MODEL_NAME "ResNet14"
    #define MODEL_DATA ResNet14_tflite
    #define TENSOR_ARENA_SIZE (80 * 1024)
#else
    #error "Invalid MODEL_SELECT! Use 1-5"
#endif
/* USER CODE END Includes */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// Image parameters
#define IMAGE_SIZE      32
#define IMAGE_CHANNELS  3       // Model expects RGB
#define IMAGE_BYTES     (IMAGE_SIZE * IMAGE_SIZE)  // Grayscale input from PC
#define INPUT_SIZE      (IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS)

// Output
#define NUM_CLASSES     10

// Protocol bytes
#define SYNC_BYTE       0xAA
#define ACK_BYTE        0x55

/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// Tensor arena - must fit in RAM (F446RE has 128KB)
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Image receive buffer
static uint8_t image_buffer[IMAGE_BYTES];

// TFLite objects
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static bool model_ready = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN PFP */
static bool InitModel(void);
static void RunInference(void);
static void PrintMenu(void);
static void PrintInfo(void);
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
  * @brief Initialize TFLite Micro model
  */
static bool InitModel(void)
{
    printf("Initializing TFLite model...\r\n");
    printf("  Model: %s\r\n", MODEL_NAME);
    printf("  Arena size: %d bytes\r\n", TENSOR_ARENA_SIZE);

    // Get model from Flash
    // Change this line when using different model!
    const tflite::Model* model = tflite::GetModel(MODEL_DATA);
    // const tflite::Model* model = tflite::GetModel(SqueezeNetMini_tflite);
    // const tflite::Model* model = tflite::GetModel(ResNet8_tflite);
    // const tflite::Model* model = tflite::GetModel(MobileNetV2Mini_tflite);
    // const tflite::Model* model = tflite::GetModel(EfficientNetMini_tflite);
    // const tflite::Model* model = tflite::GetModel(ResNet14_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: Model schema version mismatch!\r\n");
        return false;
    }

    // Create op resolver with required operations
    static tflite::MicroMutableOpResolver<25> resolver;  // Increased from 20 to 25

    // Common ops for all models
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddMean();
    resolver.AddPad();

    // Activation ops
    resolver.AddRelu();
    resolver.AddRelu6();

    // Arithmetic ops
    resolver.AddAdd();
    resolver.AddMul();

    // Quantization ops
    resolver.AddQuantize();
    resolver.AddDequantize();

    // Concatenation (for SqueezeNet fire modules)
    resolver.AddConcatenation();

    // === ADD THESE FOR ResNet and EfficientNet ===
    resolver.AddLogistic();        // Sigmoid - needed for EfficientNet Swish activation
    resolver.AddSub();             // Subtraction
    resolver.AddDiv();             // Division
    resolver.AddResizeBilinear();  // For some models
    resolver.AddResizeNearestNeighbor();

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors failed!\r\n");
        printf("  Try reducing TENSOR_ARENA_SIZE or use smaller model.\r\n");
        return false;
    }

    // Get input/output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Print tensor info
    printf("  Input:  [%d, %d, %d, %d] type=%d\r\n",
           input_tensor->dims->data[0], input_tensor->dims->data[1],
           input_tensor->dims->data[2], input_tensor->dims->data[3],
           input_tensor->type);
    printf("  Output: [%d, %d] type=%d\r\n",
           output_tensor->dims->data[0], output_tensor->dims->data[1],
           output_tensor->type);

    printf("Model initialized successfully!\r\n\r\n");
    return true;
}

/**
  * @brief Run inference on received image
  */
static void RunInference(void)
{
    uint8_t ack = ACK_BYTE;

    // Send ACK
    HAL_UART_Transmit(&huart2, &ack, 1, HAL_MAX_DELAY);

    // Receive image (32x32 = 1024 bytes grayscale)
    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, image_buffer,
                                                 IMAGE_BYTES, 5000);
    if (status != HAL_OK) {
        printf("ERROR: Image receive timeout!\r\n");
        return;
    }

    // Preprocess: convert grayscale to RGB and copy to input tensor
    // Input tensor expects uint8 values
    uint8_t* input_data = input_tensor->data.uint8;

    for (int i = 0; i < IMAGE_BYTES; i++) {
        uint8_t pixel = image_buffer[i];
        // Replicate grayscale to RGB channels
        input_data[i * 3 + 0] = pixel;  // R
        input_data[i * 3 + 1] = pixel;  // G
        input_data[i * 3 + 2] = pixel;  // B
    }

    // Run inference
    uint32_t start_time = HAL_GetTick();

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: Inference failed!\r\n");
        return;
    }

    uint32_t inference_time = HAL_GetTick() - start_time;

    // Get output (10 uint8 values)
    uint8_t* output_data = output_tensor->data.uint8;

    // Send predictions (10 bytes)
    HAL_UART_Transmit(&huart2, output_data, NUM_CLASSES, HAL_MAX_DELAY);

    // Send inference time (4 bytes, uint32)
    HAL_UART_Transmit(&huart2, (uint8_t*)&inference_time, 4, HAL_MAX_DELAY);

    // Find argmax and print locally
    int predicted = 0;
    uint8_t max_val = output_data[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            predicted = i;
        }
    }

    printf("Inference: Digit=%d (conf=%d) Time=%lums\r\n",
           predicted, max_val, inference_time);
}

/**
  * @brief Print menu
  */
static void PrintMenu(void)
{
    printf("\r\n");
    printf("================================================\r\n");
    printf("  EE4065 HW6 - CNN Digit Recognition\r\n");
    printf("  STM32 Nucleo-F446RE + TFLite Micro\r\n");
    printf("================================================\r\n");
    printf("  Model: %s\r\n", MODEL_NAME);
    printf("  Status: %s\r\n", model_ready ? "READY" : "NOT READY");
    printf("------------------------------------------------\r\n");
    printf("  Commands:\r\n");
    printf("    0xAA - Start inference (send image)\r\n");
    printf("    i    - System info\r\n");
    printf("    ?    - This menu\r\n");
    printf("================================================\r\n");
}

/**
  * @brief Print system info
  */
static void PrintInfo(void)
{
    printf("\r\n>>> System Info:\r\n");
    printf("  MCU: STM32F446RE (Cortex-M4 @ 180MHz)\r\n");
    printf("  Flash: 512 KB\r\n");
    printf("  RAM: 128 KB\r\n");
    printf("  Clock: %lu MHz\r\n", HAL_RCC_GetSysClockFreq() / 1000000);
    printf("  Model: %s\r\n", MODEL_NAME);
    printf("  Arena: %d bytes\r\n", TENSOR_ARENA_SIZE);
    printf("  Input: %dx%dx%d (uint8)\r\n", IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS);
    printf("  Output: %d classes (uint8)\r\n", NUM_CLASSES);
    printf("  Status: %s\r\n", model_ready ? "Ready" : "Error");
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
    printf("************************************************\r\n");
    printf("*  EE4065 Embedded Machine Learning HW6       *\r\n");
    printf("*  CNN Handwritten Digit Recognition          *\r\n");
    printf("*  STM32 Nucleo-F446RE + TFLite Micro         *\r\n");
    printf("************************************************\r\n\r\n");

    // Initialize model
    model_ready = InitModel();

    if (model_ready) {
        PrintMenu();
        printf("\r\nReady. Waiting for images...\r\n");
    } else {
        printf("\r\nERROR: Model initialization failed!\r\n");
        printf("Check arena size and model compatibility.\r\n");
    }

    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    uint8_t rx_byte;

    while (1)
    {
        // Toggle LED to show we're alive
        static uint32_t last_blink = 0;
        if (HAL_GetTick() - last_blink > 1000) {
            HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
            last_blink = HAL_GetTick();
        }

        // Check for incoming commands
        if (HAL_UART_Receive(&huart2, &rx_byte, 1, 100) == HAL_OK)
        {
            switch (rx_byte)
            {
                case SYNC_BYTE:
                    // Inference request from PC
                    if (model_ready) {
                        RunInference();
                    } else {
                        printf("ERROR: Model not ready!\r\n");
                    }
                    break;

                case 'i':
                case 'I':
                    PrintInfo();
                    break;

                case '?':
                    PrintMenu();
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

