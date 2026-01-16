/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE BEGIN Includes */
#include "mandrill.h"  // Az önce sürüklediğin dosya
#include <string.h>
#include <stdlib.h>
/* USER CODE END Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
/* USER CODE BEGIN PV */

// Resim boyutları
#define IMG_W   160
#define IMG_H   120
#define IMG_SIZE (IMG_W * IMG_H)

uint8_t img_in[IMG_SIZE];
uint8_t img_eq[IMG_SIZE];
uint8_t img_low[IMG_SIZE];
uint8_t img_high[IMG_SIZE];
uint8_t img_med[IMG_SIZE];

// Histogramlar
uint32_t hist_orig[256];
uint32_t hist_eq[256];

/* USER CODE END PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void copy_image(const uint8_t *src, uint8_t *dst, uint32_t size);
void calc_histogram(const uint8_t *img, uint32_t size, uint32_t *hist);
void hist_equalize(const uint8_t *inImg, uint8_t *outImg, uint32_t size,
                   uint32_t *hist_in, uint32_t *hist_out);
void conv3x3(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h,
             const int8_t kernel[3][3], int divisor);
void low_pass_filter(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h);
void high_pass_filter(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h);
void median_filter3x3(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h);

void copy_image(const uint8_t *src, uint8_t *dst, uint32_t size) {
    for (uint32_t i = 0; i < size; i++)
        dst[i] = src[i];
}

void calc_histogram(const uint8_t *img, uint32_t size, uint32_t *hist) {
    for (int i = 0; i < 256; i++)
        hist[i] = 0;
    for (uint32_t i = 0; i < size; i++)
        hist[img[i]]++;
}

// integer tabanlı histogram equalization
void hist_equalize(const uint8_t *inImg, uint8_t *outImg, uint32_t size,
                   uint32_t *hist_in, uint32_t *hist_out) {
    uint32_t cdf[256];
    uint8_t  lut[256];

    calc_histogram(inImg, size, hist_in);

    cdf[0] = hist_in[0];
    for (int i = 1; i < 256; i++)
        cdf[i] = cdf[i - 1] + hist_in[i];

    for (int i = 0; i < 256; i++) {
        uint32_t num = cdf[i] * 255u + (size / 2u);
        lut[i] = (uint8_t)(num / size);
    }

    for (uint32_t i = 0; i < size; i++)
        outImg[i] = lut[inImg[i]];

    calc_histogram(outImg, size, hist_out);
}

// Genel 3x3 konvolüsyon, border'ları input'tan kopyalıyor
void conv3x3(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h,
             const int8_t kernel[3][3], int divisor) {
    // kenarlar: input'u direkt kopyala
    for (uint32_t x = 0; x < w; x++) {
        out[x] = in[x];
        out[(h - 1) * w + x] = in[(h - 1) * w + x];
    }
    for (uint32_t y = 0; y < h; y++) {
        out[y * w] = in[y * w];
        out[y * w + (w - 1)] = in[y * w + (w - 1)];
    }

    for (uint32_t y = 1; y < h - 1; y++) {
        for (uint32_t x = 1; x < w - 1; x++) {
            int32_t sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    uint8_t pixel = in[(y + ky) * w + (x + kx)];
                    sum += pixel * kernel[ky + 1][kx + 1];
                }
            }

            if (divisor != 0)
                sum /= divisor;

            if (sum < 0)   sum = 0;
            if (sum > 255) sum = 255;

            out[y * w + x] = (uint8_t)sum;
        }
    }
}

void low_pass_filter(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h) {
    const int8_t k[3][3] = {
        { 1, 1, 1 },
        { 1, 1, 1 },
        { 1, 1, 1 }
    };
    conv3x3(in, out, w, h, k, 9);
}

void high_pass_filter(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h) {
    const int8_t k[3][3] = {
        {  0, -1,  0 },
        { -1,  4, -1 },
        {  0, -1,  0 }
    };
    conv3x3(in, out, w, h, k, 1);
}

// 3x3 median filtresi
static void sort9(uint8_t *v) {
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (v[j] < v[i]) {
                uint8_t t = v[i];
                v[i] = v[j];
                v[j] = t;
            }
        }
    }
}

void median_filter3x3(const uint8_t *in, uint8_t *out, uint32_t w, uint32_t h) {
    // kenarlar: input'u kopyala
    for (uint32_t x = 0; x < w; x++) {
        out[x] = in[x];
        out[(h - 1) * w + x] = in[(h - 1) * w + x];
    }
    for (uint32_t y = 0; y < h; y++) {
        out[y * w] = in[y * w];
        out[y * w + (w - 1)] = in[y * w + (w - 1)];
    }

    uint8_t win[9];

    for (uint32_t y = 1; y < h - 1; y++) {
        for (uint32_t x = 1; x < w - 1; x++) {

            int k = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    win[k++] = in[(y + ky) * w + (x + kx)];
                }
            }

            sort9(win);
            out[y * w + x] = win[4];
        }
    }
}

/* USER CODE END 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  // burada HAL_Init vs kullanıyorsan kendi template'ine göre ekle

  // mandrill.h içindeki dizi ismine göre değiştir
  // ör: extern const uint8_t mandrill[IMG_SIZE];
  copy_image(grayscale_image, img_in, IMG_SIZE);

  // Q1: orijinal histogram
  calc_histogram(img_in, IMG_SIZE, hist_orig);

  // Q2: histogram equalization
  hist_equalize(img_in, img_eq, IMG_SIZE, hist_orig, hist_eq);

  // Q3: low-pass ve high-pass filtre
  low_pass_filter(img_in,   img_low,  IMG_W, IMG_H);
  high_pass_filter(img_in,  img_high, IMG_W, IMG_H);

  // Q4: median filtre
  median_filter3x3(img_in,  img_med,  IMG_W, IMG_H);

  // Buradan sonrası sadece bekleme; Memory Window'dan
  // img_in, img_eq, img_low, img_high, img_med,
  // hist_orig ve hist_eq dizilerini inceleyebilirsin.
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
