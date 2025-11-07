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
#include "mandrill.h" // Önceki adımda oluşturduğunuz dosya
#include <math.h> // Gamma düzeltmesi için (powf fonksiyonu)
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
// Orijinal görüntünün boyutunu (Python betiğinden) biliyoruz
// Örn: 160x120 = 19200
#define IMG_WIDTH   160
#define IMG_HEIGHT  120
#define IMG_SIZE    (IMG_WIDTH * IMG_HEIGHT)

// İşlenmiş görüntüleri saklamak için STATIK diziler
static unsigned char g_negativeImage[IMG_SIZE];
static unsigned char g_thresholdImage[IMG_SIZE];
static unsigned char g_gammaImage_3[IMG_SIZE];      // Gamma = 3 için
static unsigned char g_gammaImage_1_3[IMG_SIZE];  // Gamma = 1/3 için
static unsigned char g_piecewiseImage[IMG_SIZE];

// Gamma düzeltmesi için Arama Tabloları (LUT)
static unsigned char g_gammaLUT_3[256];     // Gamma = 3.0 için
static unsigned char g_gammaLUT_1_3[256]; // Gamma = 1/3 için

static unsigned char g_piecewiseLUT[256];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

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
  // Orijinal görüntü dizisine bir pointer alalım
  const unsigned char* originalImage = GRAYSCALE_IMG_ARRAY;

  // Q2-a: Negatif Görüntü
  for (int i = 0; i < IMG_SIZE; i++)
  {
      g_negativeImage[i] = 255 - originalImage[i];
  }

  // Q2-b: Eşikleme
  #define THRESHOLD_VALUE 128
  for (int i = 0; i < IMG_SIZE; i++)
  {
      if (originalImage[i] > THRESHOLD_VALUE)
      {
          g_thresholdImage[i] = 255;
      }
      else
      {
          g_thresholdImage[i] = 0;
      }

      // Veya daha kısa yolu:
      // g_thresholdImage[i] = (originalImage[i] > THRESHOLD_VALUE) ? 255 : 0;
  }

  // Q2-c: Gamma Düzeltmesi
  float gamma_3_0 = 3.0f;
  float gamma_1_3 = 1.0f / 3.0f;

  // 1. Adım: Arama Tablolarını (LUT) bir kez doldur
  for (int i = 0; i < 256; i++)
  {
      // Değeri 0-1 arasına normalize et
      float normalized_val = (float)i / 255.0f;

      // Gamma = 3.0 için hesapla
      float corrected_val_3_0 = powf(normalized_val, gamma_3_0);
      g_gammaLUT_3[i] = (unsigned char)(corrected_val_3_0 * 255.0f);

      // Gamma = 1/3 için hesapla
      float corrected_val_1_3 = powf(normalized_val, gamma_1_3);
      g_gammaLUT_1_3[i] = (unsigned char)(corrected_val_1_3 * 255.0f);
  }

  // 2. Adım: Görüntüleri LUT kullanarak hızlıca işle
  for (int i = 0; i < IMG_SIZE; i++)
  {
      unsigned char pixel = originalImage[i];
      g_gammaImage_3[i]   = g_gammaLUT_3[pixel];
      g_gammaImage_1_3[i] = g_gammaLUT_1_3[pixel];
  }

  // Q2-d: Parçalı Doğrusal Dönüşüm (Kontrast Germe)
  #define r1 50
  #define s1 0
  #define r2 200
  #define s2 255

  // 1. Adım: Parçalı LUT'u doldur
  for (int i = 0; i < 256; i++)
  {
      if (i < r1)
      {
          g_piecewiseLUT[i] = s1; // r1'den küçükler s1'e (0)
      }
      else if (i > r2)
      {
          g_piecewiseLUT[i] = s2; // r2'den büyükler s2'ye (255)
      }
      else // r1 ile r2 arasında
      {
          // Doğrusal enterpolasyon: y = y1 + (x-x1) * (y2-y1)/(x2-x1)
          g_piecewiseLUT[i] = (unsigned char)(s1 + (i - r1) * ((float)(s2 - s1) / (float)(r2 - r1)));
      }
  }

  // 2. Adım: Görüntüyü LUT kullanarak hızlıca işle
  for (int i = 0; i < IMG_SIZE; i++)
  {
      g_piecewiseImage[i] = g_piecewiseLUT[originalImage[i]];
  }
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
#ifdef USE_FULL_ASSERT
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
