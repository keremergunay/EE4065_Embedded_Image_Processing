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
#include "mandrill.h"
#include <math.h>
#include <string.h>
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

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
#define UART_HANDLE &huart2
/* USER CODE BEGIN PV */
// Q1: Grayscale (Flash'tan okunur, RAM'e yazılır)
#define GRAY_WIDTH   160
#define GRAY_HEIGHT  120
#define GRAY_SIZE    (GRAY_WIDTH * GRAY_HEIGHT)
uint8_t binary_image[GRAY_SIZE];

// Q2: Color (RAM'e yazılır ve okunur)
#define RGB_WIDTH    160
#define RGB_HEIGHT   120
#define RGB_SIZE     (RGB_WIDTH * RGB_HEIGHT * 3)
uint8_t rgb_image[RGB_SIZE];

#define SE_SIZE 3
#define BORDER (SE_SIZE / 2) // 1
uint8_t temp_buffer[GRAY_SIZE];

uint8_t morph_result[GRAY_SIZE];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
/* -------------------- MORFOLOJİK İŞLEMLER -------------------- */

// EROZYON (Korozyon)
// Merkeze denk gelen tüm SE pikselleri SİYAH (0) ise, merkez piksel siyah olur.
void erosion(const uint8_t *input_img, uint8_t *output_img) {
    // Kenarları temizle (Siyah yap) - Sınır kontrolü yerine basitçe çerçeve çiziyoruz
    for (int i = 0; i < GRAY_SIZE; i++) output_img[i] = 0;

    for (int y = BORDER; y < GRAY_HEIGHT - BORDER; y++) {
        for (int x = BORDER; x < GRAY_WIDTH - BORDER; x++) {

            // Eğer merkez piksel SİYAH ise, sonuç kesinlikle siyahtır (işleme gerek yok)
            if (input_img[y * GRAY_WIDTH + x] == 0) {
                output_img[y * GRAY_WIDTH + x] = 0;
                continue;
            }

            // Merkez BEYAZ ise, komşulara bak
            int keep_white = 1;
            for (int i = -BORDER; i <= BORDER; i++) {
                for (int j = -BORDER; j <= BORDER; j++) {
                    // Herhangi bir komşu SİYAH ise, erozyon gerçekleşir
                    if (input_img[(y + i) * GRAY_WIDTH + (x + j)] == 0) {
                        keep_white = 0;
                        goto e_break; // Hızlı çıkış
                    }
                }
            }
        e_break:
            output_img[y * GRAY_WIDTH + x] = (keep_white) ? 255 : 0;
        }
    }
}

// GENİŞLEME (Dilation)
// Merkeze denk gelen tüm SE piksellerinden en az biri BEYAZ (255) ise, merkez piksel beyaz olur.
void dilation(const uint8_t *input_img, uint8_t *output_img) {
    // Kenarları temizle
    for (int i = 0; i < GRAY_SIZE; i++) output_img[i] = 0;

    for (int y = BORDER; y < GRAY_HEIGHT - BORDER; y++) {
        for (int x = BORDER; x < GRAY_WIDTH - BORDER; x++) {

            // Eğer merkez BEYAZ ise, sonuç kesinlikle beyazdır
            if (input_img[y * GRAY_WIDTH + x] == 255) {
                output_img[y * GRAY_WIDTH + x] = 255;
                continue;
            }

            // Merkez SİYAH ise, komşulara bak (Herhangi biri beyaz mı?)
            int make_white = 0;
            for (int i = -BORDER; i <= BORDER; i++) {
                for (int j = -BORDER; j <= BORDER; j++) {
                    // Herhangi bir komşu BEYAZ ise genişleme olur
                    if (input_img[(y + i) * GRAY_WIDTH + (x + j)] == 255) {
                        make_white = 1;
                        goto d_break;
                    }
                }
            }
        d_break:
            output_img[y * GRAY_WIDTH + x] = (make_white) ? 255 : 0;
        }
    }
}

// AÇMA (Opening): Erosion -> Dilation
void opening(const uint8_t *input_img, uint8_t *output_img) {
    // 1. Erosion: Input -> Temp (rgb_image)
    erosion(input_img, rgb_image);

    // 2. Dilation: Temp (rgb_image) -> Output
    dilation(rgb_image, output_img);
}

// KAPAMA (Closing): Dilation -> Erosion
void closing(const uint8_t *input_img, uint8_t *output_img) {
    // 1. Dilation: Input -> Temp (rgb_image)
    dilation(input_img, rgb_image);

    // 2. Erosion: Temp (rgb_image) -> Output
    erosion(rgb_image, output_img);
}

void otsu_gray_process(void) {
    int histogram[256] = {0};
    const uint8_t *input_img = GRAYSCALE_IMG_ARRAY; // Header dosyasından

    // 1. Histogram
    for (int i = 0; i < GRAY_SIZE; i++) {
        histogram[input_img[i]]++;
    }

    double total_pixels = (double)GRAY_SIZE;
    double current_max_variance = 0.0;
    int optimal_threshold = 0;
    double sum_total = 0.0;

    for (int i = 0; i < 256; i++) sum_total += i * histogram[i];

    double sum_background = 0.0;
    double weight_background = 0.0;
    double weight_foreground = 0.0;

    // 2. Threshold Bulma
    for (int t = 0; t < 256; t++) {
        weight_background += histogram[t];
        if (weight_background == 0) continue;
        weight_foreground = total_pixels - weight_background;
        if (weight_foreground == 0) break;

        sum_background += (double)(t * histogram[t]);
        double mean_b = sum_background / weight_background;
        double mean_f = (sum_total - sum_background) / weight_foreground;
        double var_between = weight_background * weight_foreground * pow((mean_b - mean_f), 2);

        if (var_between > current_max_variance) {
            current_max_variance = var_between;
            optimal_threshold = t;
        }
    }

    // 3. Binary'e Çevirme
    for (int i = 0; i < GRAY_SIZE; i++) {
        binary_image[i] = (input_img[i] > optimal_threshold) ? 255 : 0;
    }
}


/* --- Q2: COLOR OTSU FONKSİYONLARI --- */
int calculate_single_channel_otsu(uint8_t *data, int step, int offset, int total_pixels) {
    int histogram[256] = {0};
    for (int i = 0; i < total_pixels; i++) {
        histogram[data[i * step + offset]]++;
    }

    double total = (double)total_pixels;
    double max_var = 0.0;
    int threshold = 0;
    double sum_total = 0.0;

    for(int i=0; i<256; i++) sum_total += i * histogram[i];

    double sum_bg = 0.0;
    double w_bg = 0.0, w_fg = 0.0;

    for (int t = 0; t < 256; t++) {
        w_bg += histogram[t];
        if (w_bg == 0) continue;
        w_fg = total - w_bg;
        if (w_fg == 0) break;

        sum_bg += (double)(t * histogram[t]);
        double m_b = sum_bg / w_bg;
        double m_f = (sum_total - sum_bg) / w_fg;
        double var = w_bg * w_fg * (m_b - m_f) * (m_b - m_f);

        if (var > max_var) {
            max_var = var;
            threshold = t;
        }
    }
    return threshold;
}

void process_color_otsu(void) {
    int total_pixels = RGB_WIDTH * RGB_HEIGHT;

    // R, G, B için ayrı eşik değerleri
    int th_r = calculate_single_channel_otsu(rgb_image, 3, 0, total_pixels);
    int th_g = calculate_single_channel_otsu(rgb_image, 3, 1, total_pixels);
    int th_b = calculate_single_channel_otsu(rgb_image, 3, 2, total_pixels);

    for (int i = 0; i < total_pixels; i++) {
        // Red Kanalı
        rgb_image[i*3 + 0] = (rgb_image[i*3 + 0] > th_r) ? 255 : 0;
        // Green Kanalı
        rgb_image[i*3 + 1] = (rgb_image[i*3 + 1] > th_g) ? 255 : 0;
        // Blue Kanalı
        rgb_image[i*3 + 2] = (rgb_image[i*3 + 2] > th_b) ? 255 : 0;
    }
}
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
  /* USER CODE END 2 */
  uint8_t cmd = 0;
  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  // PC'den 1 byte komut bekle (Bloklayıcı)
	  if (HAL_UART_Receive(&huart2, &cmd, 1, HAL_MAX_DELAY) == HAL_OK)
	  {
	      /* --- SENARYO 1: Q1 GRAYSCALE --- */
	      if (cmd == '1')
	      {
	          // İşlemi yap
	          otsu_gray_process();
	          // Sonucu yolla
	          HAL_UART_Transmit(&huart2, binary_image, GRAY_SIZE, HAL_MAX_DELAY);
	      }

	      /* --- SENARYO 2: Q2 RESİM YÜKLEME VE İŞLEME --- */
	      else if (cmd == '2')
	      {
	          // Onay gönder ('A' - Ack)
	          uint8_t ack = 'A';
	          HAL_UART_Transmit(&huart2, &ack, 1, 100);

	          // 57.6KB Renkli Veriyi Al (Timeout yüksek olmalı)
	          if (HAL_UART_Receive(&huart2, rgb_image, RGB_SIZE, 10000) == HAL_OK)
	          {
	              // İşle
	              process_color_otsu();
	              // İşlem Bitti onayı ('D' - Done)
	              uint8_t done = 'D';
	              HAL_UART_Transmit(&huart2, &done, 1, 100);
	          }
	      }

	      /* --- SENARYO 3: Q2 SONUCU OKUMA --- */
	      else if (cmd == '3')
	      {
	          // İşlenmiş renkli veriyi geri yolla
	          HAL_UART_Transmit(&huart2, rgb_image, RGB_SIZE, HAL_MAX_DELAY);
	      }
	      /* --- Q3: EROZYON --- */
	      else if (cmd == '4')
	      {
	          // Q1 sonucunu (binary_image) kullan
	          erosion(binary_image, morph_result);
	          HAL_UART_Transmit(&huart2, morph_result, GRAY_SIZE, HAL_MAX_DELAY);
	      }

	      /* --- Q3: GENİŞLEME --- */
	      else if (cmd == '5')
	      {
	          dilation(binary_image, morph_result);
	          HAL_UART_Transmit(&huart2, morph_result, GRAY_SIZE, HAL_MAX_DELAY);
	      }

	      /* --- Q3: AÇMA (OPENING) --- */
	      else if (cmd == '6')
	      {
	          // Önce Q1 sonucunu kopyala, çünkü opening in-place işlem yapar

	          opening(binary_image, morph_result);
	          HAL_UART_Transmit(&huart2, morph_result, GRAY_SIZE, HAL_MAX_DELAY);
	      }

	      /* --- Q3: KAPAMA (CLOSING) --- */
	      else if (cmd == '7')
	      {
	          // Önce Q1 sonucunu kopyala
	          closing(binary_image, morph_result);
	          HAL_UART_Transmit(&huart2, morph_result, GRAY_SIZE, HAL_MAX_DELAY);
	      }
	  }
      /* USER CODE END WHILE */
  }
  /* USER CODE BEGIN 3 */

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
