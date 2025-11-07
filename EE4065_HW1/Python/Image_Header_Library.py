import cv2
import numpy as np

SPI_C = 1
SPI_C_GRAYSCALE = 2
SPI_Cpp = 3
LTDC = 4
SPI_MP = 5


def generate(filename, width, height, outputFileName, format):

	Img = cv2.imread(filename)
	Img = cv2.resize(Img, (width, height))

	if format == SPI_C:
		spi_c_generate(Img, outputFileName)

	elif format == SPI_C_GRAYSCALE:
		spi_c_generate_grayscale(Img, outputFileName)

	elif format == SPI_Cpp:
		spi_cpp_generate(Img, outputFileName)

	elif format == SPI_MP:
		spi_mp_generate(Img, outputFileName)

	elif format == LTDC:
		ltdc_generate(Img, outputFileName)


def spi_c_generate(im, outputFileName):
    f = open(outputFileName + ".h", "w+")

    height, width, _ = im.shape
    img565 = cv2.cvtColor(im, cv2.COLOR_BGR2BGR565)
    img565 = img565.astype(np.uint8)

	# Byte swap -> Opencv and ili9341 are not compatiable in byte order.
    imgh = img565[:, :, 0].copy()
    img565[:, :, 0] = img565[:, :, 1].copy()
    img565[:, :, 1] = imgh.copy()

	# 2D to 1D conversion
    img565 = np.reshape(img565, (width*height*2))

    f.write("uint8_t RGB565_IMG_ARRAY[%d]={\n" % (width*height*2))

    for i in range(width*height*2):
        f.write("%s, " % hex(img565[i]))
        if i != 0 and i % 20 == 0:
            f.write("\n")

    f.write("};\n\n\n")
    f.write("ImageTypeDef RGB565_IMG = {\n")
    f.write(".pData = RGB565_IMG_ARRAY,\n")
    f.write(".width = %d,\n" % (width))
    f.write(".height = %d,\n" % (height))
    f.write(".size = %d,\n" % (width*height*2))
    f.write(".format = 1\n")
    f.write("};\n\n")

    f.close()

def spi_c_generate_grayscale(im, outputFileName):
    """
    Bu fonksiyon, görüntüyü 8-bit grayscale C header dosyasına dönüştürür.
    (Ödev 1a için güncellendi)
    """
    f = open(outputFileName + ".h", "w+")

    # Görüntüyü grayscale'e (gri tonlamalı) çevir
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Görüntü boyutlarını al (grayscale görüntü (height, width) döndürür)
    height, width = im_gray.shape
    
    # Toplam piksel sayısı (ve byte sayısı, çünkü her piksel 1 byte)
    array_size = width * height

    print(f"Header dosyasi (SPI_C - Grayscale) '{outputFileName}.h' olusturuluyor...")
    print(f"Cozunurluk: {width} x {height}")
    print(f"Toplam Boyut: {array_size} bytes")

    # 2D diziyi 1D (düz) bir diziye çevir
    img_flat = np.reshape(im_gray, (array_size))

    f.write(f"// Format: 8-bit Grayscale, Cozunurluk: {width}x{height}\n")
    # Dizi adını ödeve uygun olarak (veya genel) değiştirelim
    f.write(f"unsigned char GRAYSCALE_IMG_ARRAY[{array_size}] = {{\n")

    for i in range(array_size):
        f.write("%s, " % hex(img_flat[i]))
        if (i + 1) % 20 == 0:
            f.write("\n")

    f.write("\n};\n\n\n")
    
    # Projedeki struct tanımına uygun bir yapı oluşturalım
    # Orijinal kodunuzdaki ImageTypeDef'e benzer bir yapı
    f.write("/* Projenizdeki struct tanimina gore bu yapiyi guncelleyin (ImageTypeDef) */\n")
    f.write("typedef struct {\n")
    f.write("  const unsigned char* pData; /* Görüntü verisi pointer'ı */\n")
    f.write("  unsigned short width;     /* Görüntü genişliği */\n")
    f.write("  unsigned short height;    /* Görüntü yüksekliği */\n")
    f.write("  unsigned int   size;      /* Toplam boyut (byte) */\n")
    f.write("  unsigned int   format;    /* 0: Grayscale (varsayim) */\n")
    f.write("} ImageTypeDef_t;\n\n")

    f.write(f"ImageTypeDef_t GRAYSCALE_IMG = {{\n")
    f.write(f"  .pData = GRAYSCALE_IMG_ARRAY,\n")
    f.write(f"  .width = {width},\n")
    f.write(f"  .height = {height},\n")
    f.write(f"  .size = {array_size},\n")
    f.write(f"  .format = 0  // 8-bit Grayscale\n")
    f.write("};\n\n")

    f.close()
    print("Dosya olusturma tamamlandi.")


def spi_cpp_generate(im, outputFileName):
    f = open(outputFileName + ".hpp", "w+")

    height, width, _ = im.shape
    img565 = cv2.cvtColor(im, cv2.COLOR_BGR2BGR565)
    img565 = img565.astype(np.uint8)

    # Byte swap -> Opencv and ili9341 are not compatiable in byte order.
    imgh = img565[:, :, 0].copy()
    img565[:, :, 0] = img565[:, :, 1].copy()
    img565[:, :, 1] = imgh.copy()

	# 2D to 1D conversion
    img565 = np.reshape(img565, (width*height*2))

    f.write("#include \"image.hpp\"\n\n\n")
    f.write("uint8_t RGB565_IMG_ARRAY[%d]={\n" % (width*height*2))

    for i in range(width*height*2):
        f.write("%s, " % hex(img565[i]))
        if i != 0 and i % 20 == 0:
            f.write("\n")

    f.write("};\n\n\n")
    f.write("IMAGE RGB565_IMG;\n")

    f.close()


def spi_mp_generate(im, outputFileName):
    f = open(outputFileName + ".py", "w+")
    
    height, width, _ = im.shape
    img565 = cv2.cvtColor(im, cv2.COLOR_BGR2BGR565)
    img565 = img565.astype(np.uint8)
    
    # Byte swap -> Opencv and ili9341 are not compatiable in byte order.
    imgh = img565[:, :, 0].copy()
    img565[:, :, 0] = img565[:, :, 1].copy()
    img565[:, :, 1] = imgh.copy()
    
    # 2D to 1D conversion
    img565 = np.reshape(img565, (width*height*2))
    
    f.write("pData=[\n")
    
    for i in range(width*height*2):
        f.write("%s, " % hex(img565[i]))
        if i != 0 and i % 20 == 0:
            f.write("\n")
            
    f.write("]\n\n\n")
    f.write("width = %d\n" % (width))
    f.write("height = %d\n" % (height))
    f.write("size = %d\n" % (width*height*2))
    
    f.close()


def ltdc_generate(im, outputFileName):
    f = open(outputFileName + ".h", "w+")
    
    height, width, _ = im.shape
    img8888 = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    img8888 = img8888.astype(np.uint32)

	# 2D to 1D conversion
    img8888 = np.reshape(img8888, (width*height*4))
    
    f.write("uint8_t RGB8888_IMG_ARRAY[%d]={\n" % (width*height*4))

    for i in range(width*height*4):
        f.write("%s, " % hex(img8888[i]))
        if i != 0 and i % 20 == 0:
            f.write("\n")

    f.write("};\n")
    f.close()

