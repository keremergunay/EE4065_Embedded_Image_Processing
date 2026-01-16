import Image_Header_Library as headerGenerator

Img = "D:\Projects\Embedded\PC_Python\mandrill.tiff"
outputFileName = "mandrill"

width = 160
height = 120

# headerGenerator.generate(Img, width, height, outputFileName, headerGenerator.SPI_C)

# headerGenerator.generate(Img, width, height, outputFileName, headerGenerator.SPI_Cpp)

# headerGenerator.generate(Img, width, height, outputFileName, headerGenerator.SPI_MP)

# headerGenerator.generate(Img, width, height, outputFileName, headerGenerator.LTDC)

headerGenerator.generate(Img, width, height, outputFileName, headerGenerator.SPI_C_GRAYSCALE)