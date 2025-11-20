import numpy as np
import serial
import msvcrt
import cv2
import time

# -----------------------------
# PROTOKOL SABİTLERİ
# -----------------------------
MCU_WRITES = 87   # 'W'
MCU_READS  = 82   # 'R'

rqTypeText = {
    MCU_WRITES: "MCU -> PC (Image Send)",
    MCU_READS : "PC -> MCU (Image Request)"
}

formatText = {
    1: "Grayscale",
    2: "RGB565",
    3: "RGB888",
}

IMAGE_FORMAT_GRAYSCALE = 1
IMAGE_FORMAT_RGB565    = 2
IMAGE_FORMAT_RGB888    = 3

# -----------------------------
# SERIAL INIT
# -----------------------------
def SERIAL_Init(port):
    global __serial
    __serial = serial.Serial(port, 2000000, timeout=10)
    __serial.flush()
    print(f"{__serial.name} Opened\n")


# -----------------------------
# MCU'DAN REQUEST BEKLEME
# -----------------------------
def SERIAL_IMG_PollForRequest():
    global requestType, height, width, format, imgSize

    while True:
        if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
            print("Exit program!")
            exit(0)

        # Başlangıç byte'ını oku
        b1 = __serial.read(1)
        if len(b1) > 0 and np.frombuffer(b1, dtype=np.uint8)[0] == 83: # 'S'
            b2 = __serial.read(1)
            if len(b2) > 0 and np.frombuffer(b2, dtype=np.uint8)[0] == 84: # 'T'

                # NumPy uyarısını düzeltmek için [0] ekledik (Dizi -> Skaler)
                requestType = int(np.frombuffer(__serial.read(1), dtype=np.uint8)[0])
                height      = int(np.frombuffer(__serial.read(2), dtype=np.uint16)[0])
                width       = int(np.frombuffer(__serial.read(2), dtype=np.uint16)[0])
                format      = int(np.frombuffer(__serial.read(1), dtype=np.uint8)[0])

                imgSize = height * width * format # Grayscale için format=1

                print("======= REQUEST DETECTED =======")
                print("Request Type :", rqTypeText.get(requestType, "Unknown"))
                print(f"Size         : {width}x{height}")
                print("================================\n")

                return [requestType, height, width, format]

# -----------------------------
# MCU → PC IMAGE (READ)
# -----------------------------
def SERIAL_IMG_Read():
    # Tamponu temizlemeden önce veriyi tam okumaya çalışalım
    # serial.read() bazen timeout yüzünden eksik dönebilir, bunu garantiye alalım:
    data = b''
    remaining = imgSize
    
    start_time = time.time()
    while remaining > 0:
        chunk = __serial.read(remaining)
        if len(chunk) > 0:
            data += chunk
            remaining -= len(chunk)
        
        # 10 saniye içinde veri gelmezse döngüyü kır (Sonsuz döngüden kaçış)
        if (time.time() - start_time) > 10:
            print("Zaman aşımı! Veri eksik geldi.")
            break

    if len(data) != imgSize:
        print(f"HATA: Beklenen {imgSize} byte, alınan {len(data)} byte.")
        return None

    img = np.frombuffer(data, dtype=np.uint8)
    
    # Şekillendirme hatası almamak için boyut kontrolü
    try:
        img = np.reshape(img, (height, width, format))
    except ValueError as e:
        print(f"Reshape Hatası: {e}")
        return None

    # Format dönüştürme (Görüntüleme için)
    display_img = img
    if format == IMAGE_FORMAT_GRAYSCALE:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    timestamp = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
    filename = f"received_{timestamp}.png"
    cv2.imwrite(filename, display_img)
    print(f"[+] Image saved as: {filename}")

    cv2.imshow("Received", display_img)
    
    # KRİTİK NOKTA: Bekleme süresini kısalttık. 
    # 1ms bekler, pencereyi günceller ve hemen yeni veri dinlemeye döner.
    cv2.waitKey(1) 
    
    # destroyAllWindows() kaldırdık çünkü döngüde sürekli pencere aç-kapa yavaşlatır.

    return img

# -----------------------------
# PC → MCU IMAGE (WRITE)
# -----------------------------
def SERIAL_IMG_Write(path):
    img = cv2.imread(path)

    # MCU'nun istediği boyuta RESIZE
    img = cv2.resize(img, (width, height))

    # Format dönüştürme
    if format == IMAGE_FORMAT_GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif format == IMAGE_FORMAT_RGB565:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGR565)

    __serial.write(img.tobytes())
    print("[+] Image sent to MCU.")


# ======================================================
# ANA PROGRAM
# ======================================================
if __name__ == "__main__":

    COM_PORT = "COM5"        # ← Kerem, senin F446RE için COM8 doğru
    TEST_IMAGE = "mandrill.png"

    print(f"Seri port {COM_PORT} başlatılıyor...\n")
    SERIAL_Init(COM_PORT)
    print("Port başlatıldı.")

    while True:
        print("\nSTM32'den istek bekleniyor (PollForRequest)...")
        rqType, height, width, format = SERIAL_IMG_PollForRequest()

        # MCU → PC (F446RE görüntü yolluyor)
        if rqType == MCU_WRITES:
            print("MCU görüntü gönderiyor, alınıyor...")
            SERIAL_IMG_Read()

        # PC → MCU (F446RE görüntü istiyor)
        elif rqType == MCU_READS:
            print("MCU görüntü istiyor, gönderiliyor...")
            SERIAL_IMG_Write(TEST_IMAGE)
