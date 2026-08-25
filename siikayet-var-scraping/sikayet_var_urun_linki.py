import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Site listesini tanımla
site = ["https://www.sikayetvar.com/beko"]

# Link listesi
kategori_links = []

def get_driver():
    """Selenium driver yapılandırması"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Arka planda çalışması için aktif edilebilir
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"Driver başlatılamadı: {e}")
        # Alternatif olarak manuel driver yolu gerekebilir veya tarayıcı yüklü olmayabilir
        raise e

def kategori_linklerini_cek(driver, base_url):
    """Kategori linklerini çeker ve listeye ekler"""
    driver.get(base_url)
    time.sleep(3)  # Sayfanın yüklenmesi için bekleme (Gerekirse WebDriverWait kullanılabilir)
    
    # Kategori linklerini bul
    elements = driver.find_elements(By.CSS_SELECTOR, "a.company-collection-list-item-inner")
    
    for element in elements:
        href = element.get_attribute("href")
        title = element.get_attribute("title")
        if href:
            kategori_links.append(href)
            print(f"Kategori Bulundu: {title} -> {href}")

def urun_linklerini_cek_ve_kaydet(driver, kategori_links):
    """Kategori linklerini dolaşır, ürün linklerini bulur ve CSV'ye kaydetir"""
    with open('sikayet_var_urun_linki.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Kategori', 'Urun Adi', 'Urun Linki'])
        
        for kat_link in kategori_links:
            try:
                # Pencere kapandıysa döngüyü tamamen bitir
                try:
                    _ = driver.window_handles
                except:
                    print("Tarayıcı penceresi kapalı, işlem sonlandırılıyor.")
                    return

                print(f"İşleniyor: {kat_link}")
                driver.get(kat_link)
                time.sleep(3)
                
                # "Daha fazla göster" butonuna tıkla
                while True:
                    try:
                        # Butonun varlığını ve tıklanabilirliğini kontrol et
                        load_more_button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-ga-element='Company_Models_Load_More']"))
                        )
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_button)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", load_more_button)
                        print("  'Daha fazla göster' butonuna basıldı...")
                        time.sleep(2)
                    except Exception:
                        # Buton kalmadıysa veya hata oluştuysa (pencere kapanması hariç) bu sayfa için aramayı bitir
                        break
                
                # Sayfa hala açıksa ürünleri çek
                urun_elements = driver.find_elements(By.CSS_SELECTOR, "a.company-models__container")
                
                for urun in urun_elements:
                    try:
                        urun_href = urun.get_attribute("href")
                        urun_title = urun.get_attribute("title")
                        
                        if urun_href:
                            writer.writerow([kat_link, urun_title, urun_href])
                            file.flush() # Her üründe dosyayı güncelle (veri kaybını önlemek için)
                            print(f"  Ürün Kaydedildi: {urun_title}")
                    except:
                        continue
                        
            except Exception as e:
                # Pencere kapandıysa (NoSuchWindowException vb.)
                if "window" in str(e).lower():
                    print("Tarayıcı penceresi kapandı. Kayıtlar tamamlanıyor...")
                    return
                print(f"Kategori hatası ({kat_link}): {e}")
                continue

# Örnek kullanım (kodun sonuna eklenebilir)
if __name__ == "__main__":
    driver = get_driver()
    try:
        for s in site:
            kategori_linklerini_cek(driver, s)
        
        print(f"\nToplam {len(kategori_links)} kategori linki çekildi.")
        
        if kategori_links:
            print("\nÜrün linkleri toplanıyor...")
            urun_linklerini_cek_ve_kaydet(driver, kategori_links)
            print("\nTüm ürün linkleri 'sikayet_var_urun_linki.csv' dosyasına kaydedildi.")
            
    finally:
        driver.quit()

