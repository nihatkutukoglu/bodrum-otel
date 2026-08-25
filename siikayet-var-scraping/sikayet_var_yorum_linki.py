import pandas as pd
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import re
import os 
import base64
import math
import undetected_chromedriver as uc

# Kodun çalıştığı klasörün yolunu otomatik bul
calisma_dizini = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(calisma_dizini, 'sikayet_var_urun_linki.csv')

# CSV dosyasını oku
df = pd.read_csv(csv_file)

# Ürün linklerini ve bilgilerini al (Listeye çevirmek çok önemli!)
urun_linkleri = df['Urun Linki'].unique().tolist()

print(f"Toplam {len(urun_linkleri)} adet benzersiz ürün bulundu.\n")

# 10 eşit parçaya böl
# urun_linkleri artık bir liste olduğu için len() > 0 kontrolü daha güvenlidir
chunk_size = math.ceil(len(urun_linkleri) / 10) if len(urun_linkleri) > 0 else 1
link_parcalari = [urun_linkleri[i:i + chunk_size] for i in range(0, len(urun_linkleri), chunk_size)]

# Boş liste kalmaması için 10'a tamamla (Gizli boşluk karakterleri düzeltildi)
while len(link_parcalari) < 10:
    link_parcalari.append([])

# Değişkenlere ata
urun_linkleri1 = link_parcalari[0]
urun_linkleri2 = link_parcalari[1]
urun_linkleri3 = link_parcalari[2]
urun_linkleri4 = link_parcalari[3]
urun_linkleri5 = link_parcalari[4]
urun_linkleri6 = link_parcalari[5]
urun_linkleri7 = link_parcalari[6]
urun_linkleri8 = link_parcalari[7]
urun_linkleri9 = link_parcalari[8]
urun_linkleri10 = link_parcalari[9]

# İlk 4 parçayı birleştir (Artık elemanlar liste olduğu için sorunsuz çalışır)
# urun_linkleri1_4 = sum(link_parcalari[0:4], [])

# Yorum satırındaki diğer birleştirmeleri de ileride açarsanız diye örnek:
urun_linkleri4_10 = sum(link_parcalari[4:10], [])

# Sonuçları saklamak için liste
sonuclar = []

# CSV dosyası için header tanımla
csv_file_out = 'sikayet_var_yorum_linki.csv'
is_first_write = True


def start_driver():
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # Buraya version_main=147 parametresini ekliyoruz!
    driver = uc.Chrome(options=options, version_main=147)
    return driver

driver = start_driver()

try:
    for idx, link in enumerate(tqdm(urun_linkleri4_10, desc="İlerleme"), 1):
        try:
            print(f"[{idx}/{len(urun_linkleri4_10)}] Ziyaret ediliyor: {link}")
            
            # CSV'den ilgili ürün bilgisini al
            urun_bilgisi = df[df['Urun Linki'] == link].iloc[0]
            urun_adi = urun_bilgisi['Urun Adi']
            
            # Linke git
            driver.get(link)
        
            # Sayfanın yüklenmesi için bekle
            time.sleep(3)


            # --- ÇEREZ (COOKIE) DEDEKTÖRÜ ---
            try:
                # Çerez butonunu ID'si ile arıyoruz (Maksimum 3 saniye bekler)
                cerez_butonu = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))
                )
                # Eğer buton ekrandaysa tıkla
                cerez_butonu.click()
                print("   🍪 Çerez penceresi kapatıldı.")
                time.sleep(1) # Pencerenin ekrandan kaybolması için 1 saniye bekle
            except:
                # Buton sayfada yoksa (daha önce kabul edildiyse veya o üründe çıkmadıysa) sessizce geç
                pass
            # --------------------------------
       

            # Ekstra veriler için değişkenleri başlat
            # Ekstra veriler için değişkenleri başlat
            urun_adi_cekilen = ""
            urun_puan_yuzdelik = ""
            anahtar_etiket = ""
            yorum_linkleri_str = ""
            sikayet_sayisi = 0

            # 1. urun_adi çek
            try:
                elem_h1 = driver.find_element(By.CSS_SELECTOR, "h1.model-name")
                urun_adi_cekilen = elem_h1.text.strip()
            except:
                pass

            # 2. urun_puan_yuzdelik çek
            try:
                # <div class="rate-num"><span>40</span><span>100</span></div> yapısından ilk span'ı (örn: 40) çeker
                elem_puan = driver.find_element(By.CSS_SELECTOR, "div.rate-num span:first-child")
                urun_puan_yuzdelik = elem_puan.text.strip()
            except:
                pass

            # 3. anahtar_etiket çek (yetkili servis vs.)
            try:
                elem_etiketler = driver.find_elements(By.CSS_SELECTOR, "a.search-item")
                anahtar_etiket = ", ".join([el.text.strip() for el in elem_etiketler if el.text.strip()])
            except:
                pass


            # 4. Yorumları çek - Sayfalama (URL Manipülasyonu), Direkt Dibe İnme ve Şifre Kırıcı
            try:
                yorum_linkleri = []
                sayfa_no = 1
                
                while True:
                    print(f"   🔍 Sayfa {sayfa_no} taranıyor...")
                    
                    # --- REKLAM DEDEKTÖRÜ (Google Vignette) ---
                    if "#google_vignette" in driver.current_url:
                        print("      > ⚠️ Google reklamı tespit edildi, reklam atlanıyor...")
                        temiz_url = f"{link}?page={sayfa_no}" if sayfa_no > 1 else link
                        driver.get(temiz_url)
                        time.sleep(4) 
                    # --------------------------------------------------------
                    
                    # --- DİREKT EN DİBE İNME ALGORİTMASI ---
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    
                    # Sayfa uzadığı sürece direkt en dibe inmeye devam et
                    for _ in range(5): 
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2.5) 
                        
                        yeni_yukseklik = driver.execute_script("return document.body.scrollHeight")
                        if yeni_yukseklik == last_height:
                            break 
                        last_height = yeni_yukseklik
                        
                    # API'yi son bir kez uyandırmak için hafif yukarı çıkıp tekrar dibe vur
                    driver.execute_script("window.scrollBy(0, -600);")
                    time.sleep(1)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    # --------------------------------------------------
                    
                    # --- YENİ VE KUSURSUZ FİLTRE (Şifre Kırıcı Entegreli) ---
                    # Artık sadece 'a' değil, sınıfında complaint-layer olan her şeyi (div veya a) alıyoruz
                    yorum_elemanlari = driver.find_elements(By.CSS_SELECTOR, ".complaint-layer")
                    
                    yeni_bulunan = 0
                    for elem in yorum_elemanlari:
                        try:
                            href = None
                            tag_name = elem.tag_name
                            
                            # Durum 1: Normal, gizlenmemiş link
                            if tag_name == "a":
                                href = elem.get_attribute("href")
                                
                            # Durum 2: Base64 ile şifrelenmiş div
                            elif tag_name == "div":
                                data_url = elem.get_attribute("data-url")
                                if data_url:
                                    # Base64 şifresini çözüyoruz
                                    decoded_bytes = base64.b64decode(data_url)
                                    decoded_str = decoded_bytes.decode("utf-8")
                                    # Başına site adresini ekleyerek tam link yapıyoruz
                                    href = f"https://www.sikayetvar.com{decoded_str}"
                            
                            # Linki elde ettik, filtrelerden geçiriyoruz
                            if href and href.startswith("https://www.sikayetvar.com/beko/"):
                                ana_link_temiz = href.split("?")[0] 
                                
                                if ana_link_temiz != link and ana_link_temiz not in yorum_linkleri:
                                    yorum_linkleri.append(ana_link_temiz)
                                    yeni_bulunan += 1
                        except:
                            continue
                                
                    print(f"      > Bu sayfadan {yeni_bulunan} net şikayet alındı.")
                    
                    # Eğer bu sayfada YENİ hiçbir şikayet bulamadıysak, sayfalar bitmiş demektir
                    if yeni_bulunan == 0:
                        print("      > Yeni şikayet bulunamadı, sayfalar bitti.")
                        break 
                    
                    sayfa_no += 1
                    next_url = f"{link}?page={sayfa_no}"
                    
                    print(f"      > Sonraki sayfaya geçiliyor: {next_url}") 
                    driver.get(next_url)
                    time.sleep(4.5) 
                    
                yorum_linkleri_str = ", ".join(yorum_linkleri)
                
                if yorum_linkleri_str:
                    gecerli_sayfa = sayfa_no - 1 if sayfa_no > 1 else 1 
                    print(f"   ✅ Toplam {gecerli_sayfa} sayfa gezildi, {len(yorum_linkleri)} adet şikayet linki çekildi.")
                else:
                    print("   ⚠️ Link bulunamadı.")
                    
                if sayfa_no > 1:
                    driver.get(link)
                    time.sleep(3) 
                    
            except Exception as e:
                print(f"   ✗ Yorum çekme hatası: {e}")
                yorum_linkleri_str = ", ".join(yorum_linkleri) if 'yorum_linkleri' in locals() else ""


            # 5. Şikayet sayısını içeren elemanı bul
            try:
                complaint_element = driver.find_element(By.CLASS_NAME, "complaint-count")
                complaint_text = complaint_element.text
                match = re.search(r'(\d+)', complaint_text)
                if match:
                    sikayet_sayisi = int(match.group(1))
                    print(f"   ✓ Sayfadan: {urun_adi_cekilen} | Puan: {urun_puan_yuzdelik} | Şikayet: {sikayet_sayisi}")
                else:
                    print(f"   ✗ Şikayet sayısı bulunamadı: {complaint_text}")
            except Exception as e:
                print(f"   ✗ Hata (şikayet sayısı): {str(e)}")
            
            # Sonucu kaydet
            # Eğer yorum linki varsa her linki ayrı satır yaz
            if yorum_linkleri:
                rows = []

                for yorum_linki in yorum_linkleri:
                    row_data = {
                        'Urun_Adi_CSV': urun_adi,
                        'Urun_Linki': link,
                        'Urun_Adi_Sayfadan': urun_adi_cekilen,
                        'Urun_Puan_Yuzdelik': urun_puan_yuzdelik,
                        'Anahtar_Etiketler': anahtar_etiket,
                        'Yorum_Linki': yorum_linki,
                        'Sikayet_Sayisi': sikayet_sayisi
                    }

                    rows.append(row_data)
                    sonuclar.append(row_data)

                sonuc_df_temp = pd.DataFrame(rows)

            else:
                # Hiç yorum linki bulunamazsa ürün yine kaydedilsin
                row_data = {
                    'Urun_Adi_CSV': urun_adi,
                    'Urun_Linki': link,
                    'Urun_Adi_Sayfadan': urun_adi_cekilen,
                    'Urun_Puan_Yuzdelik': urun_puan_yuzdelik,
                    'Anahtar_Etiketler': anahtar_etiket,
                    'Yorum_Linki': '',
                    'Sikayet_Sayisi': sikayet_sayisi
                }

                sonuc_df_temp = pd.DataFrame([row_data])
                sonuclar.append(row_data)

            # CSV'ye yaz
            sonuc_df_temp.to_csv(
                csv_file_out,
                mode='a',
                header=is_first_write,
                index=False,
                encoding='utf-8-sig'
            )

            is_first_write = False
            
        except Exception as e:
            err_msg = str(e)
            print(f"   ✗ Link ziyareti hatası: {err_msg}")
            
            # Eğer session düştüyse (tarayıcı kapandıysa) yeniden başlat
            if "invalid session id" in err_msg.lower() or "target window already closed" in err_msg.lower():
                print("   ➔ Tarayıcı oturumu koptu, yeniden başlatılıyor...")
                try:
                    driver.quit()
                except:
                    pass
                driver = start_driver()
                
            hata_row = {
                'Urun_Adi_CSV': '',
                'Urun_Linki': link,
                'Urun_Adi_Sayfadan': '',
                'Urun_Puan_Yuzdelik': '',
                'Anahtar_Etiketler': '',
                'Yorum_Linkleri': '',
                'Sikayet_Sayisi': None
            }
            sonuclar.append(hata_row)
            
            # Hata kaydını da CSV'ye kaydet
            sonuc_df_temp = pd.DataFrame([hata_row])
            sonuc_df_temp.to_csv(csv_file_out, mode='a', header=is_first_write, 
                                index=False, encoding='utf-8-sig')
            is_first_write = False
        
        # Rate limiting
        time.sleep(4)

finally:
    driver.quit()

# Sonuçları DataFrame'e dönüştür (özet istatistikler için)
sonuc_df = pd.DataFrame(sonuclar)

print(f"\n✅ Tüm veriler '{csv_file_out}' dosyasına kaydedildi.")
