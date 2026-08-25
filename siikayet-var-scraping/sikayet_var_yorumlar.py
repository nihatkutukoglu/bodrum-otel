import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import math
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, 'sikayet_var_yorum_linki.csv')
output_csv = os.path.join(script_dir, 'sikayet_var_yorumlar.csv')

# Load the input csv
try:
    df = pd.read_csv(input_csv)
    links = df['Yorum_Linki'].dropna().unique().tolist()

    
    print(f"Toplam {len(links)} adet benzersiz link bulundu.\n")
    
    # 5 eşit parçaya böl
    chunk_size = math.ceil(len(links) / 5) if len(links) > 0 else 1
    link_parcalari = [links[i:i + chunk_size] for i in range(0, len(links), chunk_size)]
    
    # Boş liste kalmaması için 5'a tamamla
    while len(link_parcalari) < 5:
        link_parcalari.append([])
        
    links1 = link_parcalari[0]
    links2 = link_parcalari[1]
    links3 = link_parcalari[2]
    links4 = link_parcalari[3]
    links5 = link_parcalari[4]

    
    links2_5 = sum(link_parcalari[1:5], [])

except Exception as e:
    print(f'Hata: {e}')
    exit()

# CSV başlıkları yeni sütunlar (Destek, Cevap ve Cevaba Cevaplar) ile güncellendi
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        basliklar = [
            'Yorum_Linki', 'Sikayet_Basligi', 'Sikayet', 'Sikayet_Tarihi', 
            'Goruntuleme_Sayisi', 'Destek_Sayisi', 'Kategori', 'Urun_Adi', 'Sirket_Cevabi', 
            'Gelisme_Tarih', 'Gelisme_Yorum', 'Cevap_Tarihi', 'Cevap_Mesaji',
            'Cevaba_Cevap_1_Tarih', 'Cevaba_Cevap_1_Mesaj',
            'Cevaba_Cevap_2_Tarih', 'Cevaba_Cevap_2_Mesaj',
            'Cevaba_Cevap_3_Tarih', 'Cevaba_Cevap_3_Mesaj',
            'Cevaba_Cevap_4_Tarih', 'Cevaba_Cevap_4_Mesaj',
            'Cevaba_Cevap_5_Tarih', 'Cevaba_Cevap_5_Mesaj'
        ]
        writer.writerow(basliklar)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# İlerleyen süreçlerde buradaki links1 ifadesini links2, links3 vs. olarak değiştirebilirsiniz
for link in tqdm(links2_5, desc="Şikayetler İşleniyor"):
    try:
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        desc_div = soup.find('div', class_='complaint-detail-description')
        if desc_div:
            # Şikayet metni
            paragraphs = desc_div.find_all('p')
            comment_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Şikayet Başlığı
            title_h1 = soup.find('h1', class_='complaint-detail-title')
            sikayet_basligi = title_h1.get_text(strip=True) if title_h1 else ""
            
            # Şikayet Tarihi
            date_div = soup.find('div', attrs={'class': lambda e: e and 'js-tooltip' in e and 'time' in e})
            sikayet_tarihi = date_div.get_text(strip=True) if date_div else ""
            
            # Görüntüleme Sayısı
            view_span = soup.find('span', attrs={'class': lambda e: e and 'js-view-count' in e})
            goruntuleme_sayisi = view_span.get_text(strip=True) if view_span else ""

            # --- YENİ EKLENEN: Destek Sayısı Çekme ---
            destek_sayisi = "0"
            destek_span = soup.find('span', class_='total-support')
            if destek_span and destek_span.find('span'):
                destek_sayisi = destek_span.find('span').get_text(strip=True)
            
            # Kategori ve Ürün Adı
            kategori = ""
            urun_adi = ""
            
            breadcrumb_ul = soup.find('ul', class_='breadcrumb')
            if breadcrumb_ul:
                crumb_links = breadcrumb_ul.find_all('a')
                crumb_texts = [a.get_text(strip=True) for a in crumb_links]
                
                if len(crumb_texts) >= 4:
                    kategori = crumb_texts[-2]
                    urun_adi = crumb_texts[-1]
                elif len(crumb_texts) == 3:
                    kategori = crumb_texts[-1]
                    urun_adi = ""
                    
            # Şirket Cevabı
            sirket_cevabi = 0
            profile_wraps = soup.find_all('div', class_='profile-name-wrap')
            for wrap in profile_wraps:
                company_div = wrap.find('div', class_='company-name')
                if company_div:
                    a_tag = company_div.find('a', href='/beko')
                    if a_tag:
                        sirket_cevabi = 1
                        break

            # Gelişme Kontrolü ve Detayları
            gelisme_div = soup.find('div', class_='progressed-card')
            gelisme_tarih = ""
            gelisme_yorum = ""

            if gelisme_div:
                # Gelişme Tarihi
                tarih_span = gelisme_div.find('span', class_='time-history')
                if tarih_span:
                    gelisme_tarih = tarih_span.get_text(strip=True)
                
                # Gelişme Yorumu
                yorum_div = gelisme_div.find('div', class_='progressed-item-text')
                if yorum_div:
                    yorum_p = yorum_div.find_all('p')
                    gelisme_yorum = ' '.join([p.get_text(strip=True) for p in yorum_p])

            # --- YENİ EKLENEN: Marka Cevabı ve Cevaba Cevapları Çekme ---
            cevap_tarihi = ""
            cevap_mesaji = ""
            cevaba_cevaplar_listesi = []

            # Sayfadaki tüm yorum/cevap kutularını buluyoruz
            reply_blocks = soup.find_all('div', class_='complaint-reply')
            for block in reply_blocks:
                # Tarihi çekiyoruz ve yanındaki parantezli gereksiz "(Şikayetten 1 saat sonra)" kısmını siliyoruz
                time_div = block.find('div', class_=lambda x: x and 'time' in x and 'js-tooltip' in x)
                b_time = time_div.get_text(strip=True) if time_div else ""
                if "(" in b_time:
                    b_time = b_time.split('(')[0].strip()

                # Mesajı çekiyoruz
                msg_p = block.find('p', class_='message')
                b_msg = msg_p.get_text(separator=" ", strip=True) if msg_p else ""

                # Bu kutu markanın (Beko) cevabı mı kontrol ediyoruz
                is_brand = False
                if block.get('data-ga-element') == 'Complaint_Answer_Brand' or block.find('a', href='/beko'):
                    is_brand = True

                # Eğer markanın yanıtıysa ve henüz cevap atanmamışsa, ana şirket cevabı olarak kaydet
                if is_brand and cevap_mesaji == "":
                    cevap_tarihi = b_time
                    cevap_mesaji = b_msg
                else:
                    # Diğer tüm durumlar (kullanıcının yanıtları vb.) cevaba cevaptır
                    if b_msg:
                        cevaba_cevaplar_listesi.extend([b_time, b_msg])

            
            
            # Ana veriler (13 sabit sütun)
            satir_verisi = [
                link, sikayet_basligi, comment_text, sikayet_tarihi, 
                goruntuleme_sayisi, destek_sayisi, kategori, urun_adi, sirket_cevabi,
                gelisme_tarih, gelisme_yorum, cevap_tarihi, cevap_mesaji
            ]
            
            # Cevaba cevaplar için başlıkta 5 cevaplık (10 sütun) yer açtık.
            beklenen_ek_sutun_sayisi = 10
            
            # Eğer beklenen sayıdan az cevap varsa, listeyi boş stringlerle doldur (Pandas bunları NaN/Null okuyacaktır)
            eksik_sutun_sayisi = beklenen_ek_sutun_sayisi - len(cevaba_cevaplar_listesi)
            if eksik_sutun_sayisi > 0:
                cevaba_cevaplar_listesi.extend([""] * eksik_sutun_sayisi)
            
            # Eğer beklenen sayıdan FAZLA cevap varsa (örneğin 6. cevap atılmışsa), listeyi kesip hata almayı önle
            cevaba_cevaplar_listesi = cevaba_cevaplar_listesi[:beklenen_ek_sutun_sayisi]

            # Sabit uzunluktaki listeyi ana veriye ekle (Toplam 23 sütun garanti edildi)
            satir_verisi.extend(cevaba_cevaplar_listesi)

            # CSV'ye doğrudan kaydet
            with open(output_csv, mode='a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(satir_verisi)
            
            print(f'Başarılı: {link}')
            
        else:
            print(f'Şikayet bulunamadı: {link}')
        
        time.sleep(1) # Be polite to the server
    except Exception as e:
        print(f'Hata oluştu ({link}): {e}')