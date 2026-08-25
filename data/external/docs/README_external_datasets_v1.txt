BODRUM HOTEL & DESTINATION INTELLIGENCE - EXTERNAL DATASETS V1
Generated: 2026-08-25

FILES
1) hotel_attributes_official_bodrum.csv
   - 168 rows transcribed from Muğla İl Kültür ve Turizm Müdürlüğü "İşletme Belgeli Tesisler" source.
   - Includes official facility type, extracted star rating where explicitly present, room count, bed count, address area and phone.
   - IMPORTANT: This is NOT yet merged to the project's 192 Google/Places hotels.
   - Use name + address + phone fuzzy/manual audit before assigning project hotel_id.
   - Never infer a star rating where the official type does not explicitly encode one.

2) destination_intelligence_v1.csv
   - 14 project areas.
   - Combines current repo EDA metrics with confidently mapped official accommodation capacity plus official market/marina context.
   - V1 intentionally does NOT invent restaurant/beach/nightlife POI counts. Those can be added later from OpenStreetMap/Overpass.

3) tourism_demand_monthly_mugla_2025.csv
   - 12 monthly rows for Muğla province, 2025.
   - Domestic/foreign/total arrivals, overnights, occupancy.
   - Average stay and foreign share are explicitly marked as derived.
   - IMPORTANT: Do not label this monthly series as Bodrum; it is Muğla province-level.

4) tourism_demand_annual_mugla_2009_2025.csv
   - Historical Muğla demand series for trend/seasonality context.

5) tourism_demand_bodrum_annual_2025.csv
   - Official Bodrum district-level 2025 annual arrivals and overnights.
   - Monthly Bodrum district values were not available in the surfaced source.

6) milas_bodrum_airport_monthly_2025.csv
   - DHMİ cumulative passenger statistics with monthly values derived by differencing.
   - Domestic, international and total passengers.
   - DHMİ notes that revisions to earlier months can appear in later snapshots.

MAIN SOURCES
- Official accommodation facilities: https://mugla.ktb.gov.tr/Eklenti/102365%2Cisletme-belgeli-tesisler--agustospdf.pdf?0=
- Muğla tourism data compilation from KTB official statistics: https://turizmveri.com/iller/mugla
- KTB metadata: https://yigm.ktb.gov.tr/TR-201124/metaveri.html
- Official Bodrum markets/marina context: https://mugla.ktb.gov.tr/TR-296343/alisveris.html
- Current project repo: https://github.com/nihatkutukoglu/bodrum-otel
- DHMİ passenger PDFs: URLs are stored row-by-row in the airport dataset.

RECOMMENDED NEXT STEP
- Create 06_external_data_collection_audit.ipynb.
- Audit official facility names and match to the 192-hotel master with normalized name + area/address + phone.
- Save only high-confidence matches automatically; send ambiguous matches to a manual review table.
- Then build hotels_enriched.csv without overwriting existing processed files.
