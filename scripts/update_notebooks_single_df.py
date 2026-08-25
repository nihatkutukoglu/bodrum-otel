"""01 ve 02 notebooklarını tek ana DataFrame düzenine geçirir."""

import re
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def replace_variable(source: str) -> str:
    return re.sub(r"\bhotels\b", "df", source)


def update_collection() -> None:
    path = PROJECT_ROOT / "notebooks" / "01_data_collection.ipynb"
    nb = nbformat.read(path, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = replace_variable(cell.source)

    nb.cells[4].source = nb.cells[4].source.replace(
        '    "bodrum_hotels_area_summary_2026-08-24.csv": "Destination/area-level summary",\n',
        "",
    )
    nb.cells[8].source = """MASTER_PATH = file_paths["bodrum_hotels_master_2026-08-24.csv"]
df = pd.read_csv(MASTER_PATH, dtype={"phone": "string"})
df["area_hotel_count"] = df.groupby("area")["hotel_id"].transform("size")

dataset_dimensions = pd.DataFrame({"metric": ["rows", "columns"], "value": df.shape})
display(dataset_dimensions)
display(pd.DataFrame({"column": df.columns}))
display(df.head())
display(df.tail())"""
    nb.cells[9].source = """### 5. Tek DataFrame içinde destinasyon kapsamı

Ayrı bölge özet dosyası yüklenmez. Her otelin bulunduğu bölgedeki tesis sayısı, ana `df` üzerinden `area_hotel_count` kolonuna hesaplanır. Böylece analiz boyunca tek çalışma tablosu kullanılır."""
    nb.cells[10].source = """display(
    df[["area", "area_hotel_count"]]
    .drop_duplicates()
    .sort_values("area")
    .reset_index(drop=True)
)"""
    nb.cells[16].source = """### 8. Data Collection özeti

- Ana CSV başarıyla yüklendi: **192 otel kaydı ve 19 kaynak kolonu**.
- `area_hotel_count` ana `df` içinden türetildi; çalışma tablosu **20 kolon** içeriyor.
- Ana veride **14 destinasyon/bölge** bulunuyor.
- Ayrı bölge özet dosyası analiz girdisi olarak kullanılmıyor.
- README dokümantasyonu mevcut; Excel kopyası yardımcı format olarak bulunuyor.
- Bu notebook ham veriyi değiştirmedi, eksikleri doldurmadı ve kayıt silmedi.
- Sonraki adım `02_data_audit.ipynb` ile eksiklik, benzersizlik, geçerlilik, tutarlılık ve kaynak kapsamını denetlemektir."""

    nbformat.write(nb, path)


def update_audit() -> None:
    path = PROJECT_ROOT / "notebooks" / "02_data_audit.ipynb"
    nb = nbformat.read(path, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = replace_variable(cell.source)

    setup = nb.cells[2].source
    setup = setup.replace('AREA_SUMMARY_PATH = find_project_file("bodrum_hotels_area_summary_2026-08-24.csv")\n', "")
    setup = re.sub(r"^area_summary\s*=\s*pd\.read_csv\(AREA_SUMMARY_PATH\)\s*$", "", setup, flags=re.MULTILINE)
    if 'df["area_hotel_count"]' not in setup:
        setup = setup.replace(
            'df = pd.read_csv(MASTER_PATH, dtype={"phone": "string"})',
            'df = pd.read_csv(MASTER_PATH, dtype={"phone": "string"})\n'
            'df["area_hotel_count"] = df.groupby("area")["hotel_id"].transform("size")',
        )
    nb.cells[2].source = setup
    nb.cells[19].source = """### 10. Coğrafi / destinasyon kapsamı

Bölge bilgisi ayrı bir özet dosyasından alınmaz. `area_hotel_count` ana `df` içinde bulunur; ayrıntılı destinasyon raporu da aynı tablo üzerinden hesaplanır. `hotel_count < 5` olan bölgeler karşılaştırmalarda örneklem hassasiyeti nedeniyle işaretlenir."""

    nbformat.write(nb, path)


if __name__ == "__main__":
    update_collection()
    update_audit()
    print("01 ve 02 notebookları güncellendi.")
