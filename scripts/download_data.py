"""Download the ONISR road-accident open data (2019-2025) into data/raw/.

Source: data.gouv.fr, "Bases de donnees annuelles des accidents corporels de la
circulation routiere", Licence Ouverte 2.0.
File names are normalised to <table>-<year>.csv (caract, lieux, vehicules, usagers).
Each file's size is checked against the size verified on 2026-11-20, so a broken
or partial download is detected instead of silently used.

Usage (from the repo root):  python scripts/download_data.py
"""
from pathlib import Path
import urllib.request

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# (local file name, URL, expected size in bytes)
FILES = [
    ("caract-2019.csv", "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20201105-104400/caracteristiques-2019.csv", 6998722),
    ("caract-2020.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/20211110-111202/caracteristiques-2020.csv", 5655584),
    ("caract-2021.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2020/20221024-113743/carcteristiques-2021.csv", 7045926),
    ("caract-2022.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/20231005-093927/carcteristiques-2022.csv", 6884406),
    ("caract-2023.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241028-103125/caract-2023.csv", 6591826),
    ("caract-2024.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/20251021-115900/caract-2024.csv", 6538156),
    ("caract-2025.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2025/20261030-141517/caract-2025.csv", 6622493),
    ("lieux-2019.csv", "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20201105-104338/lieux-2019.csv", 5390071),
    ("lieux-2020.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/20211110-111603/lieux-2020.csv", 4521088),
    ("lieux-2021.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2020/20221024-113901/lieux-2021.csv", 5771984),
    ("lieux-2022.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/20231005-094112/lieux-2022.csv", 5617313),
    ("lieux-2023.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153219/lieux-2023.csv", 7176266),
    ("lieux-2024.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/20251021-115812/lieux-2024.csv", 7070795),
    ("lieux-2025.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2025/20261030-141502/lieux-2025.csv", 7260013),
    ("usagers-2019.csv", "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20201105-104232/usagers-2019.csv", 12454879),
    ("usagers-2020.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/20211110-111817/usagers-2020.csv", 10054646),
    ("usagers-2021.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/20231009-140337/usagers-2021.csv", 13028412),
    ("usagers-2022.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/20231005-094229/usagers-2022.csv", 13013255),
    ("usagers-2023.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153328/usagers-2023.csv", 13938227),
    ("usagers-2024.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/20251021-115506/usagers-2024.csv", 13843733),
    ("usagers-2025.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2025/20261030-141441/usagers-2025.csv", 13967977),
    ("vehicules-2019.csv", "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20201105-104310/vehicules-2019.csv", 6921630),
    ("vehicules-2020.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2019/20211110-111722/vehicules-2020.csv", 5492931),
    ("vehicules-2021.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2020/20221024-113925/vehicules-2021.csv", 6202316),
    ("vehicules-2022.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/20231005-094147/vehicules-2022.csv", 6021965),
    ("vehicules-2023.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153253/vehicules-2023.csv", 6431828),
    ("vehicules-2024.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024/20251107-100240/vehicules-2024.csv", 6369908),
    ("vehicules-2025.csv", "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2025/20261030-141410/vehicules-2025.csv", 6450936),
]


def download(name: str, url: str, size: int) -> None:
    target = RAW_DIR / name
    if target.exists() and target.stat().st_size == size:
        print(f"ok (already there)  {name}")
        return
    print(f"downloading         {name}")
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink()
        raise RuntimeError(f"{name}: expected {size} bytes, got {got}. Try again.")
    tmp.replace(target)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url, size in FILES:
        download(name, url, size)
    print(f"Done: {len(FILES)} files in {RAW_DIR}")


if __name__ == "__main__":
    main()
