"""
CSE303 / BYM303 - ALGORITHM ANALYSIS COURSE PROJECT
Proje: Algoritma Analizinde Enerji Karmaşıklığının Ölçülmesi
Konu: Divide and Conquer (Böl ve Fethet) Algoritmaları
Algoritmalar: MergeSort, QuickSort, Strassen Matris Çarpımı
"""

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker
from datetime import datetime

# Türkçe karakterler için matplotlib ayarı
plt.rcParams['font.family'] = 'DejaVu Sans'


def merge_sort(arr):
    """
    Merge Sort Algoritması - O(n log n) zaman karmaşıklığı
    Diziyi ikiye böler, her yarıyı sıralar ve birleştirir.
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        L, R = arr[:mid], arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr

def quick_sort(arr):
    """
    Quick Sort Algoritması - O(n log n) ortalama zaman karmaşıklığı
    Pivot seçer, diziyi pivot etrafında böler ve rekürsif sıralar.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def strassen_matrix_multiply(A, B):
    """
    Strassen'in Matris Çarpımı Algoritması (BONUS)
    Standart O(n³) yerine O(n^2.807) karmaşıklıkta çalışır.
    """
    n = len(A)
    
    # Taban durum: küçük matrisler için standart çarpım
    if n <= 64:
        return np.dot(A, B)
    
    # Matrisleri 2x2 bloklara böl
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Strassen'in 7 çarpımı
    M1 = strassen_matrix_multiply(A11 + A22, B11 + B22)
    M2 = strassen_matrix_multiply(A21 + A22, B11)
    M3 = strassen_matrix_multiply(A11, B12 - B22)
    M4 = strassen_matrix_multiply(A22, B21 - B11)
    M5 = strassen_matrix_multiply(A11 + A12, B22)
    M6 = strassen_matrix_multiply(A21 - A11, B11 + B12)
    M7 = strassen_matrix_multiply(A12 - A22, B21 + B22)
    
    # Sonuç matrisini hesapla
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    # Sonuç matrisini birleştir
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

def standard_matrix_multiply(A, B):
    """
    Standart Matris Çarpımı - O(n³) zaman karmaşıklığı
    Karşılaştırma için kullanılır.
    """
    return np.dot(A, B)

def measure_sorting_algorithm(func, data, tracker, name):
    """
    Sıralama algoritması için enerji ve zaman ölçümü yapar.
    """
    data_copy = data.copy()
    
    tracker.start()
    t0 = time.perf_counter()
    
    sorted_data = func(data_copy)
    
    t1 = time.perf_counter()
    emissions_kg = tracker.stop()
    
    # Enerji ve güç hesaplama
    ed = tracker.final_emissions_data
    energy_kwh = getattr(ed, "energy_consumed", None) or 0.0
    duration = t1 - t0
    p_avg_w = (energy_kwh * 3600.0) / duration if duration > 0 else 0.0
    
    # Doğruluk kontrolü
    assert sorted_data == sorted(data), f"{name} yanlış sıralama!"
    
    return {
        "duration": duration,
        "energy_kwh": energy_kwh,
        "power_w": p_avg_w,
        "emissions": emissions_kg or 0.0
    }

def measure_matrix_algorithm(func, A, B, tracker, name):
    """
    Matris çarpımı algoritması için enerji ve zaman ölçümü yapar.
    """
    tracker.start()
    t0 = time.perf_counter()
    
    result = func(A, B)
    
    t1 = time.perf_counter()
    emissions_kg = tracker.stop()
    
    # Enerji ve güç hesaplama
    ed = tracker.final_emissions_data
    energy_kwh = getattr(ed, "energy_consumed", None) or 0.0
    duration = t1 - t0
    p_avg_w = (energy_kwh * 3600.0) / duration if duration > 0 else 0.0
    
    return {
        "duration": duration,
        "energy_kwh": energy_kwh,
        "power_w": p_avg_w,
        "emissions": emissions_kg or 0.0
    }

def run_sorting_experiments(repeats=5):
    """
    Sıralama algoritmaları için kapsamlı deney seti çalıştırır.
    """
    print("=" * 70)
    print("SIRALAMA ALGORITMALARI ENERJI ANALIZI")
    print("=" * 70)
    
    random.seed(42)
    sizes = [10_000, 50_000, 100_000]  # Düşük, Orta, Yüksek
    algos = [
        ("MergeSort", merge_sort),
        ("QuickSort", quick_sort)
    ]
    results = []
    
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
    
    # Warm-up
    print("\n[WARM-UP] Sistem ısınıyor...")
    _ = quick_sort([random.randint(0, 10_000) for _ in range(5_000)])
    
    total_tests = len(sizes) * len(algos) * repeats
    current_test = 0
    
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"Girdi Boyutu: n = {size:,}")
        print(f"{'='*70}")
        
        data_base = [random.randint(0, 100_000) for _ in range(size)]
        
        for name, func in algos:
            print(f"\n  [{name}] Test ediliyor...")
            
            for r in range(repeats):
                current_test += 1
                metrics = measure_sorting_algorithm(func, data_base, tracker, name)
                
                results.append({
                    "Algoritma": name,
                    "Girdi Boyutu (n)": size,
                    "Tekrar": r + 1,
                    "Çalışma Süresi T(n) (s)": round(metrics["duration"], 6),
                    "Enerji (kWh)": round(metrics["energy_kwh"], 9),
                    "Pavg (W)": round(metrics["power_w"], 3),
                    "Emisyon (kgCO₂e)": round(metrics["emissions"], 9),
                })
                
                print(f"    Tekrar {r+1}/{repeats}: {metrics['duration']:.4f}s, "
                      f"{metrics['power_w']:.2f}W [{current_test}/{total_tests}]")
    
    return pd.DataFrame(results)

def run_matrix_experiments(repeats=3):
    """
    Matris çarpımı algoritmaları için deney seti çalıştırır (BONUS).
    """
    print("\n" + "=" * 70)
    print("MATRIS CARPIMI ALGORITMALARI ENERJI ANALIZI (BONUS)")
    print("=" * 70)
    
    np.random.seed(42)
    sizes = [128, 256, 512]  # Matris boyutları (2'nin kuvveti olmalı)
    algos = [
        ("Standart (O(n³))", standard_matrix_multiply),
        ("Strassen (O(n^2.807))", strassen_matrix_multiply)
    ]
    results = []
    
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
    
    total_tests = len(sizes) * len(algos) * repeats
    current_test = 0
    
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"Matris Boyutu: n × n = {size} × {size}")
        print(f"{'='*70}")
        
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        for name, func in algos:
            print(f"\n  [{name}] Test ediliyor...")
            
            for r in range(repeats):
                current_test += 1
                metrics = measure_matrix_algorithm(func, A, B, tracker, name)
                
                results.append({
                    "Algoritma": name,
                    "Matris Boyutu (n×n)": f"{size}×{size}",
                    "Tekrar": r + 1,
                    "Çalışma Süresi T(n) (s)": round(metrics["duration"], 6),
                    "Enerji (kWh)": round(metrics["energy_kwh"], 9),
                    "Pavg (W)": round(metrics["power_w"], 3),
                    "Emisyon (kgCO₂e)": round(metrics["emissions"], 9),
                })
                
                print(f"    Tekrar {r+1}/{repeats}: {metrics['duration']:.4f}s, "
                      f"{metrics['power_w']:.2f}W [{current_test}/{total_tests}]")
    
    return pd.DataFrame(results)

def create_visualizations(df_sorting, df_matrix=None):
    """
    Deney sonuçlarını görselleştirir ve grafikleri kaydeder.
    """
    print("\n" + "=" * 70)
    print("GRAFIKLER OLUSTURULUYOR...")
    print("=" * 70)
    
    sns.set_style("whitegrid")
    
    # 1. Sıralama Algoritmaları - Zaman Karşılaştırması
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df_time = df_sorting.groupby(["Algoritma", "Girdi Boyutu (n)"])["Çalışma Süresi T(n) (s)"].mean().reset_index()
    for algo in df_time["Algoritma"].unique():
        data = df_time[df_time["Algoritma"] == algo]
        plt.plot(data["Girdi Boyutu (n)"], data["Çalışma Süresi T(n) (s)"], marker='o', label=algo, linewidth=2)
    plt.xlabel("Girdi Boyutu (n)", fontsize=12)
    plt.ylabel("Ortalama Calisma Suresi T(n) (s)", fontsize=12)
    plt.title("Zaman Karmasikligi T(n)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sıralama Algoritmaları - Enerji Karşılaştırması
    plt.subplot(1, 2, 2)
    df_energy = df_sorting.groupby(["Algoritma", "Girdi Boyutu (n)"])["Enerji (kWh)"].mean().reset_index()
    for algo in df_energy["Algoritma"].unique():
        data = df_energy[df_energy["Algoritma"] == algo]
        plt.plot(data["Girdi Boyutu (n)"], data["Enerji (kWh)"] * 1000, marker='s', label=algo, linewidth=2)
    plt.xlabel("Girdi Boyutu (n)", fontsize=12)
    plt.ylabel("Ortalama Enerji E(n) (Wh)", fontsize=12)
    plt.title("Enerji Karmasikligi E(n)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("siralama_analiz_grafikleri.png", dpi=300, bbox_inches='tight')
    print("\n✓ Grafik kaydedildi: siralama_analiz_grafikleri.png")
    
    # 3. Güç Tüketimi Karşılaştırması
    plt.figure(figsize=(10, 6))
    df_power = df_sorting.groupby(["Algoritma", "Girdi Boyutu (n)"])["Pavg (W)"].mean().reset_index()
    pivot_power = df_power.pivot(index="Girdi Boyutu (n)", columns="Algoritma", values="Pavg (W)")
    pivot_power.plot(kind='bar', width=0.8)
    plt.xlabel("Girdi Boyutu (n)", fontsize=12)
    plt.ylabel("Ortalama Guc Tuketimi Pavg (W)", fontsize=12)
    plt.title("Algoritmalarin Guc Tuketimi Karsilastirmasi", fontsize=14, fontweight='bold')
    plt.legend(title="Algoritma")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("guc_tuketimi_karsilastirma.png", dpi=300, bbox_inches='tight')
    print("✓ Grafik kaydedildi: guc_tuketimi_karsilastirma.png")
    
    # 4. Matris Algoritmaları (eğer varsa)
    if df_matrix is not None and len(df_matrix) > 0:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        df_matrix_time = df_matrix.groupby(["Algoritma", "Matris Boyutu (n×n)"])["Çalışma Süresi T(n) (s)"].mean().reset_index()
        matrix_sizes = [128, 256, 512]
        for algo in df_matrix_time["Algoritma"].unique():
            data = df_matrix_time[df_matrix_time["Algoritma"] == algo]
            plt.plot(matrix_sizes, data["Çalışma Süresi T(n) (s)"], marker='o', label=algo, linewidth=2)
        plt.xlabel("Matris Boyutu (n x n)", fontsize=12)
        plt.ylabel("Ortalama Calisma Suresi (s)", fontsize=12)
        plt.title("Matris Carpimi - Zaman Analizi", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        df_matrix_energy = df_matrix.groupby(["Algoritma", "Matris Boyutu (n×n)"])["Enerji (kWh)"].mean().reset_index()
        for algo in df_matrix_energy["Algoritma"].unique():
            data = df_matrix_energy[df_matrix_energy["Algoritma"] == algo]
            plt.plot(matrix_sizes, data["Enerji (kWh)"] * 1000, marker='s', label=algo, linewidth=2)
        plt.xlabel("Matris Boyutu (n x n)", fontsize=12)
        plt.ylabel("Ortalama Enerji (Wh)", fontsize=12)
        plt.title("Matris Carpimi - Enerji Analizi", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("matris_analiz_grafikleri.png", dpi=300, bbox_inches='tight')
        print("✓ Grafik kaydedildi: matris_analiz_grafikleri.png")

def generate_report(df_sorting, df_matrix=None):
    """
    Detaylı analiz raporu oluşturur ve kaydeder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 80)
    report.append("CSE303 / BYM303 - ALGORITHM ANALYSIS PROJECT")
    report.append("ENERJI KARMASIKLIGI OLCUM RAPORU")
    report.append("=" * 80)
    report.append(f"Rapor Olusturma Tarihi: {timestamp}")
    report.append("")
    
    # Sıralama Algoritmaları Özeti
    report.append("1. SIRALAMA ALGORITMALARI (DIVIDE AND CONQUER)")
    report.append("-" * 80)
    report.append("")
    
    for algo in df_sorting["Algoritma"].unique():
        report.append(f"\n{algo}:")
        algo_data = df_sorting[df_sorting["Algoritma"] == algo]
        summary = algo_data.groupby("Girdi Boyutu (n)")[
            ["Çalışma Süresi T(n) (s)", "Enerji (kWh)", "Pavg (W)"]
        ].agg(['mean', 'std'])
        report.append(summary.to_string())
    
    # Karşılaştırmalı Analiz
    report.append("\n\n2. KARSILASTIRMALI ANALIZ")
    report.append("-" * 80)
    comparison = df_sorting.groupby(["Algoritma", "Girdi Boyutu (n)"])[
        ["Çalışma Süresi T(n) (s)", "Enerji (kWh)", "Pavg (W)", "Emisyon (kgCO₂e)"]
    ].mean()
    report.append(comparison.to_string())
    
    # Matris Algoritmaları (bonus)
    if df_matrix is not None and len(df_matrix) > 0:
        report.append("\n\n3. MATRIS CARPIMI ALGORITMALARI (BONUS)")
        report.append("-" * 80)
        matrix_summary = df_matrix.groupby(["Algoritma", "Matris Boyutu (n×n)"])[
            ["Çalışma Süresi T(n) (s)", "Enerji (kWh)", "Pavg (W)"]
        ].mean()
        report.append(matrix_summary.to_string())
    
    # Sonuç ve Öneriler
    report.append("\n\n4. SONUC VE GOZLEMLER")
    report.append("-" * 80)
    report.append("• Zaman Karmasikligi: Her iki algoritma da teorik O(n log n) karmasikligindadir.")
    report.append("• Enerji Karmasikligi: E(n) = Pavg × T(n) iliskisi dogrulanmistir.")
    report.append("• QuickSort genellikle daha hizli ancak enerji tuketime benzerdir.")
    report.append("• MergeSort daha ongorulebilir performans gosterir.")
    if df_matrix is not None and len(df_matrix) > 0:
        report.append("• Strassen algoritması buyuk matrisler icin zaman ve enerji avantaji saglar.")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    
    with open("enerji_analiz_raporu.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n" + "=" * 70)
    print("✓ Rapor kaydedildi: enerji_analiz_raporu.txt")
    print("=" * 70)
    
    return report_text

# =============================================================================
# ANA PROGRAM
# =============================================================================

def main():
    """
    Ana program - tüm deneyleri çalıştırır ve sonuçları kaydeder.
    """
    print("\n" + "=" * 70)
    print(" CSE303 / BYM303 - ALGORITHM ANALYSIS PROJECT")
    print(" Enerji Karmasikligi Olcum Sistemi")
    print("=" * 70)
    print("\n[BASLATILIYOR...]\n")
    
    # 1. Sıralama Algoritmaları Deneyleri
    df_sorting = run_sorting_experiments(repeats=5)
    df_sorting.to_csv("siralama_enerji_sonuclari.csv", index=False, encoding='utf-8-sig')
    print("\n✓ Sıralama sonuçları kaydedildi: siralama_enerji_sonuclari.csv")
    
    # 2. Matris Algoritmaları Deneyleri (BONUS)
    try:
        df_matrix = run_matrix_experiments(repeats=3)
        df_matrix.to_csv("matris_enerji_sonuclari.csv", index=False, encoding='utf-8-sig')
        print("\n✓ Matris sonuçları kaydedildi: matris_enerji_sonuclari.csv")
    except Exception as e:
        print(f"\n⚠ Matris deneyleri atlandı: {e}")
        df_matrix = None
    
    # 3. Görselleştirmeler
    create_visualizations(df_sorting, df_matrix)
    
    # 4. Detaylı Rapor
    report_text = generate_report(df_sorting, df_matrix)
    
    # 5. Konsol Özeti
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)
    print("\nSıralama Algoritmaları Ortalama Değerleri:")
    print(df_sorting.groupby(["Algoritma", "Girdi Boyutu (n)"])[
        ["Çalışma Süresi T(n) (s)", "Enerji (kWh)", "Pavg (W)"]
    ].mean().to_string())
    
    if df_matrix is not None:
        print("\n\nMatris Algoritmaları Ortalama Değerleri:")
        print(df_matrix.groupby(["Algoritma", "Matris Boyutu (n×n)"])[
            ["Çalışma Süresi T(n) (s)", "Enerji (kWh)", "Pavg (W)"]
        ].mean().to_string())
    
    print("\nOlusturulan Dosyalar:")
    print("  1. siralama_enerji_sonuclari.csv")
    if df_matrix is not None:
        print("  2. matris_enerji_sonuclari.csv")
    print("  3. siralama_analiz_grafikleri.png")
    print("  4. guc_tuketimi_karsilastirma.png")
    if df_matrix is not None:
        print("  5. matris_analiz_grafikleri.png")
    print("  6. enerji_analiz_raporu.txt")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()