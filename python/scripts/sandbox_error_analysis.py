import os
import numpy as np
import matplotlib.pyplot as plt
import eispac
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def error_model(intensity, a, b):
    # a: scaling factor (gain)
    # b: constant variance term (read noise squared)
    # Clip at 0 to mathematically guarantee no negative values reach sqrt
    return np.sqrt(np.clip(a * np.abs(intensity) + b, 0, None))

def main():
    pathdir = '/home/kamo/resources/slitless/data/eis_data/'
    
    # A selection of dates spanning the mission (based on your reader scripts)
    dates = [
        '20070124_181113',  # Early mission
        '20070512_094534',  # Original test
        '20090114_233904',  # Mid mission
        '20120216_182857',  # Late mission
        '20150807_022045'   # Very late mission
    ]
    
    local_dir = os.path.join(pathdir, 'l2')
    os.makedirs(local_dir, exist_ok=True)
    
    results = []

    for date in dates:
        print(f"\n--- Analyzing scan from {date} ---")
        eis_filepath = os.path.join(local_dir, f'eis_{date}.data.h5')
        
        if not os.path.exists(eis_filepath):
            print(f"Downloading EIS scan for {date}...")
            from slitless.eistools import download_eis
            try:
                download_eis(date, local_dir)
            except Exception as e:
                print(f"Download failed for {date}: {e}")
                continue
                
        try:
            data_cube = eispac.read_cube(eis_filepath, window=195.119)
        except Exception as e:
            print(f"Failed to load/read {date}: {e}")
            continue
        
        # Extract raw arrays
        data = np.array(data_cube.data)
        errs = np.array(data_cube.uncertainty.array)
        
        valid = (data > 0) & (errs > 0) & np.isfinite(data) & np.isfinite(errs)
        x_data = data[valid]
        y_errs = errs[valid]
        
        if len(x_data) > 100000:
            idx = np.random.choice(len(x_data), 100000, replace=False)
            x_data = x_data[idx]
            y_errs = y_errs[idx]

        try:
            popt, pcov = curve_fit(
                error_model, x_data, y_errs, 
                p0=[40.0, 1000.0],
                bounds=([0, 0], [np.inf, np.inf]) # Enforce physical bounds: a > 0, b >= 0
            )
            a_fit, b_fit = popt
            
            y_pred = error_model(x_data, a_fit, b_fit)
            r2 = r2_score(y_errs, y_pred)
            
            results.append((date, a_fit, b_fit, r2))
            print(f"a: {a_fit:.4e}, b: {b_fit:.4e}, R^2: {r2:.6f}")
        except Exception as e:
            print(f"Fitting failed for {date}: {e}")
            
    print("\n================ SUMMARY ACROSS YEARS ================")
    print(f"{'Date':<18} | {'Gain (a)':<12} | {'ReadNoise^2 (b)':<16} | {'R^2 Score':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r[0]:<18} | {r[1]:<12.4f} | {r[2]:<16.4f} | {r[3]:<10.6f}")
    print("======================================================")

if __name__ == '__main__':
    main()