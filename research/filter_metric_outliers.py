import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV data
df = pd.read_csv('/home/buka2004/DRMNet/validation_outputs/validation_results.csv')

# Display basic statistics
print("Basic Statistics:")
print(df[['rmse', 'ssim', 'psnr']].describe())

# Identify potential outliers (using IQR method)
def find_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

print("\nOutliers in RMSE:")
rmse_outliers = find_outliers_iqr(df, 'rmse')
print(rmse_outliers[['obj_name', 'rmse']].sort_values('rmse', ascending=False).head(10))

print("\nOutliers in PSNR:")
psnr_outliers = find_outliers_iqr(df, 'psnr')
print(psnr_outliers[['obj_name', 'psnr']].sort_values('psnr', ascending=True).head(10))  # ascending=True for worst (lowest PSNR)

# Drop topK outliers (default 20)
topK = 10
rmse_outliers_top = rmse_outliers.nlargest(topK, 'rmse')[['obj_name']]  # Worst: highest RMSE
psnr_outliers_top = psnr_outliers.nsmallest(topK, 'psnr')[['obj_name']]  # Worst: lowest PSNR
all_outliers_to_drop = pd.concat([rmse_outliers_top, psnr_outliers_top]).drop_duplicates()
print(f"\nOutliers to drop (top {topK} from RMSE + top {topK} from PSNR, unique):")
print(all_outliers_to_drop)

# Drop from original df
df_clean = df[~df['obj_name'].isin(all_outliers_to_drop['obj_name'])]
print(f"\nOriginal shape: {df.shape}")
print(f"Clean shape: {df_clean.shape}")
print("Dropped objects:", len(all_outliers_to_drop))

# Save cleaned CSV
df_clean.to_csv('/home/buka2004/DRMNet/validation_outputs/validation_results_cleaned.csv', index=False)
print("\nCleaned CSV saved to: /home/buka2004/DRMNet/validation_outputs/validation_results_cleaned.csv")

# Stats comparison
print("\nCleaned Statistics:")
print(df_clean[['rmse', 'ssim', 'psnr']].describe())
