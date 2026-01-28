import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("COMPREHENSIVE DATA CLEANING - VIN DECODER + MINIMAL DELETION")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load original dataset
df = pd.read_csv('car_prices.csv')
print(f"Original dataset: {len(df):,} records, {len(df.columns)} columns")

# Tracking
cleaning_steps = []
deletion_reasons = []

# Step 1: Standardize capitalization for categorical fields
print("\n" + "="*80)
print("STEP 1: STANDARDIZING CAPITALIZATION")
print("="*80)

categorical_fields = ['make', 'model', 'body', 'transmission', 'state', 'color', 'interior', 'seller']
for field in categorical_fields:
    if field in df.columns:
        before_unique = df[field].nunique()
        # Standardize to title case for better readability
        df[field] = df[field].str.strip().str.title()
        after_unique = df[field].nunique()
        if before_unique != after_unique:
            print(f"  {field}: {before_unique} → {after_unique} unique values (standardized)")
            cleaning_steps.append(f"{field}: Standardized capitalization ({before_unique} → {after_unique} unique)")

# Step 2: VIN Decoding - Build comprehensive lookup tables
print("\n" + "="*80)
print("STEP 2: VIN DECODING - BUILDING LOOKUP TABLES")
print("="*80)

# Create lookup tables from records with complete data
valid_vins = df[df['vin'].notna() & (df['vin'].str.len() == 17)]
print(f"Valid VINs for building lookups: {len(valid_vins):,}")

# Build lookup tables for each attribute
# Make lookup
make_lookup = valid_vins[valid_vins['make'].notna()].groupby('vin')['make'].first().to_dict()

# Model lookup
model_lookup = valid_vins[valid_vins['model'].notna()].groupby('vin')['model'].first().to_dict()

# Body lookup (by make+model+year combination)
body_complete = df[(df['make'].notna()) & (df['model'].notna()) & (df['year'].notna()) & (df['body'].notna())]
body_lookup = body_complete.groupby(['make', 'model', 'year'])['body'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
print(f"Body lookup table: {len(body_lookup):,} combinations")

# Trim lookup (by make+model+year combination)
trim_complete = df[(df['make'].notna()) & (df['model'].notna()) & (df['year'].notna()) & (df['trim'].notna())]
trim_lookup = trim_complete.groupby(['make', 'model', 'year'])['trim'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
print(f"Trim lookup table: {len(trim_lookup):,} combinations")

# Color lookup (by make+model+year combination)
color_complete = df[(df['make'].notna()) & (df['model'].notna()) & (df['year'].notna()) & (df['color'].notna())]
color_lookup = color_complete.groupby(['make', 'model', 'year'])['color'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
print(f"Color lookup table: {len(color_lookup):,} combinations")

# Interior lookup (by make+model+year combination)
interior_complete = df[(df['make'].notna()) & (df['model'].notna()) & (df['year'].notna()) & (df['interior'].notna())]
interior_lookup = interior_complete.groupby(['make', 'model', 'year'])['interior'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
print(f"Interior lookup table: {len(interior_lookup):,} combinations")

cleaning_steps.append(f"Built lookup tables: body ({len(body_lookup):,}), trim ({len(trim_lookup):,}), color ({len(color_lookup):,}), interior ({len(interior_lookup):,})")

# Step 3: Apply VIN decoder to fill missing values
print("\n" + "="*80)
print("STEP 3: APPLYING VIN DECODER LOGIC")
print("="*80)

# Fill make from VIN (vectorized)
make_nulls_before = df['make'].isna().sum()
if make_nulls_before > 0:
    make_mapped = df['vin'].map(make_lookup)
    df.loc[df['make'].isna(), 'make'] = make_mapped[df['make'].isna()]
    make_filled = make_nulls_before - df['make'].isna().sum()
    print(f"Make: Filled {make_filled:,} / {make_nulls_before:,} using VIN lookup")
    cleaning_steps.append(f"Make: VIN decoder filled {make_filled:,} values")

# Fill model from VIN (vectorized)
model_nulls_before = df['model'].isna().sum()
if model_nulls_before > 0:
    model_mapped = df['vin'].map(model_lookup)
    df.loc[df['model'].isna(), 'model'] = model_mapped[df['model'].isna()]
    model_filled = model_nulls_before - df['model'].isna().sum()
    print(f"Model: Filled {model_filled:,} / {model_nulls_before:,} using VIN lookup")
    cleaning_steps.append(f"Model: VIN decoder filled {model_filled:,} values")

# Fill body using make+model+year lookup (vectorized)
body_nulls_before = df['body'].isna().sum()
if body_nulls_before > 0:
    body_key = df[['make', 'model', 'year']].apply(tuple, axis=1)
    body_mapped = body_key.map(body_lookup)
    df.loc[df['body'].isna(), 'body'] = body_mapped[df['body'].isna()]
    body_filled = body_nulls_before - df['body'].isna().sum()
    print(f"Body: Filled {body_filled:,} / {body_nulls_before:,} using make+model+year lookup")
    cleaning_steps.append(f"Body: Decoder filled {body_filled:,} values using make+model+year patterns")

# Fill trim using make+model+year lookup (vectorized)
trim_nulls_before = df['trim'].isna().sum()
if trim_nulls_before > 0:
    trim_key = df[['make', 'model', 'year']].apply(tuple, axis=1)
    trim_mapped = trim_key.map(trim_lookup)
    df.loc[df['trim'].isna(), 'trim'] = trim_mapped[df['trim'].isna()]
    trim_filled = trim_nulls_before - df['trim'].isna().sum()
    print(f"Trim: Filled {trim_filled:,} / {trim_nulls_before:,} using make+model+year lookup")
    cleaning_steps.append(f"Trim: Decoder filled {trim_filled:,} values using make+model+year patterns")

# Fill color using make+model+year lookup (vectorized)
color_nulls_before = df['color'].isna().sum()
if color_nulls_before > 0:
    color_key = df[['make', 'model', 'year']].apply(tuple, axis=1)
    color_mapped = color_key.map(color_lookup)
    df.loc[df['color'].isna(), 'color'] = color_mapped[df['color'].isna()]
    color_filled = color_nulls_before - df['color'].isna().sum()
    print(f"Color: Filled {color_filled:,} / {color_nulls_before:,} using make+model+year lookup")
    cleaning_steps.append(f"Color: Decoder filled {color_filled:,} values using make+model+year patterns")

# Fill interior using make+model+year lookup (vectorized)
interior_nulls_before = df['interior'].isna().sum()
if interior_nulls_before > 0:
    interior_key = df[['make', 'model', 'year']].apply(tuple, axis=1)
    interior_mapped = interior_key.map(interior_lookup)
    df.loc[df['interior'].isna(), 'interior'] = interior_mapped[df['interior'].isna()]
    interior_filled = interior_nulls_before - df['interior'].isna().sum()
    print(f"Interior: Filled {interior_filled:,} / {interior_nulls_before:,} using make+model+year lookup")
    cleaning_steps.append(f"Interior: Decoder filled {interior_filled:,} values using make+model+year patterns")

# Step 4: Statistical imputation for remaining nulls
print("\n" + "="*80)
print("STEP 4: STATISTICAL IMPUTATION")
print("="*80)

# Transmission: mode
transmission_nulls = df['transmission'].isna().sum()
if transmission_nulls > 0:
    transmission_mode = df['transmission'].mode()[0]
    df['transmission'] = df['transmission'].fillna(transmission_mode)
    print(f"Transmission: Filled {transmission_nulls:,} with mode '{transmission_mode}'")
    cleaning_steps.append(f"Transmission: Filled {transmission_nulls:,} with mode '{transmission_mode}'")

# Condition: stratified mean by year (vectorized)
condition_nulls = df['condition'].isna().sum()
if condition_nulls > 0:
    condition_by_year = df.groupby('year')['condition'].mean()
    df.loc[df['condition'].isna(), 'condition'] = df.loc[df['condition'].isna(), 'year'].map(condition_by_year)
    # Fill any remaining with overall mean
    remaining = df['condition'].isna().sum()
    if remaining > 0:
        df['condition'] = df['condition'].fillna(df['condition'].mean())
    print(f"Condition: Filled {condition_nulls:,} with stratified mean by year")
    cleaning_steps.append(f"Condition: Filled {condition_nulls:,} with stratified mean by year")

# Body: mode (for remaining nulls after decoder)
body_nulls = df['body'].isna().sum()
if body_nulls > 0:
    body_mode = df['body'].mode()[0]
    df['body'] = df['body'].fillna(body_mode)
    print(f"Body: Filled remaining {body_nulls:,} with mode '{body_mode}'")
    cleaning_steps.append(f"Body: Filled remaining {body_nulls:,} with mode '{body_mode}'")

# Trim: mode (for remaining nulls after decoder)
trim_nulls = df['trim'].isna().sum()
if trim_nulls > 0:
    trim_mode = df['trim'].mode()[0]
    df['trim'] = df['trim'].fillna(trim_mode)
    print(f"Trim: Filled remaining {trim_nulls:,} with mode '{trim_mode}'")
    cleaning_steps.append(f"Trim: Filled remaining {trim_nulls:,} with mode '{trim_mode}'")

# Color: mode (for remaining nulls after decoder)
color_nulls = df['color'].isna().sum()
if color_nulls > 0:
    color_mode = df['color'].mode()[0]
    df['color'] = df['color'].fillna(color_mode)
    print(f"Color: Filled remaining {color_nulls:,} with mode '{color_mode}'")
    cleaning_steps.append(f"Color: Filled remaining {color_nulls:,} with mode '{color_mode}'")

# Interior: mode (for remaining nulls after decoder)
interior_nulls = df['interior'].isna().sum()
if interior_nulls > 0:
    interior_mode = df['interior'].mode()[0]
    df['interior'] = df['interior'].fillna(interior_mode)
    print(f"Interior: Filled remaining {interior_nulls:,} with mode '{interior_mode}'")
    cleaning_steps.append(f"Interior: Filled remaining {interior_nulls:,} with mode '{interior_mode}'")

# Odometer: stratified median by year (vectorized)
odometer_nulls = df['odometer'].isna().sum()
if odometer_nulls > 0:
    odometer_by_year = df.groupby('year')['odometer'].median()
    df.loc[df['odometer'].isna(), 'odometer'] = df.loc[df['odometer'].isna(), 'year'].map(odometer_by_year)
    # Fill any remaining with overall median
    remaining = df['odometer'].isna().sum()
    if remaining > 0:
        df['odometer'] = df['odometer'].fillna(df['odometer'].median())
    print(f"Odometer: Filled {odometer_nulls:,} with stratified median by year")
    cleaning_steps.append(f"Odometer: Filled {odometer_nulls:,} with stratified median by year")

# MMR and Selling Price: median
mmr_nulls = df['mmr'].isna().sum()
if mmr_nulls > 0:
    df['mmr'] = df['mmr'].fillna(df['mmr'].median())
    print(f"MMR: Filled {mmr_nulls:,} with median")
    cleaning_steps.append(f"MMR: Filled {mmr_nulls:,} with median")

price_nulls = df['sellingprice'].isna().sum()
if price_nulls > 0:
    df['sellingprice'] = df['sellingprice'].fillna(df['sellingprice'].median())
    print(f"Selling Price: Filled {price_nulls:,} with median")
    cleaning_steps.append(f"Selling Price: Filled {price_nulls:,} with median")

# Saledate: mode
saledate_nulls = df['saledate'].isna().sum()
if saledate_nulls > 0:
    saledate_mode = df['saledate'].mode()[0]
    df['saledate'] = df['saledate'].fillna(saledate_mode)
    print(f"Sale Date: Filled {saledate_nulls:,} with mode")
    cleaning_steps.append(f"Sale Date: Filled {saledate_nulls:,} with mode")

# VIN: keep null (can't infer VIN)

# Step 5: State validation
print("\n" + "="*80)
print("STEP 5: STATE VALIDATION")
print("="*80)

valid_states = {
    'Al', 'Ak', 'Az', 'Ar', 'Ca', 'Co', 'Ct', 'De', 'Fl', 'Ga',
    'Hi', 'Id', 'Il', 'In', 'Ia', 'Ks', 'Ky', 'La', 'Me', 'Md',
    'Ma', 'Mi', 'Mn', 'Ms', 'Mo', 'Mt', 'Ne', 'Nv', 'Nh', 'Nj',
    'Nm', 'Ny', 'Nc', 'Nd', 'Oh', 'Ok', 'Or', 'Pa', 'Ri', 'Sc',
    'Sd', 'Tn', 'Tx', 'Ut', 'Vt', 'Va', 'Wa', 'Wv', 'Wi', 'Wy',
    'Dc', 'Pr', 'Vi', 'Gu', 'As', 'Mp'
}

df['valid_state'] = df['state'].isin(valid_states) | df['state'].isna()
invalid_state_count = (~df['valid_state']).sum()
print(f"Invalid state codes found: {invalid_state_count:,}")

if invalid_state_count > 0:
    invalid_examples = df[~df['valid_state']]['state'].value_counts().head(10)
    print(f"Top invalid codes: {dict(invalid_examples)}")

# Step 6: Identify corrupted records (minimal deletion)
print("\n" + "="*80)
print("STEP 6: IDENTIFYING CORRUPTED RECORDS (MINIMAL DELETION)")
print("="*80)

critical_fields = ['year', 'make', 'model', 'sellingprice', 'vin']

# Check for missing critical fields
df['has_critical_data'] = True
for field in critical_fields:
    df['has_critical_data'] &= df[field].notna()

missing_critical = (~df['has_critical_data']).sum()
print(f"Records missing critical fields: {missing_critical:,}")
if missing_critical > 0:
    deletion_reasons.append(f"Missing critical fields (year/make/model/price/vin): {missing_critical:,} records")

# Check for invalid states
invalid_states_final = (~df['valid_state']).sum()
print(f"Records with invalid state codes: {invalid_states_final:,}")
if invalid_states_final > 0:
    deletion_reasons.append(f"Invalid state codes (non-US): {invalid_states_final:,} records")

# Final filter: keep only clean records
df['is_clean'] = df['has_critical_data'] & df['valid_state']
total_to_remove = (~df['is_clean']).sum()

print(f"\nTotal records to delete: {total_to_remove:,} ({(total_to_remove/len(df)*100):.2f}%)")
print("Deletion reasons:")
for reason in deletion_reasons:
    print(f"  - {reason}")

# Apply deletion
df_final = df[df['is_clean']].copy()

# Drop helper columns
helper_cols = ['valid_state', 'has_critical_data', 'is_clean']
df_final = df_final.drop(columns=[col for col in helper_cols if col in df_final.columns])

# Step 7: Final verification
print("\n" + "="*80)
print("STEP 7: FINAL VERIFICATION")
print("="*80)

print(f"Original records: {len(df):,}")
print(f"Final records: {len(df_final):,}")
print(f"Records deleted: {total_to_remove:,} ({(total_to_remove/len(df)*100):.2f}%)")
print(f"Retention rate: {(len(df_final)/len(df)*100):.2f}%")

final_nulls = df_final.isnull().sum()
final_nulls = final_nulls[final_nulls > 0].sort_values(ascending=False)

print("\nRemaining nulls:")
if len(final_nulls) > 0:
    for col, count in final_nulls.items():
        pct = (count / len(df_final)) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("  None - All fields complete!")

# Save final dataset
df_final.to_csv('car_prices_comprehensive_clean.csv', index=False)
print(f"\n[OK] Saved: car_prices_comprehensive_clean.csv")
print(f"    Records: {len(df_final):,}")
print(f"    Columns: {len(df_final.columns)}")

# Generate detailed report
print("\n" + "="*80)
print("GENERATING CLEANING REPORT")
print("="*80)

report = []
report.append("="*80)
report.append("COMPREHENSIVE DATA CLEANING REPORT")
report.append("="*80)
report.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("")
report.append("METHODOLOGY:")
report.append("  1. Capitalization standardization")
report.append("  2. VIN decoder logic (lookup tables from complete records)")
report.append("  3. Statistical imputation (mode, stratified means)")
report.append("  4. State validation (US codes only)")
report.append("  5. Minimal deletion (only corrupted/invalid records)")
report.append("")
report.append(f"INITIAL STATE:")
report.append(f"  Records: {len(df):,}")
report.append(f"  Columns: {len(df.columns)}")
report.append("")
report.append("CLEANING ACTIONS:")
for i, step in enumerate(cleaning_steps, 1):
    report.append(f"  {i}. {step}")
report.append("")
report.append("DELETION SUMMARY:")
report.append(f"  Total deleted: {total_to_remove:,} ({(total_to_remove/len(df)*100):.2f}%)")
report.append("  Reasons:")
for reason in deletion_reasons:
    report.append(f"    - {reason}")
report.append("")
report.append(f"FINAL STATE:")
report.append(f"  Records: {len(df_final):,} ({(len(df_final)/len(df)*100):.2f}% retention)")
report.append(f"  Columns: {len(df_final.columns)}")
report.append(f"  Data completeness: {100 - (df_final.isnull().sum().sum() / (len(df_final) * len(df_final.columns)) * 100):.2f}%")
report.append("")
report.append("FINAL NULL STATUS:")
if len(final_nulls) > 0:
    for col, count in final_nulls.items():
        pct = (count / len(df_final)) * 100
        report.append(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    report.append("  All original columns 100% complete!")
report.append("")
report.append("="*80)
report.append("DATASET READY FOR ANALYSIS & MODELING")
report.append("="*80)

# Save report
report_text = '\n'.join(report)
with open('comprehensive_cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print(f"\n[OK] Report saved: comprehensive_cleaning_report.txt")

print("\n" + "="*80)
print("CLEANING COMPLETE")
print("="*80)
