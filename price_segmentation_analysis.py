import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("PRICE SEGMENTATION ANALYSIS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load dataset
df = pd.read_csv('car_prices_comprehensive_clean.csv')
print(f"Dataset loaded: {len(df):,} records\n")

# Define price segments
segments = {
    'Budget': (500, 10000),
    'Mid-Range': (10000, 30000),
    'Premium': (30000, 100000),
    'Luxury/Exotic': (100000, float('inf')),
    'Unrealistic Low': (0, 500)
}

print("="*80)
print("PRICE SEGMENTATION BREAKDOWN")
print("="*80)

results = []
for segment_name, (min_price, max_price) in segments.items():
    # Filter segment
    if max_price == float('inf'):
        segment_df = df[df['sellingprice'] >= min_price]
    else:
        segment_df = df[(df['sellingprice'] >= min_price) & (df['sellingprice'] < max_price)]
    
    count = len(segment_df)
    pct = (count / len(df)) * 100
    
    if count > 0:
        avg_price = segment_df['sellingprice'].mean()
        median_price = segment_df['sellingprice'].median()
        avg_year = segment_df['year'].mean()
        avg_odometer = segment_df['odometer'].mean()
        top_make = segment_df['make'].mode()[0] if len(segment_df) > 0 else 'N/A'
        top_body = segment_df['body'].mode()[0] if len(segment_df) > 0 else 'N/A'
        
        print(f"\n{segment_name}: ${min_price:,} - ${max_price:,.0f}" if max_price != float('inf') 
              else f"\n{segment_name}: ${min_price:,}+")
        print(f"  Records: {count:,} ({pct:.2f}%)")
        print(f"  Avg Price: ${avg_price:,.2f}")
        print(f"  Median Price: ${median_price:,.2f}")
        print(f"  Avg Year: {avg_year:.0f}")
        print(f"  Avg Odometer: {avg_odometer:,.0f} miles")
        print(f"  Top Make: {top_make}")
        print(f"  Top Body: {top_body}")
        
        results.append({
            'Segment': segment_name,
            'Price Range': f"${min_price:,} - ${max_price:,.0f}" if max_price != float('inf') else f"${min_price:,}+",
            'Count': count,
            'Percentage': pct,
            'Avg Price': avg_price,
            'Median Price': median_price,
            'Avg Year': avg_year,
            'Avg Odometer': avg_odometer,
            'Top Make': top_make,
            'Top Body': top_body
        })

# Create summary DataFrame
summary_df = pd.DataFrame(results)
summary_df.to_csv('price_segmentation_summary.csv', index=False)
print(f"\n[OK] Summary saved to: price_segmentation_summary.csv")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

unrealistic_low = df[df['sellingprice'] < 500]
luxury_exotic = df[df['sellingprice'] > 100000]
realistic_range = df[(df['sellingprice'] >= 500) & (df['sellingprice'] <= 100000)]

print(f"\n1. UNREALISTIC LOW PRICES (<$500): {len(unrealistic_low):,} records ({len(unrealistic_low)/len(df)*100:.2f}%)")
print("   - Likely salvage vehicles, parts cars, or data entry errors")
print("   - Recommendation: Exclude from typical market analysis")

print(f"\n2. LUXURY/EXOTIC (>$100k): {len(luxury_exotic):,} records ({len(luxury_exotic)/len(df)*100:.2f}%)")
print("   - High-end luxury and exotic vehicles")
print("   - Recommendation: Analyze separately due to different market dynamics")

print(f"\n3. REALISTIC RANGE ($500-$100k): {len(realistic_range):,} records ({len(realistic_range)/len(df)*100:.2f}%)")
print("   - Represents typical auction market")
print("   - Recommendation: Use for general market analysis and modeling")

# Market composition
print("\n" + "="*80)
print("MARKET COMPOSITION (REALISTIC RANGE)")
print("="*80)

budget_pct = len(df[(df['sellingprice'] >= 500) & (df['sellingprice'] < 10000)]) / len(realistic_range) * 100
midrange_pct = len(df[(df['sellingprice'] >= 10000) & (df['sellingprice'] < 30000)]) / len(realistic_range) * 100
premium_pct = len(df[(df['sellingprice'] >= 30000) & (df['sellingprice'] <= 100000)]) / len(realistic_range) * 100

print(f"Budget (<$10k): {budget_pct:.1f}%")
print(f"Mid-Range ($10k-$30k): {midrange_pct:.1f}%")
print(f"Premium ($30k-$100k): {premium_pct:.1f}%")

# Create filtered dataset (realistic range only)
df_realistic = realistic_range.copy()
df_realistic.to_csv('car_prices_realistic_range.csv', index=False)
print(f"\n[OK] Realistic range dataset saved: car_prices_realistic_range.csv")
print(f"    Records: {len(df_realistic):,} ({len(df_realistic)/len(df)*100:.2f}% of original)")

# Generate detailed report
report = []
report.append("="*80)
report.append("PRICE SEGMENTATION ANALYSIS REPORT")
report.append("="*80)
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("")
report.append(f"Total Records: {len(df):,}")
report.append("")
report.append("SEGMENTATION BREAKDOWN:")
report.append("-"*80)

for _, row in summary_df.iterrows():
    report.append(f"\n{row['Segment']}: {row['Price Range']}")
    report.append(f"  Count: {row['Count']:,} ({row['Percentage']:.2f}%)")
    report.append(f"  Average Price: ${row['Avg Price']:,.2f}")
    report.append(f"  Median Price: ${row['Median Price']:,.2f}")
    report.append(f"  Average Year: {row['Avg Year']:.0f}")
    report.append(f"  Average Odometer: {row['Avg Odometer']:,.0f} miles")
    report.append(f"  Most Common Make: {row['Top Make']}")
    report.append(f"  Most Common Body: {row['Top Body']}")

report.append("")
report.append("="*80)
report.append("KEY FINDINGS")
report.append("="*80)
report.append(f"1. Unrealistic Low Prices (<$500): {len(unrealistic_low):,} ({len(unrealistic_low)/len(df)*100:.2f}%)")
report.append(f"2. Luxury/Exotic (>$100k): {len(luxury_exotic):,} ({len(luxury_exotic)/len(df)*100:.2f}%)")
report.append(f"3. Realistic Market Range: {len(realistic_range):,} ({len(realistic_range)/len(df)*100:.2f}%)")
report.append("")
report.append("REALISTIC MARKET COMPOSITION:")
report.append(f"  - Budget (<$10k): {budget_pct:.1f}%")
report.append(f"  - Mid-Range ($10k-$30k): {midrange_pct:.1f}%")
report.append(f"  - Premium ($30k-$100k): {premium_pct:.1f}%")
report.append("")
report.append("RECOMMENDATIONS:")
report.append("  1. Use car_prices_realistic_range.csv for general market analysis")
report.append("  2. Exclude <$500 prices (salvage/errors) from statistical analysis")
report.append("  3. Analyze luxury segment (>$100k) separately due to different dynamics")
report.append("  4. Mid-Range segment dominates market - focus modeling efforts here")
report.append("")
report.append("="*80)

report_text = '\n'.join(report)
with open('price_segmentation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFiles Generated:")
print("  1. price_segmentation_summary.csv - Segment statistics")
print("  2. car_prices_realistic_range.csv - Filtered dataset (realistic prices)")
print("  3. price_segmentation_report.txt - Detailed analysis report")
print("\n" + "="*80)
