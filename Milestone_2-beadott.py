# Generated from: Milestone_2.ipynb
# Converted at: 2026-11-14T22:08:32.674Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Milestone 2


%pip install kaleido==1.5.0 pandas==2.2.3 numpy==2.1.3 plotly==6.1.1 scikit-learn==1.6.1 -q

TRAIN_PATH = "https://raw.githubusercontent.com/lego111Aron/shelter-animal-outcome-prediction/refs/heads/main/Data/Original/train.csv"
TEST_PATH = "https://raw.githubusercontent.com/lego111Aron/shelter-animal-outcome-prediction/refs/heads/main/Data/Original/test.csv"

TARGET = "OutcomeType"
LEAKAGE_COLS = ['OutcomeSubtype']
ID_COLS = ['AnimalID']
DROP_FEATURES = LEAKAGE_COLS + ID_COLS
RANDOM_STATE = 42

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
PLOTLY_WIDTH = 950
PLOTLY_HEIGHT = 520

# # Data Splitting


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

"""Load data"""
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

"""
Define the target variable (y) and input features (X).
Drop the columns that will not be used during model training 
(the target itself, the target subtype, and the unique identifier).
"""
y = train_df[TARGET]

drop_cols = DROP_FEATURES + [TARGET]
X = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])

print("Selected features:", list(X.columns))

"""
Initialize and execute the Stratified K-Fold cross-validation split.
Stratification ensures that the target variable class proportions 
are preserved across all folds.
"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_df['fold'] = -1

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    train_df.loc[val_idx, 'fold'] = fold_idx

print("Number of rows per fold:")
print(train_df['fold'].value_counts())

"""
Check the distribution of the target variable.
Compare the percentage proportions of the target classes between the 
full dataset and fold 0 to ensure the stratified split worked correctly.
"""
fold_0_dist = train_df[train_df['fold'] == 0]['OutcomeType'].value_counts(normalize=True) * 100
total_dist = train_df['OutcomeType'].value_counts(normalize=True) * 100

check_df = pd.DataFrame({'Total %': total_dist, 'Fold 0 %': fold_0_dist})
print(check_df)

# # Data cleaning


"""Load the original test dataset."""
test_df = pd.read_csv(TEST_PATH)

print("Missing values in train:")
print(train_df[['Name', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']].isnull().sum())

print("\nMissing values in test:")
print(test_df[['Name', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']].isnull().sum())

MONTH_CONVERSION = {
    'year': 12.0,
    'month': 1.0,
    'week': 7 / 30.4375,
    'day': 1 / 30.4375
}

def parse_age_in_months(age_str):
    """
    Parse a string representing age and convert it to a continuous 
    numerical value in months.

    Args:
        age_str (str): The age string to parse (e.g., '2 years', '3 weeks').

    Returns:
        float: The converted age in months, or np.nan if parsing fails.
    """
    if pd.isna(age_str):
        return np.nan
    parts = str(age_str).split()
    if len(parts) < 2:
        return np.nan

    number = float(parts[0])
    unit = parts[1].lower().rstrip('s')

    multiplier = MONTH_CONVERSION.get(unit, np.nan)

    return number * multiplier

"""Apply the conversion function to create a new 'AgeInMonths' numerical column."""
train_df['AgeInMonths'] = train_df['AgeuponOutcome'].apply(parse_age_in_months)
test_df['AgeInMonths'] = test_df['AgeuponOutcome'].apply(parse_age_in_months)

"""
Impute missing values across the datasets.
Categorical columns ('Name', 'SexuponOutcome') are filled with constant placeholders.
The numerical column ('AgeInMonths') is imputed using the median age of the corresponding 'AnimalType'.

To prevent data leakage, the training set imputation is performed fold-by-fold 
using out-of-fold medians. The test set is imputed using overall training set medians.
"""
for df in [train_df, test_df]:
    df['Name'] = df['Name'].fillna('No Name')
    df['SexuponOutcome'] = df['SexuponOutcome'].fillna('Unknown')

for fold in range(5):
    train_fold = train_df[train_df['fold'] != fold]
    val_idx = train_df[train_df['fold'] == fold].index

    type_medians = train_fold.groupby('AnimalType')['AgeInMonths'].median()

    for animal_type, median_val in type_medians.items():
        mask = (train_df.index.isin(val_idx)) & (train_df['AnimalType'] == animal_type) & (train_df['AgeInMonths'].isnull())
        train_df.loc[mask, 'AgeInMonths'] = median_val

overall_medians = train_df.groupby('AnimalType')['AgeInMonths'].median()
for animal_type, median_val in overall_medians.items():
    test_df.loc[(test_df['AnimalType'] == animal_type) & (test_df['AgeInMonths'].isnull()), 'AgeInMonths'] = median_val

"""
Verify that no missing values remain in the specified columns
"""
print("Remaining missing values in train_df:")
print(train_df[['Name', 'SexuponOutcome', 'AgeInMonths']].isnull().sum())

print("\nRemaining missing values in test_df:")
print(test_df[['Name', 'SexuponOutcome', 'AgeInMonths']].isnull().sum())

# # Data transformation


from sklearn.preprocessing import StandardScaler

"""
Concatenate the cleaned training and test datasets into a single DataFrame (df_all) to ensure consistent 
feature engineering and encoding across both sets.
"""
df_all = pd.concat([train_df, test_df], ignore_index=True)

"""
Create numerical features and scale them:
1. Apply log transformation (log1p) to 'AgeInMonths' to handle skewness.
2. Convert 'DateTime' strings to datetime objects.
3. Standardize the new numerical feature using StandardScaler.
"""
df_all['AgeInMonths_log'] = np.log1p(df_all['AgeInMonths'])

df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])

scaler = StandardScaler()
df_all['AgeInMonths_scaled'] = scaler.fit_transform(
    df_all[['AgeInMonths_log']]
)

"""
Encode categorical features for machine learning models:
1. Apply One-Hot Encoding to low-cardinality features ('AnimalType', 'SexuponOutcome').
2. Apply Frequency Encoding to high-cardinality features ('Breed', 'Color') 
   to replace categories with their relative frequencies.
3. Create a binary feature ('has_name') indicating if the animal has a specific name.
"""
df_transformed = pd.get_dummies(df_all, columns=['AnimalType', 'SexuponOutcome'], drop_first=True)

for col in ['Breed', 'Color']:
    freq_map = df_all[col].value_counts(normalize=True).to_dict()
    df_transformed[f'{col}_freq'] = df_all[col].map(freq_map)

df_transformed['has_name'] = (df_transformed['Name'] != 'No Name').astype(int)

# ## Feature Engineering


"""Create new features from the 'DateTime' column to capture temporal patterns."""
df_transformed['DateTime_month'] = df_transformed['DateTime'].dt.month
df_transformed['DateTime_weekday'] = df_transformed['DateTime'].dt.dayofweek
df_transformed['DateTime_hour'] = df_transformed['DateTime'].dt.hour

"""Create a binary feature indicating whether the Breed value contains the word 'Mix'."""
df_transformed['isMix'] = (
    df_transformed['Breed']
    .str.contains(r'\bMix\b', case=False, na=False)
    .astype(int)
)

# ## Feature Selection


df_transformed.columns

"""
Drop original and intermediate columns that are no longer needed.
Split the combined dataset back into the original training and test sets 
based on the presence of the target variable ('OutcomeType').
Finally, drop set-specific redundant columns and save the prepared datasets.
"""
cols_to_drop = ['Name', 'DateTime', 'AgeuponOutcome', 'AgeInMonths', 'AgeInMonths_log', 'Breed', 'Color', 'AnimalID', 'OutcomeSubtype']
df_final = df_transformed.drop(columns=[c for c in cols_to_drop if c in df_transformed.columns])

train_transformed = df_final[df_final['OutcomeType'].notnull()].copy()
test_transformed = df_final[df_final['OutcomeType'].isnull()].copy()

train_transformed = train_transformed.drop(columns=['ID'], errors='ignore')
test_transformed = test_transformed.drop(columns=['OutcomeType', 'OutcomeSubtype', 'fold'], errors='ignore')

train_transformed.to_csv('train_transformed.csv', index=False)
test_transformed.to_csv('test_transformed.csv', index=False)

print(f"Transformed Train shape: {train_transformed.shape}")
print(f"Transformed Test shape: {test_transformed.shape}")

# ## Feature analysis


import plotly.express as px

"""
Normalize the contingency table by columns to inspect the outcome distribution 
separately within purebred and mixed animal groups. Reshape the resulting 
DataFrame into a long format suitable for Plotly, map categorical labels, 
and generate a grouped bar chart to compare their outcome profiles.
"""
mix_outcome_col_pct = (
    pd.crosstab(
        train_transformed["OutcomeType"],
        train_transformed["isMix"],
        normalize="columns",
    )
    * 100
).reset_index()

mix_melted = mix_outcome_col_pct.melt(
    id_vars="OutcomeType", var_name="isMix", value_name="Percentage"
)
mix_melted["isMix"] = mix_melted["isMix"].map({0: "Purebred (0)", 1: "Mix (1)"})

fig2 = px.bar(
    mix_melted,
    x="OutcomeType",
    y="Percentage",
    color="isMix",
    barmode="group",
    title="Outcome Distribution Across Purebred and Mixed Animals (%)",
    labels={
        "OutcomeType": "Outcome Type",
        "Percentage": "Share Within Group (%)",
        "isMix": "Animal Type",
    },
    color_discrete_sequence=["#ff9999", "#66b3ff"],
)
fig2.show()

"""
Calculate row-wise percentage distributions of name availability within each 
outcome category. Melt the table into a long format and plot a grouped bar chart 
to evaluate the predictive impact of animal names on shelter outcomes.
"""
name_outcome_pct = (
    pd.crosstab(
        train_transformed["OutcomeType"],
        train_transformed["has_name"],
        normalize="index",
    )
    * 100
).reset_index()
name_outcome_pct.columns = ["OutcomeType", "No Name (0)", "Has Name (1)"]

name_melted = name_outcome_pct.melt(
    id_vars="OutcomeType", var_name="Name Status", value_name="Percentage (%)"
)

fig3 = px.bar(
    name_melted,
    x="OutcomeType",
    y="Percentage (%)",
    color="Name Status",
    barmode="group",
    title="Distribution of Name Availability Across Outcome Subgroups (%)",
    labels={
        "OutcomeType": "Outcome Type",
        "Percentage (%)": "Proportion Within Outcome (%)",
    },
    color_discrete_sequence=["#ff7f0e", "#1f77b4"],
)
fig3.update_layout(xaxis_tickangle=-15)
fig3.show()

"""
Aggregate outcome frequencies by month and normalize row-wise to examine 
monthly behavioral seasonality. Melt the aggregated matrix and plot a line chart 
to track percentage shifts of each outcome type over the course of a year.
"""
month_outcome = (
    pd.crosstab(
        train_transformed["DateTime_month"],
        train_transformed["OutcomeType"],
        normalize="index",
    )
    * 100
).reset_index()

month_outcome_melted = month_outcome.melt(
    id_vars="DateTime_month",
    var_name="OutcomeType",
    value_name="Percentage",
)

fig4 = px.line(
    month_outcome_melted,
    x="DateTime_month",
    y="Percentage",
    color="OutcomeType",
    markers=True,
    title="Percentage Trend of Outcomes Across Months (Seasonality Analysis)",
    labels={
        "DateTime_month": "Month",
        "Percentage": "Share (%)",
        "OutcomeType": "Outcome Type",
    },
)
fig4.show()

"""
Aggregate outcome frequencies by day of the week and normalize row-wise 
to examine weekly behavioral patterns. Melt the aggregated matrix and plot 
a line chart to track percentage shifts of each outcome type throughout the week.
"""
weekday_outcome = (
    pd.crosstab(
        train_transformed["DateTime_weekday"],
        train_transformed["OutcomeType"],
        normalize="index",
    )
    * 100
).reset_index()

weekday_outcome_melted = weekday_outcome.melt(
    id_vars="DateTime_weekday",
    var_name="OutcomeType",
    value_name="Percentage",
)

fig_weekday = px.line(
    weekday_outcome_melted,
    x="DateTime_weekday",
    y="Percentage",
    color="OutcomeType",
    markers=True,
    title="Percentage Trend of Outcomes Across Days of the Week",
    labels={
        "DateTime_weekday": "Day of the Week (0 = Monday, 6 = Sunday)",
        "Percentage": "Share (%)",
        "OutcomeType": "Outcome Type",
    },
)
fig_weekday.show()

"""
Aggregate outcome frequencies by hour of the day and normalize row-wise 
to examine daily operational patterns. Melt the aggregated matrix and plot 
a line chart to track how outcome proportions fluctuate across different hours.
"""
hour_outcome = (
    pd.crosstab(
        train_transformed["DateTime_hour"],
        train_transformed["OutcomeType"],
        normalize="index",
    )
    * 100
).reset_index()

hour_outcome_melted = hour_outcome.melt(
    id_vars="DateTime_hour",
    var_name="OutcomeType",
    value_name="Percentage",
)

fig_hour = px.line(
    hour_outcome_melted,
    x="DateTime_hour",
    y="Percentage",
    color="OutcomeType",
    markers=True,
    title="Percentage Trend of Outcomes Across Hours of the Day",
    labels={
        "DateTime_hour": "Hour of the Day",
        "Percentage": "Share (%)",
        "OutcomeType": "Outcome Type",
    },
)
fig_hour.show()

"""
Analyze the final feature space dimensions, compute and visualize the correlation 
matrix of numerical and binary features, and automatically detect highly correlated 
feature pairs to handle potential multicollinearity issues.
"""
# Prepare dataset for correlation (exclude 'fold' and non-numeric columns if any)
df_corr = train_transformed.copy()
if "fold" in df_corr.columns:
  df_corr = df_corr.drop(columns=["fold"])

print("=== FEATURE SPACE DIMENSIONS ===")
print(f"Number of rows (samples): {df_corr.shape[0]}")
print(f"Number of columns (features + target): {df_corr.shape[1]}")
print("\nList of available features:")
for col in df_corr.columns:
  print(f" - {col}")

if "DateTime_hour" in df_corr.columns:
    df_corr['hour_sin'] = np.sin(2 * np.pi * df_corr['DateTime_hour'] / 24)
    df_corr['hour_cos'] = np.cos(2 * np.pi * df_corr['DateTime_hour'] / 24)
    df_corr = df_corr.drop(columns=['DateTime_hour'])

if "DateTime_weekday" in df_corr.columns:
    df_corr['weekday_sin'] = np.sin(2 * np.pi * df_corr['DateTime_weekday'] / 7)
    df_corr['weekday_cos'] = np.cos(2 * np.pi * df_corr['DateTime_weekday'] / 7)
    df_corr = df_corr.drop(columns=['DateTime_weekday'])

if "DateTime_month" in df_corr.columns:
    df_corr['month_sin'] = np.sin(2 * np.pi * df_corr['DateTime_month'] / 12)
    df_corr['month_cos'] = np.cos(2 * np.pi * df_corr['DateTime_month'] / 12)
    df_corr = df_corr.drop(columns=['DateTime_month'])

numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
corr_matrix = df_corr[numeric_cols].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Feature Space Correlation Matrix",
)
fig_corr.show()

"""
Identify feature pairs with an absolute correlation coefficient greater than 
a specified threshold (e.g., 0.85) to detect redundant variables.
"""
threshold = 0.85
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = []
for col in upper_tri.columns:
  matches = upper_tri.index[upper_tri[col].abs() > threshold].tolist()
  for row in matches:
    corr_value = upper_tri.loc[row, col]
    high_corr_pairs.append((row, col, corr_value))

print("\n=== MULTICOLLINEARITY CHECK (High Correlations > 0.85) ===")
if high_corr_pairs:
  print(
      f"Found {len(high_corr_pairs)} highly correlated feature pairs (potential"
      " redundancy):"
  )
  for f1, f2, val in high_corr_pairs:
    print(f" - '{f1}' and '{f2}': Correlation = {val:.2f}")
else:
  print(
      "No severe multicollinearity found above the threshold (all feature pairs"
      f" are <= {threshold})."
  )