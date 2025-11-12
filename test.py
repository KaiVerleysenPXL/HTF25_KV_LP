import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

label = "CW"

# Extract
df = pd.read_csv("Ocean_Health_Index_2018_global_scores.csv")

# Transform
# Eenvoudige data cleaning: duplicaten verwijderen
df.drop_duplicates(inplace=True)

# Load
# Getransformeerde data opslaan als parquet file
df.to_parquet("cleaned_data.parquet", engine='pyarrow', index=False)

# 2. Prepare features and target
# Select only numeric columns for features, excluding target
numeric_cols = df.select_dtypes(include=['number']).columns
X = df[numeric_cols].drop(label, axis=1)
y = df[label]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train a Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# 7. Feature importance
feature_importance = pd.DataFrame({
'feature': X.columns,
'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 8. Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Actual vs Predicted
ax1.scatter(y_test, y_pred, alpha=0.7)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel("Actual Values")
ax1.set_ylabel("Predicted Values")
ax1.set_title("Actual vs Predicted")

# Feature importance
top_features = feature_importance.head(10)
# Map dataset abbreviation column names to friendly full names for plotting.
# If an abbreviation isn't in the map we fall back to the original column name.
abbr_to_full = {
	# Mapping updated to match the OHI attribute names provided by the user
	'AO': 'Artisanal Fisheries',
	'BD': 'Biodiversity',
	'CP': 'Coastal Protection',
	'CS': 'Carbon Storage',
	'CW': 'Clean Water',
	'ECO': 'Economies',
	'FIS': 'Wild Caught Fisheries',
	'FP': 'Food Provision',
	'HAB': 'Habitats',
	'ICO': 'Iconic Species',
	'Index_': 'Index',
	'LE': 'Livelihoods & Economies',
	'LIV': 'Livelihoods',
	'LSP': 'Lasting Special Places',
	'MAR': 'Mariculture',
	'NP': 'Natural Products',
	'SP': 'Sense of Place',
	'SPP': 'Species',
	'TR': 'Tourism & Recreation',
	'trnd_sc': 'Trend Score',
	'are_km2': 'Area (km2)',
	'Shape__Area': 'Shape Area',
	'Shape__Length': 'Shape Length'
}

# Create the horizontal bar chart using numeric y positions so we can explicitly
# set the tick labels (avoids matplotlib warnings and guarantees alignment).
y_pos = list(range(len(top_features)))
ax2.barh(y_pos, top_features['importance'])
ax2.set_xlabel('Feature Importance')
ax2.set_title('Random Forest Feature Importance')
# Replace tick labels with full terms (fall back to original abbrev when unknown)
friendly_labels = [abbr_to_full.get(f, f) for f in top_features['feature']]
ax2.set_yticks(y_pos)
ax2.set_yticklabels(friendly_labels)
# Invert so the most important is on top
ax2.invert_yaxis()
# Increase left margin so longer labels don't get clipped
fig.subplots_adjust(left=0.35)

plt.tight_layout()
plt.show()

# 9. Save the trained model locally
joblib.dump(model, "rf_regression_model.pkl")