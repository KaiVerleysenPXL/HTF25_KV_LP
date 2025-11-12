import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Minimal mapping of common abbreviations to full terms (used for nice plot labels)
abbr_to_full = {
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
    'Index': 'Index',
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

# Add a few region/id name mappings from the dataset
abbr_to_full.update({
    'rgn_typ': 'Region Type',
    'rgn_id': 'Region ID',
    'rgn_nam': 'Region Name',
    'rgn_key': 'Region Key',
    'OBJECTID': 'OBJECTID'
})


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)


def infer_task(y: pd.Series) -> (str, str):
    """Infer whether the prediction task is regression or classification.

    Returns (task, reason).
    Simple rules:
    - If dtype is numeric and number of unique values is large (>20) -> regression
    - If dtype is numeric but few unique values (<=20) -> classification (ordinal/categorical)
    - If dtype is object/category -> classification
    - Fallback to regression
    """
    n_unique = y.nunique(dropna=True)
    dtype = y.dtype
    if pd.api.types.is_numeric_dtype(dtype):
        if n_unique > 20:
            return 'regression', f'numeric with {n_unique} unique values'
        else:
            return 'classification', f'numeric with small cardinality ({n_unique})'
    else:
        return 'classification', f'non-numeric dtype {dtype} with {n_unique} unique values'


def choose_model(task: str, y: pd.Series):
    """Choose a sensible sklearn estimator given task and (optionally) label distribution."""
    if task == 'regression':
        # Use gradient boosting or random forest; RF is a good default
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        return RandomForestClassifier(n_estimators=100, random_state=42)


def build_preprocessing_pipeline(X: pd.DataFrame):
    """Build a ColumnTransformer to process numeric and categorical features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')

    return preprocessor


def build_pipeline(X: pd.DataFrame, estimator):
    """Return a sklearn Pipeline that preprocesses X and fits the estimator."""
    preprocessor = build_preprocessing_pipeline(X)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ])
    return pipe


def get_feature_names_after_preprocessing(pipe: Pipeline, X: pd.DataFrame):
    """Attempt to extract feature names after ColumnTransformer.
    This uses get_feature_names_out when available (sklearn >= 1.0).
    Falls back to a best-effort list.
    """
    pre = pipe.named_steps.get('preprocessor')
    if pre is None:
        return X.columns.tolist()
    try:
        names = pre.get_feature_names_out()
        return names.tolist()
    except Exception:
        # Best-effort: numeric features + categorical feature names from onehot enc
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        names = num_cols[:]
        for c in cat_cols:
            # we don't know unique categories here without fitting; append column name
            names.append(c)
        return names


def friendly_label_from_transformed(transformed_name: str, original_columns: list, abbr_map: dict) -> str:
    """Map a transformed feature name (e.g., 'num__LE' or 'ohe__SP_1') back to a friendly label.

    Strategy:
    - If transformed_name exactly in abbr_map, return mapping.
    - If it contains '__', take the last token after '__' and try to map that.
    - Strip common prefixes like 'num_', 'cat_', 'ohe_', 'onehot_' then try mapping.
    - If the token matches any original column name (or is contained in it), map using that column.
    - Fall back to returning the original transformed_name.
    """
    if transformed_name in abbr_map:
        return abbr_map[transformed_name]

    name = transformed_name
    # common separators/prefixes
    if '__' in name:
        token = name.split('__')[-1]
    else:
        token = name

    # remove common prefixes
    for p in ['num_', 'cat_', 'ohe_', 'onehot_', 'x0_', 'x1_']:
        if token.startswith(p):
            token = token[len(p):]

    # try direct mappings
    if token in abbr_map:
        return abbr_map[token]

    # if token contains a category suffix like 'SP_placeA' or 'SP_placeA',
    # attempt to split into parent and category and format as 'Parent — category'
    if '_' in token:
        parts = token.split('_')
        # assume first part is the parent code and the rest form the category
        parent_code = parts[0]
        category = '_'.join(parts[1:])
        parent_label = abbr_map.get(parent_code)
        if parent_label:
            # beautify category (replace underscores)
            category_clean = category.replace('_', ' ')
            return f"{parent_label} — {category_clean}"

    # try matching against original columns (partial match)
    for col in original_columns:
        if token == col or token in col or col in token:
            return abbr_map.get(col, col)

    # final fallback: return the cleaned token or original name
    return abbr_map.get(token, transformed_name)
