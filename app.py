import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

from utils import load_data, infer_task, choose_model, build_pipeline, get_feature_names_after_preprocessing, abbr_to_full


st.set_page_config(layout='wide')
st.title('OHI — Model trainer and visualizer')

# Load data
DATA_PATH = 'Ocean_Health_Index_2018_global_scores.csv'
df = load_data(DATA_PATH)

with st.sidebar:
    st.header('Settings')
    # Show friendly/full names in the dropdown but keep the actual column keys as values.
    # Exclude OBJECTID from choices
    cols = [c for c in df.columns.tolist() if c.upper() != 'OBJECTID']
    # use format_func to display full names when available
    label = st.selectbox('Choose attribute to predict (label)', cols,
                         index=cols.index('Index_') if 'Index_' in cols else 0,
                         format_func=lambda k: abbr_to_full.get(k, k))
    test_size = st.slider('Test set fraction', 0.05, 0.5, 0.2)
    random_state = st.number_input('Random state', value=42, step=1)
    run_train = st.button('Train model')

st.subheader(f'Label preview — {abbr_to_full.get(label, label)}')
st.write(df[[label]].describe(include='all'))

# Let the agent infer task and pick a model
task, reason = infer_task(df[label])
st.markdown(f'**Inferred task:** {task} — {reason}')
estimator = choose_model(task, df[label])
st.markdown(f'**Agent selected estimator:** {estimator.__class__.__name__}')

if run_train:
    st.info('Training — this may take a little while depending on data and model')

    # Prepare features and target
    X = df.drop(columns=[label])
    y = df[label]

    # Remove columns that are obviously identifiers (heuristic)
    identifier_candidates = [c for c in X.columns if c.lower().startswith(('rgn_', 'objectid', 'rgnid', 'rgn', 'rgnid', 'rgn_nam'))]
    X = X.drop(columns=[c for c in identifier_candidates if c in X.columns], errors='ignore')

    # Build pipeline
    pipe = build_pipeline(X, estimator)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    if task == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric('MSE', f'{mse:.3f}')
        st.metric('R²', f'{r2:.3f}')

        # Create smaller figures and display them side-by-side
        fig_main, ax_main = plt.subplots(figsize=(4, 3))
        ax_main.scatter(y_test, y_pred, alpha=0.7)
        ax_main.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax_main.set_xlabel('Actual')
        ax_main.set_ylabel('Predicted')
        ax_main.set_title('Actual vs Predicted')
    else:
        acc = accuracy_score(y_test, y_pred)
        st.metric('Accuracy', f'{acc:.3f}')
        st.text('Classification report:')
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig_main, ax_main = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax_main)
        ax_main.set_xlabel('Predicted')
        ax_main.set_ylabel('Actual')
        ax_main.set_title('Confusion matrix')

    # Feature importance (if supported)
    try:
        estimator_fitted = pipe.named_steps['estimator']
        fi = None
        if hasattr(estimator_fitted, 'feature_importances_'):
            fi = estimator_fitted.feature_importances_

        if fi is not None:
            # Try to obtain feature names after preprocessing
            names = get_feature_names_after_preprocessing(pipe, X)
            # If lengths mismatch, fall back to original X columns
            if len(names) != len(fi):
                names = X.columns.tolist()

            feat_df = pd.DataFrame({'feature': names, 'importance': fi})
            # Keep only the top 10 most important features for the visualization
            feat_df = feat_df.sort_values('importance', ascending=False).head(10)

            # Map any abbreviations to friendly labels when possible
            # Use the helper to map transformed feature names back to friendly labels
            from utils import friendly_label_from_transformed
            feat_df['feature_friendly'] = feat_df['feature'].apply(
                lambda f: friendly_label_from_transformed(f, X.columns.tolist(), abbr_to_full)
            )

            # Create smaller bar plot for feature importances
            fig_feat, ax_feat = plt.subplots(figsize=(4, 3))
            ax_feat.barh(feat_df['feature_friendly'][::-1], feat_df['importance'][::-1])
            ax_feat.set_title('Feature importances')

            # Display main plot and feature importance side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_main)
            with col2:
                st.pyplot(fig_feat)
        else:
            # If no feature importances, just show the main plot in a smaller column
            col1, _ = st.columns([1, 1])
            with col1:
                st.pyplot(fig_main)
    except Exception as e:
        st.warning(f'Could not compute feature importances: {e}')

    # Offer model download
    try:
        joblib.dump(pipe, 'trained_pipeline.joblib')
        with open('trained_pipeline.joblib', 'rb') as f:
            st.download_button('Download trained pipeline (joblib)', f, file_name='trained_pipeline.joblib')
    except Exception as e:
        st.warning(f'Failed to save model: {e}')
