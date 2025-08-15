

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, RocCurveDisplay, roc_auc_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# -------------------- App Config --------------------
st.set_page_config(page_title="Loan Default Prediction (Naïve Bayes & Decision Tree)", layout="wide")

# -------------------- Load dataset directly --------------------
df_loaded = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# Keep only relevant columns (if present)
keep = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed',
        'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
        'Credit_History','Property_Area','Loan_Status']
df_loaded = df_loaded[[c for c in keep if c in df_loaded.columns]].copy()

# -------------------- Session State --------------------
if 'df' not in st.session_state:
    st.session_state.df = df_loaded
if 'target' not in st.session_state:
    st.session_state.target = 'Loan_Status'
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = [c for c in df_loaded.columns if c not in ['Loan_Status', 'Loan_ID']]
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'trained' not in st.session_state:
    st.session_state.trained = {}

# -------------------- Helpers --------------------
def encode_target(series: pd.Series) -> pd.Series:
    # map Yes/No and Y/N to 1/0
    mapping = {"Y":1, "N":0, "Yes":1, "No":0, 1:1, 0:0}
    return series.map(lambda v: mapping.get(v, mapping.get(str(v), np.nan))).astype('float').astype('Int64')

def split_X_y(df: pd.DataFrame, target: str, feature_cols: list):
    df = df.copy()
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    if df[target].dtype == object or df[target].dtype.name == 'category':
        df[target] = encode_target(df[target])
    X = df[feature_cols]
    y = df[target]
    return X, y

def build_preprocessor(X: pd.DataFrame, scale_numeric=True) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))

    categorical_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
    ct = ColumnTransformer(
        transformers=[
            ('num', Pipeline(numeric_steps), num_cols),
            ('cat', Pipeline(categorical_steps), cat_cols)
        ]
    )
    return ct

def get_feature_names(ct: ColumnTransformer):
    names = []
    for name, trans, cols in ct.transformers_:
        if hasattr(trans, 'named_steps') and 'ohe' in trans.named_steps:
            ohe = trans.named_steps['ohe']
            try:
                names.extend(ohe.get_feature_names_out(cols).tolist())
            except Exception:
                pass
        else:
            if isinstance(cols, (list, tuple)):
                names.extend(list(cols))
            else:
                names.append(cols)
    return names

def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, random_state=42):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    scoring = {
        'acc': 'accuracy',
        'prec': 'precision',
        'rec': 'recall',
        'f1': 'f1',
        'roc': 'roc_auc'
    }
    cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

    # pooled predictions for CM + ROC curve
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method='predict', n_jobs=-1)
    try:
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    except Exception:
        y_proba = None

    cm = confusion_matrix(y, y_pred)
    pooled_roc = roc_auc_score(y, y_proba) if y_proba is not None else np.nan

    means = {k: float(np.mean(v)) for k, v in cv_results.items() if k.startswith('test_')}
    return means, cm, y_proba, y_pred, cv_results, pooled_roc

def highlight_better(df: pd.DataFrame):
    # highlight max per row (ignore first column 'Metric')
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    if df.shape[1] <= 2:
        return styles
    data_cols = df.columns[1:]
    for i, row in df.iterrows():
        try:
            max_val = np.nanmax(row[data_cols].values.astype(float))
        except Exception:
            max_val = None
        for c in data_cols:
            try:
                val = float(row[c])
                if max_val is not None and np.isfinite(val) and val == max_val:
                    styles.loc[i, c] = 'background-color: #d4edda; color: #155724; font-weight: 600;'
            except Exception:
                pass
    return styles

# -------------------- Pages --------------------
def page_overview():
    st.title("📦 Data Import & Overview")
    df = st.session_state.df
    st.success(f"Loaded dataset with shape {df.shape}")

    st.markdown("### Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("### Summary Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Rows**:", len(df))
        st.write("**Columns**:", len(df.columns))
    with col2:
        st.write("**Missing values (total)**:", int(df.isna().sum().sum()))
        st.write("**Missing by column**:")
        st.dataframe(df.isna().sum().rename('Missing').to_frame())
    with col3:
        st.write("**Unique values by column**:")
        st.dataframe(df.nunique().rename('Unique').to_frame())

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.markdown("### Histograms (numeric)")
        pick = st.multiselect("Select numeric columns", options=num_cols, default=num_cols[:min(6,len(num_cols))])
        cols = st.columns(2)
        for i, c in enumerate(pick):
            with cols[i % 2]:
                fig, ax = plt.subplots()
                ax.hist(df[c].dropna(), bins=30)
                ax.set_title(f"Histogram — {c}")
                st.pyplot(fig)

        st.markdown("### Correlation Matrix")
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, ax=ax)
        ax.set_title("Correlation (numeric)")
        st.pyplot(fig)

def page_preprocess():
    st.title("🧹 Data Preprocessing")
    df = st.session_state.df.copy()
    target = st.session_state.target
    feats = st.session_state.feature_cols

    st.write("**Target:**", target)
    st.write("**Features:**", feats)

    st.markdown("### Missing Values & Encoding")
    st.dataframe(df.isna().sum().rename('Missing').to_frame())

    scale_numeric = st.checkbox("Standardize numeric features", value=True)

    X, y = split_X_y(df, target, feats)
    pre = build_preprocessor(X, scale_numeric=scale_numeric)

    try:
        Xt = pre.fit_transform(X)
        feat_names = get_feature_names(pre)
        st.session_state.preprocessor = pre
        st.success(f"Preprocessor fitted. Transformed shape: {Xt.shape}")

        st.markdown("### Processed Sample (first 5 rows)")
        Xt_df = pd.DataFrame(Xt, columns=feat_names)
        st.dataframe(Xt_df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")

def page_training():
    st.title("🧠 Model Training & 10-Fold CV")
    if st.session_state.df is None or st.session_state.preprocessor is None:
        st.warning("Complete preprocessing first.")
        return

    df = st.session_state.df.copy()
    target = st.session_state.target
    feats = st.session_state.feature_cols

    X, y = split_X_y(df, target, feats)
    pre = st.session_state.preprocessor

    # Models
    st.subheader("Models & Hyperparameters")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Gaussian Naïve Bayes")
        var_smoothing_exp = st.number_input("var_smoothing (exp of 10^x)", value=-0.0, step=0.5)
        gnb = GaussianNB(var_smoothing=10 ** var_smoothing_exp)
    with colB:
        st.markdown("#### Decision Tree")
        max_depth = st.slider("max_depth", 1, 30, 5)
        min_samples_split = st.slider("min_samples_split", 2, 20, 2)
        min_samples_leaf = st.slider("min_samples_leaf", 1, 20, 1)
        criterion = st.selectbox("criterion", ["gini","entropy","log_loss"], index=0)
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )

    pipe_gnb = Pipeline([('pre', pre), ('model', gnb)])
    pipe_dt  = Pipeline([('pre', pre), ('model', dt)])

    st.markdown("---")
    st.subheader("10-Fold CV Results")

    with st.spinner("Evaluating GaussianNB..."):
        gnb_means, gnb_cm, gnb_proba, gnb_pred, gnb_cv, gnb_auc = evaluate_model(pipe_gnb, X, y)
    with st.spinner("Evaluating Decision Tree..."):
        dt_means, dt_cm, dt_proba, dt_pred, dt_cv, dt_auc = evaluate_model(pipe_dt, X, y)

    def show_block(name, means, cm, proba, auc_val):
        col1, col2 = st.columns([2,3])
        with col1:
            st.write(f"**{name} — Mean CV Metrics**")
            mtable = pd.DataFrame({
                'Metric':['Accuracy','Precision','Recall','F1','ROC AUC'],
                'Score':[means.get('test_acc',np.nan),
                         means.get('test_prec',np.nan),
                         means.get('test_rec',np.nan),
                         means.get('test_f1',np.nan),
                         auc_val]
            })
            st.dataframe(mtable, hide_index=True)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cbar=False, ax=ax)
            ax.set_title(f"Confusion Matrix — {name} (pooled)")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        with col2:
            if proba is not None and not np.isnan(auc_val):
                fig, ax = plt.subplots()
                RocCurveDisplay.from_predictions(y_true=y, y_pred=proba, ax=ax)
                ax.set_title(f"ROC Curve — {name} (pooled)")
                st.pyplot(fig)
            else:
                st.info("Probabilities unavailable; ROC skipped.")

    tabs = st.tabs(["GaussianNB","Decision Tree"])
    with tabs[0]:
        show_block("GaussianNB", gnb_means, gnb_cm, gnb_proba, gnb_auc)
    with tabs[1]:
        show_block("Decision Tree", dt_means, dt_cm, dt_proba, dt_auc)

    st.markdown("---")
    st.subheader("Model Comparison (Mean CV)")
    comp = pd.DataFrame({
        'Metric':['Accuracy','Precision','Recall','F1','ROC AUC'],
        'GaussianNB':[gnb_means.get('test_acc',np.nan), gnb_means.get('test_prec',np.nan),
                      gnb_means.get('test_rec',np.nan), gnb_means.get('test_f1',np.nan), gnb_auc],
        'Decision Tree':[dt_means.get('test_acc',np.nan), dt_means.get('test_prec',np.nan),
                         dt_means.get('test_rec',np.nan), dt_means.get('test_f1',np.nan), dt_auc]
    })
    # Color highlight the better score per metric
    styled = comp.style.apply(highlight_better, axis=None)
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.subheader("Decision Tree Diagram (Top 5 Levels)")
    # Fit final DT on full data for diagram & for later pages
    pipe_dt.fit(X, y)
    feature_names = get_feature_names(pre)
    fig, ax = plt.subplots(figsize=(24, 12))
    tree.plot_tree(
        pipe_dt.named_steps['model'],
        feature_names=feature_names,
        class_names=['No','Yes'],
        filled=True,
        max_depth=5,
        fontsize=9,
        ax=ax
    )
    # Scrollable container so large trees don't get cut off
    st.markdown('<div style="overflow-x:auto; padding-bottom: 8px;">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # Save trained models for the Prediction/Conclusions pages
    pipe_gnb.fit(X, y)
    st.session_state.trained = {
        'GaussianNB': pipe_gnb,
        'Decision Tree': pipe_dt
    }

def page_predict():
    st.title("🔮 Prediction")
    if st.session_state.df is None or not st.session_state.trained:
        st.warning("Train final models on the Training page.")
        return

    df = st.session_state.df.copy()
    feats = st.session_state.feature_cols
    target = st.session_state.target

    X, y = split_X_y(df, target, feats)

    # Build a simple form from training data
    with st.form("pred_form"):
        st.write("Enter feature values:")
        inputs = {}
        col1, col2 = st.columns(2)
        for i, c in enumerate(feats):
            if X[c].dtype.kind in 'biufc':
                with (col1 if i % 2 == 0 else col2):
                    default = float(np.nanmedian(pd.to_numeric(X[c], errors='coerce')))
                    val = st.number_input(c, value=default)
                    inputs[c] = val
            else:
                with (col1 if i % 2 == 0 else col2):
                    options = sorted([str(v) for v in X[c].dropna().unique().tolist()])
                    if not options:
                        options = [""]
                    val = st.selectbox(c, options=options, index=0)
                    inputs[c] = val
        model_name = st.selectbox("Model", ["Decision Tree","GaussianNB"], index=0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        mdl = st.session_state.trained[model_name]
        xrow = pd.DataFrame([inputs])[feats]
        try:
            proba = mdl.predict_proba(xrow)[0,1]
        except Exception:
            proba = None
        pred = mdl.predict(xrow)[0]

        st.success(f"Prediction: **{int(pred)}** (1 = Default / Positive class)")
        if proba is not None:
            st.info(f"Estimated probability of default: **{proba:.3f}**")

def page_conclusions():
    st.title("📝 Interpretation & Conclusions")
    if st.session_state.df is None or not st.session_state.trained:
        st.info("Train models to see importances and takeaways.")
        return

    df = st.session_state.df.copy()
    feats = st.session_state.feature_cols
    target = st.session_state.target

    # Feature importance from Decision Tree
    st.markdown("### Decision Tree — Feature Importances")
    dt_pipe = st.session_state.trained.get('Decision Tree')
    if dt_pipe is not None:
        try:
            dt = dt_pipe.named_steps['model']
            pre = dt_pipe.named_steps['pre']
            names = get_feature_names(pre)
            importances = pd.Series(dt.feature_importances_, index=names)
            topk = st.slider("Show Top k", 5, min(30, len(importances)), 10)
            top = importances.sort_values(ascending=False).head(topk)
            fig, ax = plt.subplots(figsize=(6,4))
            top.iloc[::-1].plot(kind='barh', ax=ax)
            ax.set_title("Top Feature Importances")
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Unable to compute importances: {e}")

    st.markdown("### Summary")
    st.write(
        """
        **Model trade-offs**
        - *Gaussian Naïve Bayes* is simple and fast, sometimes better recall if features are conditionally independent.
        - *Decision Tree* captures non-linear splits; can improve precision but needs tuning of depth and leaf sizes.

        **Interpretation notes**
        - Importances reflect transformed features (e.g., one-hot encoded categories for Gender, Education, Property_Area).
        - Typically **Credit_History**, **ApplicantIncome**, **LoanAmount**, and **Dependents** are influential, but verify via importances above.

        **Next steps**
        - Handle class imbalance with class weights or resampling if needed.
        - Try additional models (LogReg/RandomForest/XGBoost), calibrate probabilities, and run SHAP for local explanations.
        """
    )

# -------------------- Navigation --------------------
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox("Go to", [
        '1. Dataset',
        '2. Preprocessing',
        '3. Training & CV',
        '4. Prediction',
        '5. Conclusions'
    ])

if page.startswith('1.'):
    page_overview()
elif page.startswith('2.'):
    page_preprocess()
elif page.startswith('3.'):
    page_training()
elif page.startswith('4.'):
    page_predict()
else:
    page_conclusions()
