# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# st.title("🐚Abalone  Age Prediction App")
# st.header('prediction of the Age of Abalone Using Machine Learning')

# # Load model
# @st.cache_resource
# def load_model():
#     with open('model/abalone_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# model = load_model()

# # Input form
# def get_input():
#     sex = st.selectbox('Select Sex', ['Male', 'Female', 'Infant'])
#     length = st.number_input('Length', min_value=0.0, max_value=3.0, step=0.01)
#     height = st.number_input('Height', min_value=0.0, max_value=3.0, step=0.01)
#     whole_weight = st.number_input('Whole Weight', min_value=0.0, max_value=5.0, step=0.01)
#     shell_weight = st.number_input('Shell Weight', min_value=0.0, max_value=3.0, step=0.01)

#     sex = {'Male': 'M', 'Female': 'F', 'Infant': 'I'}[sex]

#     # Put into dataframe with correct column names
#     data = pd.DataFrame({
#         'sex': [sex],
#         'length': [length],
#         'height': [height],
#         'whole_weight': [whole_weight],
#         'shell_weight': [shell_weight]
#     })

#     return data

# features = get_input()

# # Prediction
# if st.button("Predict"):
#     prediction = model.predict(features)
#     st.success(f"The Age (Rings) of Abalone is: {prediction[0]}")










import re
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="Abalone Age Predictor", page_icon="🐚", layout="centered")
st.title("🐚Abalone Age Prediction App")
st.header("Prediction of the Age of Abalone Using Machine Learning")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(path="model/abalone_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------
# Simple user input (same fields you had)
# -------------------------
def get_input():
    sex = st.selectbox('Select Sex', ['M', 'F', 'I'])
    length = st.number_input('Length', min_value=0.0, max_value=3.0, step=0.01)
    height = st.number_input('Height', min_value=0.0, max_value=3.0, step=0.01)
    whole_weight = st.number_input('Whole Weight', min_value=0.0, max_value=5.0, step=0.01)
    shell_weight = st.number_input('Shell Weight', min_value=0.0, max_value=3.0, step=0.01)

    df = pd.DataFrame({
        'sex': [sex],
        'length': [length],
        'height': [height],
        'whole weight': [whole_weight],   # keep same as you had, we'll auto-match
        'shell_weight': [shell_weight]
    })
    return df

features = get_input()

# Debug toggle (optional)
debug = st.sidebar.checkbox("Show debug info", value=False)

# -------------------------
# Helpers: normalize names and detect expected features
# -------------------------
def _normalize(s: str) -> str:
    # remove spaces and underscores, lowercase -> helps match "whole weight" and "whole_weight" to same key
    return re.sub(r'[\s_]+', '', s).lower()

def get_expected_features_from_model(m):
    # 1) sklearn >= some versions have feature_names_in_
    if hasattr(m, 'feature_names_in_'):
        return list(m.feature_names_in_)

    # 2) If pipeline, try to locate a ColumnTransformer and read its column lists
    if isinstance(m, Pipeline):
        for name, step in m.named_steps.items():
            # direct ColumnTransformer
            if hasattr(step, 'transformers_'):
                cols = []
                for tr in step.transformers_:
                    # tr = (name, transformer, columns)
                    cols_spec = tr[2]
                    if isinstance(cols_spec, (list, tuple, np.ndarray)):
                        cols.extend(list(cols_spec))
                if cols:
                    return cols
            # nested pipeline: search deeper
            if hasattr(step, 'named_steps'):
                for sub in step.named_steps.values():
                    if hasattr(sub, 'transformers_'):
                        cols = []
                        for tr in sub.transformers_:
                            cols_spec = tr[2]
                            if isinstance(cols_spec, (list, tuple, np.ndarray)):
                                cols.extend(list(cols_spec))
                        if cols:
                            return cols
    return None

# try to detect expected cols
expected_cols = get_expected_features_from_model(model)
if debug:
    st.sidebar.write("Detected model type:", type(model))
    st.sidebar.write("Detected expected_cols:", expected_cols)

# -------------------------
# Align input to expected columns (if detected)
# -------------------------
aligned = features.copy()
if expected_cols is not None:
    # build normalized maps
    input_map = { _normalize(c): c for c in features.columns }
    expected_norm_map = { _normalize(c): c for c in expected_cols }

    missing = []
    aligned_dict = {}
    for en_norm, en_orig in expected_norm_map.items():
        if en_norm in input_map:
            aligned_dict[en_orig] = features[input_map[en_norm]].values
        else:
            missing.append(en_orig)

    if missing:
        st.error("Model expects these columns but they are missing from the input: " + ", ".join(missing))
        st.info("Your app is currently sending these input columns: " + ", ".join(list(features.columns)))
        st.markdown("**Suggestions:**")
        st.write("- Make sure column names used here match exactly the column names used when training/saving the pipeline.")
        st.write("- Common mismatches: `whole_weight` vs `whole weight`, `shell_weight` vs `shell weight`.")
        st.write("- If your model expects encoded sex values (e.g. 'Male','Female','Infant') but your app sends 'M','F','I', map them before prediction.")
        st.stop()
    else:
        # construct aligned dataframe in the expected order
        aligned = pd.DataFrame({k: v for k, v in aligned_dict.items()})

if debug:
    st.write("Input features (original):")
    st.write(features)
    st.write("Input features (aligned to model expectation):")
    st.write(aligned)

# -------------------------
# Optional small automatic mapping for sex values (safe attempt)
# -------------------------
# If the model was trained with 'Male'/'Female' but you supply 'M'/'F', this maps them.
# This mapping only triggers if model expects a 'sex' column whose unique training categories look like 'Male' etc.
def auto_map_sex_if_needed(df, model):
    if 'sex' not in df.columns:
        return df
    val = str(df['sex'].iloc[0])
    if val.upper() in ['M','F','I']:
        # attempt to detect if model's categorical encoder expects full words
        # we try to find categories from first OneHot/Ordinal encoder inside pipeline (best-effort)
        try:
            # Try to extract categories_ from an encoder somewhere in pipeline
            cats = None
            if isinstance(model, Pipeline):
                for step in model.named_steps.values():
                    # ColumnTransformer
                    if hasattr(step, 'transformers_'):
                        for tr in step.transformers_:
                            trans = tr[1]
                            # If transformer is OneHotEncoder itself
                            if hasattr(trans, 'categories_'):
                                cats = [c.tolist() for c in trans.categories_][0] if len(trans.categories_)>0 else None
                            # If transformer is a Pipeline, search inside
                            if hasattr(trans, 'named_steps'):
                                for sub in trans.named_steps.values():
                                    if hasattr(sub, 'categories_'):
                                        cats = [c.tolist() for c in sub.categories_][0] if len(sub.categories_)>0 else None
                            if cats:
                                break
                    if cats:
                        break
            if cats:
                # if cats look like ['Male','Female','Infant'] and we have 'M' passed, map
                if any(x.lower().startswith('m') for x in cats):
                    mapping = {'M':'Male','F':'Female','I':'Infant'}
                    mapped = mapping.get(val.upper(), val)
                    df = df.copy()
                    df['sex'] = mapped
        except Exception:
            # best-effort only; don't crash
            pass
    return df

aligned = auto_map_sex_if_needed(aligned, model)

# -------------------------
# Prediction with clear error handling
# -------------------------
if st.button("Predict"):
    try:
        pred = model.predict(aligned)
        st.success(f"The Age (Rings) of Abalone is: {pred[0]}")
    except NotFittedError as nfe:
        st.error("Model or one of its transformers appears to be not fitted.")
        st.write("Quick fix: in your training script call `pipeline.fit(X_train, y_train)` then re-save the fitted pipeline:")
        st.code("""
# after training
pipeline.fit(X_train, y_train)
import pickle
with open('model/abalone_model.pkl','wb') as f:
    pickle.dump(pipeline, f)
        """)
        st.exception(nfe)
    except Exception as e:
        # if message contains 'not fitted' give same help
        txt = str(e).lower()
        if 'not fitted' in txt:
            st.error("Detected 'not fitted' error. See quick fix above.")
        st.error("Prediction failed with the following error:")
        st.exception(e)
