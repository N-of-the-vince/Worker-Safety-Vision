import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split

def select_features(X, y, original_columns, k=10):
    """Feature selection using Chi-Square and T-Test."""
    # Chi-Square for categorical features (assume non-negative)
    X_nonneg = pd.DataFrame(X).clip(lower=0)  # Ensure non-negative for chi2
    chi_selector = SelectKBest(chi2, k=k)
    chi_selector.fit(X_nonneg, y)
    chi_features = original_columns[chi_selector.get_support()]

    # T-Test (ANOVA F-value) for continuous
    f_selector = SelectKBest(f_classif, k=k)
    f_selector.fit(X, y)
    f_features = original_columns[f_selector.get_support()]

    # Combine and unique
    selected_features = pd.unique(list(chi_features) + list(f_features))
    X_selected = X[:, [list(original_columns).index(f) for f in selected_features]]

    return X_selected, selected_features