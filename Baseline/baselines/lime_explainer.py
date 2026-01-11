from lime.lime_tabular import LimeTabularExplainer

def explain_with_lime(
    model,
    X_train,
    X_instance,
    feature_names,
    class_names,
    seed,
    top_k=5
):
    """
    Generate a LIME explanation for a single instance.

    Returns:
        List of (condition, weight)
    """
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=seed
    )

    exp = explainer.explain_instance(
        X_instance.values[0],
        model.predict_proba,
        num_features=top_k
    )

    return exp.as_list()
