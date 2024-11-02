def plot_feature_importance(feature_importance, figsize=(12, 8)):
    """
    Creates an enhanced feature importance visualization

    :param feature_importance: DataFrame with feature importance values
    :param figsize: Figure size
    """
    plt.figure(figsize=figsize)

    bars = plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color="skyblue",
        alpha=0.8,
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.5f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance Analysis", pad=20)

    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_analysis(y_true, y_pred, clf, figsize=(15, 5)):
    """
    Creates a comprehensive confusion matrix analysis with metrics

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param clf: Classifier used for prediction
    :param figsize: Figure size
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    classes = clf.classes_
    class_names = [
        f"Class {c}" if isinstance(c, (int, np.integer)) else str(c) for c in classes
    ]

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")

    metrics_data = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1-Score": f1}, index=class_names
    )

    sns.heatmap(metrics_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax2)
    ax2.set_title("Performance Metrics by Class")

    plt.tight_layout()
    plt.show()

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def analyze_model_performance(feature_importance, y_true, y_pred, clf):
    """
    Performs comprehensive model analysis

    :param feature_importance: DataFrame with feature importance values
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param clf: Classifier used for prediction
    """
    print("=== Model Performance Analysis ===\n")

    plot_feature_importance(feature_importance)
    plot_confusion_matrix_analysis(y_true, y_pred, clf)

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Model Accuracy: {accuracy:.3f}")

