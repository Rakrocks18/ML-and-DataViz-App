import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import matplotlib.pyplot as plt
import io
from PIL import Image

def visualize_tree(model, feature_names, class_names):
    """Create tree visualization using either graphviz or matplotlib"""
    try:
        import graphviz
        # Try graphviz first
        dot_data = export_graphviz(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None
        )
        
        # Create graph from DOT data
        graph = graphviz.Source(dot_data)
        
        # Save and return the graph
        graph.render("decision_tree", format="png", cleanup=True)
        return "graphviz", "decision_tree.png"
        
    except (ImportError, Exception) as e:
        # Fallback to matplotlib if graphviz fails
        plt.figure(figsize=(20, 10))
        plot_tree(
            model, 
            feature_names=feature_names,
            class_names=class_names,
            filled=True, 
            rounded=True,
            fontsize=10
        )
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return "matplotlib", buf

def decision_tree_page():
    st.title("Decision Tree Classification")

    if "df" not in st.session_state:
        st.error("Please upload data first.")
        return

    df = st.session_state.df

    # Feature selection
    features = st.multiselect("Select features", df.columns)
    target = st.selectbox("Select target variable", df.columns)

    if not features or not target:
        st.warning("Please select both features and target variable to proceed.")
        return

    if target in features:
        features.remove(target)
        st.info(f"Removed {target} from features as it's the target variable.")

    if features and target:
        # Prepare the data
        X = df[features].copy()
        y = df[target].copy()

        # Handle categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])

        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            class_names = [str(target_encoder.inverse_transform([i])[0]) for i in range(len(target_encoder.classes_))]
        else:
            class_names = [str(i) for i in sorted(y.unique())]

        # Use best model params (can also be dynamic with GridSearchCV)
        st.markdown("### ⚙️ Best Model Parameters (Pre-selected)")
        best_params = {
            "max_depth": 30,
            "min_samples_split": 2,
            "criterion": "gini",
            "test_size": 0.2
        }

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Max Depth:** {best_params['max_depth']}")
            st.write(f"**Criterion:** {best_params['criterion']}")
        with col2:
            st.write(f"**Min Samples Split:** {best_params['min_samples_split']}")
            st.write(f"**Test Size:** {best_params['test_size']}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=best_params["test_size"], random_state=42
            )

            if st.button("Train Best Model"):
                model = DecisionTreeClassifier(
                    max_depth=best_params["max_depth"],
                    min_samples_split=best_params["min_samples_split"],
                    criterion=best_params["criterion"],
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                # Display results
                st.subheader("📊 Model Results")
                st.write("**Best Model:**", model)
                st.write("**Best Score:**", round(test_score, 4))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{train_score:.4f}")
                with col2:
                    st.metric("Testing Accuracy", f"{test_score:.4f}")

                # 🎯 Decision Tree Visualization comes AFTER showing results
                st.subheader("🌳 Decision Tree Visualization")
                viz_method, viz_data = visualize_tree(model, features, class_names)
                st.image(viz_data, use_column_width=True)

                # Feature importance
                st.subheader("🔥 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig_importance = px.bar(importance_df, x='Feature', y='Importance',
                                        title='Feature Importance Plot')
                st.plotly_chart(fig_importance)

                # Confusion matrix
                st.subheader("📌 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm,
                                   labels=dict(x="Predicted", y="Actual"),
                                   title="Confusion Matrix",
                                   color_continuous_scale="Viridis")
                st.plotly_chart(fig_cm)

                # Classification Report
                st.subheader("🧾 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                # Tree Info
                st.subheader("📐 Tree Info")
                st.write(f"Number of nodes: {model.tree_.node_count}")
                st.write(f"Tree depth: {model.get_depth()}")
                st.write(f"Number of leaves: {model.get_n_leaves()}")

                # Rules
                st.subheader("📚 Decision Rules")

                def get_rules(tree, feature_names, class_names):
                    tree_ = tree.tree_
                    feature_name = [
                        feature_names[i] if i != -2 else "undefined!"
                        for i in tree_.feature
                    ]

                    paths = []
                    path = []

                    def recurse(node, path, paths):
                        if tree_.feature[node] != -2:
                            name = feature_name[node]
                            threshold = tree_.threshold[node]
                            p1, p2 = path + [(name, "<=", threshold)], path + [(name, ">", threshold)]
                            recurse(tree_.children_left[node], p1, paths)
                            recurse(tree_.children_right[node], p2, paths)
                        else:
                            path += [(class_names[np.argmax(tree_.value[node])], tree_.value[node])]
                            paths += [path]

                    recurse(0, path, paths)

                    rules = []
                    for path in paths:
                        rule = "IF "
                        for i, (feature, operation, value) in enumerate(path[:-1]):
                            if i != 0:
                                rule += " AND "
                            rule += f"{feature} {operation} {value:.2f}"
                        rule += f" THEN {path[-1][0]}"
                        rules.append(rule)
                    return rules

                rules = get_rules(model, features, class_names)
                for i, rule in enumerate(rules, 1):
                    st.write(f"Rule {i}: {rule}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure all features are properly encoded and contain valid data.")
