# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.predict import predict, predict_proba

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-box {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .safe-box {
        background-color: #e6ffe6;
        border-left: 5px solid #44ff44;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-header">💳 Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">AI-Powered Transaction Monitoring & Risk Assessment</div>',
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
    This application uses **Machine Learning** to detect potentially fraudulent credit card transactions.
    
    **Model:** XGBoost Classifier  
    **Accuracy:** High precision fraud detection  
    **Dataset:** 284,807 transactions
    """
    )

    st.markdown("---")

    st.header("📊 How It Works")
    st.markdown(
        """
    1. **Upload** transaction data (CSV)
    2. **Analyze** using ML model
    3. **Review** fraud predictions
    4. **Download** results
    """
    )

    st.markdown("---")

    st.header("📋 CSV Format")
    st.code(
        """
Time, V1, V2, ..., V28, Amount
0, -1.35, 1.19, ..., 0.14, 149.62
94, 1.19, 0.26, ..., 0.09, 2.69
    """,
        language="csv",
    )

    st.info("**30 columns required:**\n- Time\n- V1 to V28\n- Amount")

# ==========================================
# MAIN CONTENT TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["📤 Upload & Predict", "📈 Analytics Dashboard", "🔍 Single Transaction"]
)

# ==========================================
# TAB 1: BATCH UPLOAD
# ==========================================
with tab1:
    st.header("📤 Batch Transaction Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            """
        Upload a CSV file containing multiple transactions to analyze them for fraud.
        The system will process all transactions and provide detailed risk assessment.
        """
        )

    with col2:
        st.download_button(
            label="📥 Sample CSV",
            data="""Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
0,-1.359807134,-0.072781173,2.536346738,1.378155224,-0.338320770,0.462387778,0.239598554,0.098697901,0.363786970,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.025790592,0.403992960,0.251412098,-0.018306778,0.277837576,-0.110473910,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62""",
            file_name="sample_transactions.csv",
            mime="text/csv",
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file with transaction data",
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            # Preview section
            with st.expander("👀 Preview Uploaded Data", expanded=True):
                st.dataframe(input_df.head(10), use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(input_df))
                with col2:
                    st.metric("Columns", len(input_df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")

            # Validate columns
            required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            missing_cols = set(required_cols) - set(input_df.columns)

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()

            # Prediction button
            if st.button(
                "🔍 Analyze Transactions", type="primary", use_container_width=True
            ):
                with st.spinner("🔄 Processing transactions... Please wait..."):
                    try:
                        # Get predictions
                        predictions = predict(input_df)
                        probabilities = predict_proba(input_df)

                        # Create comprehensive results
                        result_df = input_df.copy()
                        result_df.insert(
                            0, "Transaction_ID", range(1, len(predictions) + 1)
                        )
                        result_df["Prediction"] = predictions
                        result_df["Prediction_Label"] = [
                            "🚨 Fraud" if p == 1 else "✅ Safe" for p in predictions
                        ]
                        result_df["Fraud_Probability"] = probabilities
                        result_df["Fraud_Probability_%"] = (probabilities * 100).round(
                            2
                        )
                        result_df["Risk_Level"] = pd.cut(
                            probabilities,
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=["🟢 Low", "🟡 Medium", "🔴 High"],
                        )

                        # Summary metrics
                        st.success("✅ Analysis Complete!")
                        st.markdown("---")
                        st.subheader("📊 Summary Statistics")

                        col1, col2, col3, col4 = st.columns(4)

                        fraud_count = (predictions == 1).sum()
                        safe_count = len(predictions) - fraud_count
                        avg_prob = probabilities.mean()
                        high_risk = (probabilities > 0.7).sum()

                        with col1:
                            st.metric(
                                "Total Transactions",
                                f"{len(predictions):,}",
                                help="Total number of transactions analyzed",
                            )

                        with col2:
                            st.metric(
                                "Fraudulent",
                                fraud_count,
                                delta=f"-{fraud_count/len(predictions)*100:.1f}%",
                                delta_color="inverse",
                                help="Transactions predicted as fraud",
                            )

                        with col3:
                            st.metric(
                                "Legitimate",
                                safe_count,
                                delta=f"{safe_count/len(predictions)*100:.1f}%",
                                help="Transactions predicted as legitimate",
                            )

                        with col4:
                            st.metric(
                                "High Risk Count",
                                high_risk,
                                help="Transactions with >70% fraud probability",
                            )

                        # Visualizations
                        st.markdown("---")
                        st.subheader("📈 Visual Analysis")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Fraud distribution pie chart
                            fig_pie = go.Figure(
                                data=[
                                    go.Pie(
                                        labels=["Legitimate", "Fraudulent"],
                                        values=[safe_count, fraud_count],
                                        marker=dict(colors=["#2ecc71", "#e74c3c"]),
                                        hole=0.4,
                                        textinfo="label+percent+value",
                                        textfont=dict(size=14),
                                    )
                                ]
                            )
                            fig_pie.update_layout(
                                title="Transaction Classification",
                                height=400,
                                showlegend=True,
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            # Risk level distribution
                            risk_counts = result_df["Risk_Level"].value_counts()
                            fig_bar = go.Figure(
                                data=[
                                    go.Bar(
                                        x=risk_counts.index,
                                        y=risk_counts.values,
                                        marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
                                        text=risk_counts.values,
                                        textposition="outside",
                                    )
                                ]
                            )
                            fig_bar.update_layout(
                                title="Risk Level Distribution",
                                xaxis_title="Risk Level",
                                yaxis_title="Number of Transactions",
                                height=400,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # Probability distribution
                        fig_hist = px.histogram(
                            result_df,
                            x="Fraud_Probability",
                            nbins=50,
                            title="Fraud Probability Distribution",
                            labels={
                                "Fraud_Probability": "Fraud Probability",
                                "count": "Frequency",
                            },
                            color_discrete_sequence=["#3498db"],
                        )
                        fig_hist.add_vline(
                            x=0.5,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Decision Threshold (0.5)",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Detailed results table
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")

                        # Filter options
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            filter_option = st.selectbox(
                                "Filter by",
                                [
                                    "All Transactions",
                                    "Fraudulent Only",
                                    "Legitimate Only",
                                    "High Risk Only",
                                ],
                            )

                        with col2:
                            sort_option = st.selectbox(
                                "Sort by",
                                [
                                    "Transaction ID",
                                    "Fraud Probability (High to Low)",
                                    "Fraud Probability (Low to High)",
                                ],
                            )

                        # Apply filters
                        filtered_df = result_df.copy()

                        if filter_option == "Fraudulent Only":
                            filtered_df = filtered_df[filtered_df["Prediction"] == 1]
                        elif filter_option == "Legitimate Only":
                            filtered_df = filtered_df[filtered_df["Prediction"] == 0]
                        elif filter_option == "High Risk Only":
                            filtered_df = filtered_df[
                                filtered_df["Fraud_Probability"] > 0.7
                            ]

                        # Apply sorting
                        if sort_option == "Fraud Probability (High to Low)":
                            filtered_df = filtered_df.sort_values(
                                "Fraud_Probability", ascending=False
                            )
                        elif sort_option == "Fraud Probability (Low to High)":
                            filtered_df = filtered_df.sort_values(
                                "Fraud_Probability", ascending=True
                            )

                        # Display filtered results - FIXED VERSION
                        display_cols = [
                            "Transaction_ID",
                            "Prediction_Label",
                            "Fraud_Probability_%",
                            "Risk_Level",
                            "Amount",
                            "Time",
                        ]

                        # Create display dataframe
                        display_df = filtered_df[display_cols].copy()

                        # Color code the rows
                        def color_rows(row):
                            # Check prediction label
                            if "🚨 Fraud" in str(row["Prediction_Label"]):
                                return ["background-color: #ffe6e6"] * len(row)
                            else:
                                return ["background-color: #e6ffe6"] * len(row)

                        # Apply styling
                        styled_df = display_df.style.apply(color_rows, axis=1)

                        # Display
                        st.dataframe(styled_df, use_container_width=True, height=400)

                        st.info(
                            f"📊 Showing {len(filtered_df)} of {len(result_df)} transactions"
                        )

                        # Download section
                        st.markdown("---")
                        st.subheader("📥 Download Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Full results CSV
                            csv_full = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📄 Download Full Results (CSV)",
                                data=csv_full,
                                file_name="fraud_detection_full_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                        with col2:
                            # Fraudulent transactions only
                            fraud_only_df = result_df[result_df["Prediction"] == 1]
                            csv_fraud = fraud_only_df.to_csv(index=False).encode(
                                "utf-8"
                            )
                            st.download_button(
                                label="🚨 Download Fraud Cases Only (CSV)",
                                data=csv_fraud,
                                file_name="fraud_cases_only.csv",
                                mime="text/csv",
                                use_container_width=True,
                                disabled=(len(fraud_only_df) == 0),
                            )

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info(
                """
            **Please ensure your CSV:**
            - Has 30 columns: Time, V1-V28, Amount
            - Contains numeric data
            - Has no missing values
            - Uses comma as delimiter
            """
            )

# ==========================================
# TAB 2: ANALYTICS DASHBOARD
# ==========================================
with tab2:
    st.header("📈 Model Performance Dashboard")

    st.info(
        """
    **Note:** This dashboard shows model performance metrics. 
    To see actual metrics, train the model and save evaluation results as JSON.
    """
    )

    # Placeholder metrics (replace with actual loaded metrics if available)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", "97.5%", help="Overall accuracy on test set")

    with col2:
        st.metric(
            "Precision (Fraud)", "85.3%", help="Of flagged frauds, % actually fraud"
        )

    with col3:
        st.metric("Recall (Fraud)", "92.1%", help="Of actual frauds, % detected")

    with col4:
        st.metric("F1-Score", "88.6%", help="Harmonic mean of precision and recall")

    st.markdown("---")

    # Model info
    st.subheader("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Algorithm:** XGBoost Classifier  
        **Training Data:** 284,807 transactions  
        **Features:** 30 (Time, V1-V28, Amount)  
        **Class Balance:** Handled via SMOTE/class weights  
        """
        )

    with col2:
        st.markdown(
            """
        **Evaluation Metrics:**
        - ROC-AUC Score
        - Precision-Recall AUC
        - Confusion Matrix
        - Classification Report
        """
        )

    st.markdown("---")

    # Feature importance (example)
    st.subheader("🎯 Feature Importance")
    st.info(
        "Feature importance will be displayed here after loading from model metadata."
    )

# ==========================================
# TAB 3: SINGLE TRANSACTION
# ==========================================
with tab3:
    st.header("🔍 Single Transaction Analysis")
    st.markdown("Enter individual transaction details for real-time fraud detection.")

    with st.form("single_transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            time_input = st.number_input(
                "Time (seconds since first transaction)",
                min_value=0,
                value=0,
                help="Seconds elapsed between this and first transaction",
            )

            amount_input = st.number_input(
                "Amount ($)",
                min_value=0.0,
                value=100.0,
                step=0.01,
                help="Transaction amount in dollars",
            )

        with col2:
            st.info(
                """
            **V1-V28 Features:**
            
            These are PCA-transformed features for privacy.
            For testing, you can leave them at default (0).
            """
            )

        # V features in expander
        with st.expander("⚙️ Advanced: V1-V28 Features (Optional)"):
            v_cols = st.columns(4)
            v_features = []

            for i in range(28):
                with v_cols[i % 4]:
                    v_val = st.number_input(
                        f"V{i+1}", value=0.0, format="%.4f", key=f"single_v{i+1}"
                    )
                    v_features.append(v_val)

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction", use_container_width=True, type="primary"
        )

        if submitted:
            # Create transaction dataframe
            transaction_data = {
                "Time": [time_input],
                **{f"V{i+1}": [v_features[i]] for i in range(28)},
                "Amount": [amount_input],
            }
            transaction_df = pd.DataFrame(transaction_data)

            with st.spinner("Analyzing..."):
                try:
                    prediction = predict(transaction_df)[0]
                    probability = predict_proba(transaction_df)[0]

                    st.markdown("---")
                    st.subheader("📊 Analysis Result")

                    # Main result display
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if prediction == 1:
                            st.markdown(
                                '<div class="fraud-box"><h2 style="color: #cc0000;">🚨 FRAUD DETECTED</h2><p>This transaction is highly suspicious</p></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="safe-box"><h2 style="color: #00cc00;">✅ LEGITIMATE</h2><p>This transaction appears safe</p></div>',
                                unsafe_allow_html=True,
                            )

                    with col2:
                        st.metric(
                            "Fraud Probability",
                            f"{probability*100:.2f}%",
                            delta=f"{(probability-0.5)*100:+.1f}% from threshold",
                        )

                    with col3:
                        if probability < 0.3:
                            risk = "🟢 Low Risk"
                        elif probability < 0.7:
                            risk = "🟡 Medium Risk"
                        else:
                            risk = "🔴 High Risk"

                        st.metric("Risk Assessment", risk)

                    # Gauge chart
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=probability * 100,
                            title={"text": "Fraud Probability", "font": {"size": 24}},
                            delta={"reference": 50, "increasing": {"color": "red"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1},
                                "bar": {
                                    "color": (
                                        "darkred"
                                        if probability > 0.5
                                        else "orange" if probability > 0.3 else "green"
                                    )
                                },
                                "bgcolor": "white",
                                "borderwidth": 2,
                                "bordercolor": "gray",
                                "steps": [
                                    {"range": [0, 30], "color": "lightgreen"},
                                    {"range": [30, 70], "color": "lightyellow"},
                                    {"range": [70, 100], "color": "lightcoral"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 50,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommended Actions")

                    if prediction == 1:
                        st.error(
                            """
                        **🚨 IMMEDIATE ACTION REQUIRED:**
                        
                        1. **Block** this transaction immediately
                        2. **Contact** cardholder for verification
                        3. **Review** recent transaction history
                        4. **Flag** account for monitoring
                        5. **Report** to fraud investigation team
                        """
                        )
                    elif probability > 0.5:
                        st.warning(
                            """
                        **⚠️ ELEVATED RISK - MONITOR CLOSELY:**
                        
                        1. **Additional verification** recommended
                        2. **Monitor** for similar patterns
                        3. **Flag** for manual review
                        4. Consider **temporary hold** pending verification
                        """
                        )
                    elif probability > 0.3:
                        st.info(
                            """
                        **ℹ️ MODERATE RISK - ROUTINE MONITORING:**
                        
                        1. **Process** transaction normally
                        2. **Log** for pattern analysis
                        3. **Watch** for anomalies in future transactions
                        """
                        )
                    else:
                        st.success(
                            """
                        **✅ LOW RISK - PROCEED:**
                        
                        1. **Approve** transaction
                        2. **Standard** monitoring applies
                        3. Transaction appears **legitimate**
                        """
                        )

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Credit Card Fraud Detection System</strong></p>
        <p>Built with Machine Learning | Powered by XGBoost & Streamlit</p>
        <p style='font-size: 0.9rem;'>
            💡 Tip: This is a demonstration project for portfolio purposes
        </p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>
            GitHub: <a href='https://github.com/habeebasid/credit-card-fraud-detection' target='_blank'>@habeebasid</a>
        </p>
    </div>
""",
    unsafe_allow_html=True,
)
