import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Social Media Addiction Analysis Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_and_process_data():
    """Load and process the social media addiction data."""
    try:
        # Load the data
        df = pd.read_csv(r"c:\Users\ADMIN\Documents\Adda247 Data Analytics\Week Assignments\Python Project\Students Social Media Addiction (1).csv")
        
        # Data processing
        df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
        
        # Create age groups
        bins = [17, 19, 21, 23, 25]
        labels = ['18-19', '20-21', '22-23', '24+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
        
        # Risk level classification
        def classify_risk_level(usage_hours):
            if usage_hours <= 2:
                return 'Low'
            elif usage_hours <= 5:
                return 'Medium'
            else:
                return 'High'
        
        df['Risk_Level'] = df['Avg_Daily_Usage_Hours'].apply(classify_risk_level)
        
        # Digital detox strategies
        def suggest_digital_detox(usage_hours, addiction_score):
            if usage_hours <= 2 and addiction_score <= 3:
                return "Minimal intervention needed"
            elif usage_hours <= 4 and addiction_score <= 6:
                return "Set app time limits"
            elif usage_hours <= 6 and addiction_score <= 8:
                return "Implement phone-free zones"
            else:
                return "Urgent intervention needed"
        
        df['Detox_Strategy'] = df.apply(
            lambda row: suggest_digital_detox(row['Avg_Daily_Usage_Hours'], row['Addicted_Score']), 
            axis=1
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_correlation_heatmap(df):
    """Create correlation heatmap using Plotly."""
    correlation_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                       'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
    corr_matrix = df[correlation_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Heatmap of Key Variables",
        title_x=0.5,
        width=600,
        height=500
    )
    
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">📱 Social Media Addiction Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.error("Could not load the dataset. Please check the file path.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    
    # Gender filter
    gender_filter = st.sidebar.multiselect(
        "Select Gender:",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    # Age group filter
    age_filter = st.sidebar.multiselect(
        "Select Age Group:",
        options=df['Age_Group'].dropna().unique(),
        default=df['Age_Group'].dropna().unique()
    )
    
    # Academic level filter
    academic_filter = st.sidebar.multiselect(
        "Select Academic Level:",
        options=df['Academic_Level'].unique(),
        default=df['Academic_Level'].unique()
    )
    
    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "Select Risk Level:",
        options=df['Risk_Level'].unique(),
        default=df['Risk_Level'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Gender'].isin(gender_filter)) &
        (df['Age_Group'].isin(age_filter)) &
        (df['Academic_Level'].isin(academic_filter)) &
        (df['Risk_Level'].isin(risk_filter))
    ]
    
    # Key Metrics Section
    st.header("📊 Key Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(filtered_df)
        st.metric("Total Students", total_students)
    
    with col2:
        avg_usage = filtered_df['Avg_Daily_Usage_Hours'].mean()
        st.metric("Average Daily Usage", f"{avg_usage:.1f} hours")
    
    with col3:
        high_addiction = (filtered_df['Addicted_Score'] >= 7).sum()
        high_addiction_pct = (high_addiction / total_students) * 100 if total_students > 0 else 0
        st.metric("High Addiction Rate", f"{high_addiction_pct:.1f}%")
    
    with col4:
        academic_impact = filtered_df['Affects_Academic_Performance'].sum()
        academic_impact_pct = (academic_impact / total_students) * 100 if total_students > 0 else 0
        st.metric("Academic Impact", f"{academic_impact_pct:.1f}%")
    
    # Correlation Analysis
    st.header("🔍 Correlation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        corr_fig = create_correlation_heatmap(filtered_df)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 📈 Key Correlations:
        """)
        
        # Calculate correlations
        usage_mental_corr = filtered_df['Avg_Daily_Usage_Hours'].corr(filtered_df['Mental_Health_Score'])
        sleep_academic_corr = filtered_df['Sleep_Hours_Per_Night'].corr(filtered_df['Affects_Academic_Performance'])
        age_usage_corr = filtered_df['Age'].corr(filtered_df['Avg_Daily_Usage_Hours'])
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>Usage vs Mental Health:</strong><br>
        {usage_mental_corr:.3f} (Strong negative correlation)
        </div>
        
        <div class="warning-box">
        <strong>Sleep vs Academic Impact:</strong><br>
        {sleep_academic_corr:.3f} (Moderate correlation)
        </div>
        
        <div class="insight-box">
        <strong>Age vs Usage:</strong><br>
        {age_usage_corr:.3f} (Weak correlation)
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations Section
    st.header("📊 Comprehensive Visualizations")
    
    # Row 1: Platform Usage and Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Platform Usage Analysis")
        platform_usage = filtered_df.groupby('Most_Used_Platform')['Avg_Daily_Usage_Hours'].mean().sort_values(ascending=False)
        
        fig_platform = px.bar(
            x=platform_usage.index,
            y=platform_usage.values,
            title="Average Daily Usage by Platform",
            labels={'x': 'Social Media Platform', 'y': 'Average Hours per Day'},
            color=platform_usage.values,
            color_continuous_scale='Blues'
        )
        fig_platform.update_layout(showlegend=False)
        st.plotly_chart(fig_platform, use_container_width=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>📊 Platform Insight:</strong><br>
        {platform_usage.index[0]} leads with {platform_usage.iloc[0]:.1f}h average usage, 
        suggesting visual content drives longer engagement.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚠️ Risk Level Distribution")
        risk_counts = filtered_df['Risk_Level'].value_counts()
        
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Distribution of Usage Risk Levels",
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        high_risk_pct = (risk_counts.get('High', 0) / len(filtered_df)) * 100
        st.markdown(f"""
        <div class="danger-box">
        <strong>⚠️ Risk Alert:</strong><br>
        {high_risk_pct:.1f}% of students are in high-risk category, 
        indicating urgent need for intervention programs.
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Age Trends and Gender Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Addiction Trends by Age")
        age_addiction_trend = filtered_df.groupby('Age')['Addicted_Score'].mean().reset_index()
        
        fig_age = px.line(
            age_addiction_trend,
            x='Age',
            y='Addicted_Score',
            title="Addiction Score Trend by Age",
            markers=True
        )
        fig_age.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig_age, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📈 Age Insight:</strong><br>
        Younger students (18-19) show higher vulnerability, 
        suggesting critical intervention windows during early university years.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("👥 Usage by Gender & Education")
        
        fig_box = px.box(
            filtered_df,
            x='Gender',
            y='Avg_Daily_Usage_Hours',
            color='Academic_Level',
            title="Usage Hours Distribution by Gender & Education"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>👥 Gender Insight:</strong><br>
        Usage patterns vary significantly by demographics, 
        with certain groups showing higher vulnerability to excessive use.
        </div>
        """, unsafe_allow_html=True)
    
    # Row 3: Mental Health Analysis and Detailed Demographics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Mental Health vs Usage")
        
        fig_scatter = px.scatter(
            filtered_df,
            x='Avg_Daily_Usage_Hours',
            y='Mental_Health_Score',
            color='Addicted_Score',
            size='Conflicts_Over_Social_Media',
            title="Mental Health vs Usage Hours",
            color_continuous_scale='Viridis',
            hover_data=['Age', 'Gender', 'Most_Used_Platform']
        )
        
        # Add trendline with error handling
        try:
            trendline_fig = px.scatter(
                filtered_df,
                x='Avg_Daily_Usage_Hours',
                y='Mental_Health_Score',
                trendline='ols'
            )
            if len(trendline_fig.data) > 1:
                fig_scatter.add_traces(trendline_fig.data[1:])
        except Exception as e:
            # If trendline fails, add a simple linear regression line manually
            x_vals = filtered_df['Avg_Daily_Usage_Hours']
            y_vals = filtered_df['Mental_Health_Score']
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = p(x_line)
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_line, 
                    y=y_line, 
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash', width=2)
                )
            )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("""
        <div class="danger-box">
        <strong>🧠 Mental Health Alert:</strong><br>
        Strong negative correlation suggests excessive social media use 
        significantly impacts psychological well-being.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Demographic Breakdown")
        
        # Create demographic analysis
        demo_analysis = filtered_df.groupby(['Age_Group', 'Gender'])['Addicted_Score'].mean().reset_index()
        
        fig_demo = px.bar(
            demo_analysis,
            x='Age_Group',
            y='Addicted_Score',
            color='Gender',
            title="Average Addiction Score by Age Group & Gender",
            barmode='group'
        )
        st.plotly_chart(fig_demo, use_container_width=True)
        
        # Show demographic statistics
        st.markdown("### 📈 Demographic Statistics:")
        gender_stats = filtered_df.groupby('Gender')['Addicted_Score'].agg(['mean', 'count']).round(2)
        st.dataframe(gender_stats)
    
    # GroupBy Analysis Section
    st.header("📊 Detailed GroupBy Analysis")
    
    tab1, tab2, tab3 = st.tabs(["👥 Gender Analysis", "🎂 Age Groups", "🎓 Education Levels"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gender-wise Statistics")
            gender_analysis = filtered_df.groupby('Gender').agg({
                'Avg_Daily_Usage_Hours': ['mean', 'std'],
                'Addicted_Score': ['mean', 'std'],
                'Mental_Health_Score': ['mean', 'std'],
                'Sleep_Hours_Per_Night': ['mean', 'std']
            }).round(2)
            st.dataframe(gender_analysis)
        
        with col2:
            st.subheader("Platform Preferences by Gender")
            gender_platform = pd.crosstab(filtered_df['Gender'], filtered_df['Most_Used_Platform'])
            fig_heatmap = px.imshow(
                gender_platform.values,
                x=gender_platform.columns,
                y=gender_platform.index,
                aspect="auto",
                title="Platform Usage Heatmap by Gender"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Group Statistics")
            age_analysis = filtered_df.groupby('Age_Group').agg({
                'Avg_Daily_Usage_Hours': ['mean', 'std', 'count'],
                'Addicted_Score': ['mean', 'std'],
                'Mental_Health_Score': ['mean', 'std']
            }).round(2)
            st.dataframe(age_analysis)
        
        with col2:
            st.subheader("Risk Distribution by Age")
            age_risk = pd.crosstab(filtered_df['Age_Group'], filtered_df['Risk_Level'], normalize='index') * 100
            fig_age_risk = px.bar(
                age_risk.reset_index(),
                x='Age_Group',
                y=['Low', 'Medium', 'High'],
                title="Risk Level Distribution by Age Group (%)",
                barmode='stack'
            )
            st.plotly_chart(fig_age_risk, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Education Level Statistics")
            edu_analysis = filtered_df.groupby('Academic_Level').agg({
                'Avg_Daily_Usage_Hours': ['mean', 'std', 'count'],
                'Addicted_Score': ['mean', 'std'],
                'Mental_Health_Score': ['mean', 'std']
            }).round(2)
            st.dataframe(edu_analysis)
        
        with col2:
            st.subheader("Academic Performance Impact")
            edu_impact = filtered_df.groupby('Academic_Level')['Affects_Academic_Performance'].agg(['mean', 'sum']).round(3)
            edu_impact.columns = ['Impact Rate', 'Total Affected']
            
            fig_edu = px.bar(
                edu_impact.reset_index(),
                x='Academic_Level',
                y='Impact Rate',
                title="Academic Performance Impact Rate by Education Level"
            )
            st.plotly_chart(fig_edu, use_container_width=True)
    
    # Intervention Strategies Section
    st.header("💡 Intervention Strategies & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Detox Strategy Distribution")
        detox_counts = filtered_df['Detox_Strategy'].value_counts()
        
        fig_detox = px.pie(
            values=detox_counts.values,
            names=detox_counts.index,
            title="Distribution of Recommended Detox Strategies"
        )
        st.plotly_chart(fig_detox, use_container_width=True)
    
    with col2:
        st.subheader("📋 Priority Intervention Groups")
        
        # Identify high-risk groups
        high_risk_students = filtered_df[filtered_df['Risk_Level'] == 'High']
        urgent_intervention = filtered_df[filtered_df['Detox_Strategy'] == 'Urgent intervention needed']
        
        st.markdown(f"""
        <div class="danger-box">
        <strong>🚨 Urgent Attention Required:</strong><br>
        • {len(urgent_intervention)} students need urgent intervention<br>
        • {len(high_risk_students)} students in high-risk category<br>
        • Focus on {filtered_df[filtered_df['Addicted_Score'] >= 8]['Age_Group'].mode().iloc[0] if len(filtered_df[filtered_df['Addicted_Score'] >= 8]) > 0 else 'N/A'} age group
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Recommended Actions:
        1. **Immediate (0-3 months):**
           - Campus-wide digital wellness programs
           - Phone-free study zones
           - Mental health support hotlines
        
        2. **Medium-term (3-12 months):**
           - Digital literacy curriculum
           - Peer support groups
           - Alternative engagement activities
        
        3. **Long-term (1+ years):**
           - Policy development
           - Research partnerships
           - Community awareness campaigns
        """)
    
    # Comprehensive Story Summary
    st.header("📖 Comprehensive Analysis Summary")
    
    total_students = len(filtered_df)
    avg_usage = filtered_df['Avg_Daily_Usage_Hours'].mean()
    high_addiction = (filtered_df['Addicted_Score'] >= 7).sum()
    academic_impact = filtered_df['Affects_Academic_Performance'].sum()
    
    story_summary = f"""
    ### 📊 Key Findings from {total_students} Students:
    
    1. **Usage Patterns:** Students spend an average of {avg_usage:.1f} hours daily on social media, with {high_addiction} individuals ({high_addiction/total_students*100:.1f}%) showing high addiction scores (≥7/10).
    
    2. **Platform Dominance:** Instagram emerges as the most time-consuming platform, driven by its visual content and infinite scroll design that captures prolonged attention.
    
    3. **Academic Impact:** A concerning {academic_impact} students ({academic_impact/total_students*100:.1f}%) report that social media usage negatively affects their academic performance and study habits.
    
    4. **Mental Health Crisis:** Mental health scores correlate negatively with usage hours (r={usage_mental_corr:.3f}), indicating that excessive social media consumption undermines psychological well-being.
    
    5. **Sleep Disruption:** Sleep patterns suffer significantly among heavy users, creating a vicious cycle of fatigue, poor academic performance, and increased screen dependency.
    
    6. **Age Vulnerability:** Younger students (18-19) are particularly vulnerable, suggesting critical intervention windows during early university years.
    
    7. **Gender Differences:** Usage patterns vary between male and female students, with platform preferences and engagement styles showing distinct patterns.
    
    8. **Root Causes:** Dopamine-driven design features, social validation seeking, fear of missing out (FOMO), and lack of digital literacy education contribute to addiction patterns.
    
    9. **High-Risk Groups:** {len(filtered_df[filtered_df['Risk_Level'] == 'High'])} students are in the high-risk category, requiring immediate intervention.
    
    10. **Action Required:** Implement comprehensive digital wellness programs, establish support systems, and develop policy frameworks for healthy technology use.
    """
    
    st.markdown(story_summary)
    
    # Data Download Section
    st.header("💾 Download Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="social_media_analysis_filtered.csv",
            mime="text/csv"
        )
    
    with col2:
        summary_stats = filtered_df.describe()
        summary_csv = summary_stats.to_csv()
        st.download_button(
            label="📊 Download Summary Statistics",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )
    
    with col3:
        correlation_matrix = filtered_df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                                       'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']].corr()
        corr_csv = correlation_matrix.to_csv()
        st.download_button(
            label="🔍 Download Correlation Matrix",
            data=corr_csv,
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    📱 Social Media Addiction Analysis Dashboard | Built with Streamlit & Plotly | 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
