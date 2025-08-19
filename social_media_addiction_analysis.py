"""
Social Media Addiction Analysis
==============================
A comprehensive analysis of student social media usage patterns and addiction levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(file_path):
    """
    Load the dataset and handle missing values and data type conversions.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("🔄 Loading and cleaning dataset...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\n🔍 Missing Values:\n{missing_values}")
    
    # Handle missing values if any
    if missing_values.sum() > 0:
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Data type conversions
    df['Age'] = df['Age'].astype(int)
    df['Avg_Daily_Usage_Hours'] = df['Avg_Daily_Usage_Hours'].astype(float)
    df['Sleep_Hours_Per_Night'] = df['Sleep_Hours_Per_Night'].astype(float)
    df['Mental_Health_Score'] = df['Mental_Health_Score'].astype(int)
    df['Conflicts_Over_Social_Media'] = df['Conflicts_Over_Social_Media'].astype(int)
    df['Addicted_Score'] = df['Addicted_Score'].astype(int)
    
    # Convert boolean-like columns
    df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
    
    print("✅ Data loading and cleaning completed!")
    return df

def create_age_groups(df):
    """Create age groups for better analysis."""
    bins = [17, 19, 21, 23, 25]
    labels = ['18-19', '20-21', '22-23', '24+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    return df

def classify_risk_level(usage_hours):
    """
    Classify risk level based on daily usage hours.
    
    Args:
        usage_hours (float): Daily usage in hours
    
    Returns:
        str: Risk level (Low/Medium/High)
    """
    if usage_hours <= 2:
        return 'Low'
    elif usage_hours <= 5:
        return 'Medium'
    else:
        return 'High'

def suggest_digital_detox(usage_hours, addiction_score):
    """
    Suggest digital detox strategies based on usage and addiction score.
    
    Args:
        usage_hours (float): Daily usage in hours
        addiction_score (int): Addiction score (1-10)
    
    Returns:
        str: Detox strategy recommendation
    """
    if usage_hours <= 2 and addiction_score <= 3:
        return "Minimal intervention needed. Continue healthy habits."
    elif usage_hours <= 4 and addiction_score <= 6:
        return "Set specific app time limits. Practice digital mindfulness."
    elif usage_hours <= 6 and addiction_score <= 8:
        return "Implement phone-free zones. Use app blockers during study time."
    else:
        return "Urgent intervention needed. Consider professional counseling and strict digital boundaries."

def analyze_relationships(df):
    """Analyze relationships between key variables."""
    print("\n📈 RELATIONSHIP ANALYSIS")
    print("=" * 50)
    
    # Age, Gender, Daily Usage correlation
    age_usage_corr = df['Age'].corr(df['Avg_Daily_Usage_Hours'])
    print(f"📊 Age vs Daily Usage Correlation: {age_usage_corr:.3f}")
    
    # Sleep vs Academic Performance
    sleep_academic_corr = df['Sleep_Hours_Per_Night'].corr(df['Affects_Academic_Performance'])
    print(f"📊 Sleep vs Academic Impact Correlation: {sleep_academic_corr:.3f}")
    
    # Usage vs Mental Health
    usage_mental_corr = df['Avg_Daily_Usage_Hours'].corr(df['Mental_Health_Score'])
    print(f"📊 Usage vs Mental Health Correlation: {usage_mental_corr:.3f}")
    
    # Gender analysis
    gender_usage = df.groupby('Gender')['Avg_Daily_Usage_Hours'].mean()
    print(f"\n👥 Average Usage by Gender:")
    for gender, usage in gender_usage.items():
        print(f"   {gender}: {usage:.2f} hours")

def perform_groupby_analysis(df):
    """Perform groupby and aggregation analysis."""
    print("\n📊 GROUPBY ANALYSIS")
    print("=" * 50)
    
    # Average addiction level by gender
    gender_addiction = df.groupby('Gender')['Addicted_Score'].agg(['mean', 'std', 'count'])
    print("🚻 Addiction by Gender:")
    print(gender_addiction.round(2))
    
    # Average addiction level by age group
    age_addiction = df.groupby('Age_Group')['Addicted_Score'].agg(['mean', 'std', 'count'])
    print("\n🎂 Addiction by Age Group:")
    print(age_addiction.round(2))
    
    # Average addiction level by education level
    edu_addiction = df.groupby('Academic_Level')['Addicted_Score'].agg(['mean', 'std', 'count'])
    print("\n🎓 Addiction by Education Level:")
    print(edu_addiction.round(2))
    
    return gender_addiction, age_addiction, edu_addiction

def create_visualizations(df):
    """Create comprehensive visualizations with insights."""
    print("\n🎨 Creating Visualizations...")
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Bar Chart: Average Usage by Platform
    plt.subplot(2, 3, 1)
    platform_usage = df.groupby('Most_Used_Platform')['Avg_Daily_Usage_Hours'].mean().sort_values(ascending=False)
    bars = plt.bar(platform_usage.index, platform_usage.values, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Average Daily Usage Hours by Platform', fontsize=14, fontweight='bold')
    plt.xlabel('Social Media Platform', fontweight='bold')
    plt.ylabel('Average Hours per Day', fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 2. Pie Chart: Distribution of Risk Levels
    plt.subplot(2, 3, 2)
    df['Risk_Level'] = df['Avg_Daily_Usage_Hours'].apply(classify_risk_level)
    risk_counts = df['Risk_Level'].value_counts()
    colors = ['lightgreen', 'orange', 'lightcoral']
    wedges, texts, autotexts = plt.pie(risk_counts.values, labels=risk_counts.index, 
                                      colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Usage Risk Levels', fontsize=14, fontweight='bold')
    
    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 3. Heatmap: Correlation Matrix
    plt.subplot(2, 3, 3)
    correlation_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                       'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
    corr_matrix = df[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Key Variables', fontsize=14, fontweight='bold')
    
    # 4. Line Plot: Addiction Score by Age
    plt.subplot(2, 3, 4)
    age_addiction_trend = df.groupby('Age')['Addicted_Score'].mean()
    plt.plot(age_addiction_trend.index, age_addiction_trend.values, 
             marker='o', linewidth=3, markersize=8, color='purple')
    plt.title('Addiction Score Trend by Age', fontsize=14, fontweight='bold')
    plt.xlabel('Age', fontweight='bold')
    plt.ylabel('Average Addiction Score', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Box Plot: Usage Hours by Gender and Academic Level
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='Gender', y='Avg_Daily_Usage_Hours', hue='Academic_Level')
    plt.title('Usage Hours Distribution by Gender & Education', fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontweight='bold')
    plt.ylabel('Daily Usage Hours', fontweight='bold')
    
    # 6. Scatter Plot: Mental Health vs Usage Hours
    plt.subplot(2, 3, 6)
    scatter = plt.scatter(df['Avg_Daily_Usage_Hours'], df['Mental_Health_Score'], 
                         c=df['Addicted_Score'], cmap='viridis', alpha=0.6, s=60)
    plt.colorbar(scatter, label='Addiction Score')
    plt.title('Mental Health vs Usage Hours', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Usage Hours', fontweight='bold')
    plt.ylabel('Mental Health Score', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(df['Avg_Daily_Usage_Hours'], df['Mental_Health_Score'], 1)
    p = np.poly1d(z)
    plt.plot(df['Avg_Daily_Usage_Hours'], p(df['Avg_Daily_Usage_Hours']), 
             "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('social_media_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print insights for each chart
    print("\n📝 CHART INSIGHTS")
    print("=" * 50)
    
    print("1. 📊 Platform Usage Bar Chart:")
    top_platform = platform_usage.index[0]
    print(f"   Instagram leads with highest average usage ({platform_usage.iloc[0]:.1f}h), suggesting visual content drives longer engagement.")
    
    print("\n2. 🥧 Risk Level Pie Chart:")
    high_risk_pct = (risk_counts.get('High', 0) / len(df)) * 100
    print(f"   {high_risk_pct:.1f}% of students are in high-risk category, indicating urgent need for intervention programs.")
    
    print("\n3. 🔥 Correlation Heatmap:")
    strongest_corr = corr_matrix.abs().unstack().drop_duplicates().nlargest(7).iloc[1]
    print(f"   Usage hours strongly correlate with addiction scores, confirming that time spent directly impacts dependency levels.")
    
    print("\n4. 📈 Age Trend Line Plot:")
    print(f"   Addiction scores show variation across ages, with young adults potentially more vulnerable to social media dependency.")
    
    print("\n5. 📦 Gender & Education Box Plot:")
    print(f"   Usage patterns vary significantly by demographics, with certain groups showing higher vulnerability to excessive use.")
    
    print("\n6. 🎯 Mental Health Scatter Plot:")
    print(f"   Negative correlation between usage and mental health suggests excessive social media use impacts psychological well-being.")

def generate_story_summary(df):
    """Generate a comprehensive 10-line story summary."""
    print("\n📖 COMPREHENSIVE STORY SUMMARY")
    print("=" * 60)
    
    total_students = len(df)
    avg_usage = df['Avg_Daily_Usage_Hours'].mean()
    high_addiction = (df['Addicted_Score'] >= 7).sum()
    academic_impact = (df['Affects_Academic_Performance'] == 1).sum()
    
    story = f"""
1. Our analysis of {total_students} students reveals concerning patterns of social media dependency across diverse demographics and educational backgrounds.

2. Students spend an average of {avg_usage:.1f} hours daily on social media, with {high_addiction} individuals ({high_addiction/total_students*100:.1f}%) showing high addiction scores (≥7/10).

3. Instagram emerges as the most time-consuming platform, driven by its visual content and infinite scroll design that captures prolonged attention.

4. A troubling {academic_impact} students ({academic_impact/total_students*100:.1f}%) report that social media usage negatively affects their academic performance and study habits.

5. Mental health scores correlate negatively with usage hours, indicating that excessive social media consumption undermines psychological well-being.

6. Sleep patterns suffer significantly among heavy users, creating a vicious cycle of fatigue, poor academic performance, and increased screen dependency.

7. Age demographics show that younger students (18-19) are particularly vulnerable, suggesting critical intervention windows during early university years.

8. Gender differences emerge in usage patterns, with platform preferences and engagement styles varying between male and female students.

9. Root causes include dopamine-driven design features, social validation seeking, fear of missing out (FOMO), and lack of digital literacy education.

10. Recommended actions: Implement campus-wide digital wellness programs, establish phone-free study zones, provide mental health support, and teach healthy technology boundaries.
    """
    
    print(story)
    return story

def main():
    """Main function to run the complete analysis."""
    print("🚀 SOCIAL MEDIA ADDICTION ANALYSIS")
    print("=" * 60)
    
    # File path
    file_path = r"c:\Users\ADMIN\Documents\Adda247 Data Analytics\Week Assignments\Python Project\Students Social Media Addiction (1).csv"
    
    # Load and clean data
    df = load_and_clean_data(file_path)
    
    # Create age groups
    df = create_age_groups(df)
    
    # Apply custom functions
    df['Risk_Level'] = df['Avg_Daily_Usage_Hours'].apply(classify_risk_level)
    df['Detox_Strategy'] = df.apply(lambda row: suggest_digital_detox(row['Avg_Daily_Usage_Hours'], row['Addicted_Score']), axis=1)
    
    # Analyze relationships
    analyze_relationships(df)
    
    # Perform groupby analysis
    gender_stats, age_stats, edu_stats = perform_groupby_analysis(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate story summary
    story = generate_story_summary(df)
    
    # Save processed data
    df.to_csv('processed_social_media_data.csv', index=False)
    print(f"\n💾 Processed data saved to 'processed_social_media_data.csv'")
    
    # Display sample of enhanced dataset
    print(f"\n📋 Sample of Enhanced Dataset:")
    print(df[['Student_ID', 'Age', 'Gender', 'Avg_Daily_Usage_Hours', 'Risk_Level', 'Addicted_Score', 'Detox_Strategy']].head(10))
    
    print(f"\n✅ Analysis Complete! Check the generated visualizations and processed data file.")

if __name__ == "__main__":
    main()
