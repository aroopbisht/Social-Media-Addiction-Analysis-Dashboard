# 📱 Social Media Addiction Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://social-media-addiction-analysis-dashboard.streamlit.app/)

A comprehensive data science project analyzing social media addiction patterns among university students, built with Streamlit and deployed on Streamlit Cloud.

## 🎯 Project Overview

This interactive dashboard analyzes social media usage patterns, addiction levels, and their impact on academic performance and mental health among 705+ university students.

### 📊 Key Features

- **Interactive Filtering**: Filter by gender, age group, academic level, and risk level
- **Real-time Metrics**: Dynamic calculation of key statistics
- **6 Visualization Types**: Bar charts, pie charts, correlation heatmaps, line plots, box plots, and scatter plots
- **Demographic Analysis**: Comprehensive breakdown by various demographics
- **Risk Assessment**: Custom functions for addiction risk classification
- **Intervention Strategies**: Personalized digital detox recommendations
- **Data Export**: Download filtered datasets and analysis results

## 🚀 Live Demo

**Streamlit Cloud**: [Deploy your own here](https://share.streamlit.io)

**Local Demo**: [https://socialmedia-fixed.loca.lt](https://socialmedia-fixed.loca.lt)

## 📁 Project Structure

```
social-media-addiction-analysis/
├── social_media_addiction_streamlit_cloud.py  # Main Streamlit app (Cloud-compatible)
├── social_media_addiction_analysis.py         # Complete analysis script
├── social_media_dashboard_comprehensive.py    # Full-featured dashboard
├── social_media_analysis_colab_guide.html     # Google Colab guide
├── requirements.txt                            # Dependencies
├── README.md                                   # This file
└── data/
    └── Students Social Media Addiction (1).csv # Dataset
```

## 🛠️ Installation & Setup

### Local Development

1. **Clone or download the project files**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run social_media_addiction_streamlit_cloud.py
   ```

### Streamlit Cloud Deployment

1. **Create a GitHub repository** with your project files

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `social_media_addiction_streamlit_cloud.py`
   - Click "Deploy"

3. **Configuration**:
   - The app uses sample data generation for demonstration
   - To use real data, upload your CSV file and modify the data loading function
   - All dependencies are listed in `requirements.txt`

## 📊 Dataset Information

### Sample Dataset Features:
- **705+ students** across different demographics
- **Age range**: 18-24 years (university students)
- **12 key variables**: Usage hours, platform preferences, addiction scores, mental health metrics
- **Platforms**: Instagram, TikTok, Snapchat, Facebook, YouTube, Twitter, LinkedIn

### Key Variables:
- `Student_ID`: Unique identifier
- `Age`: Student age (18-24)
- `Gender`: Male/Female
- `Academic_Level`: High School/Undergraduate/Graduate
- `Avg_Daily_Usage_Hours`: Hours spent on social media daily
- `Most_Used_Platform`: Primary social media platform
- `Affects_Academic_Performance`: Yes/No academic impact
- `Sleep_Hours_Per_Night`: Nightly sleep duration
- `Mental_Health_Score`: 1-10 scale
- `Addicted_Score`: 1-10 addiction level

## 💡 Key Insights

### 🔍 Main Findings
- **Average Daily Usage**: 4.9 hours
- **High Addiction Rate**: 57.9% of students (scores ≥7/10)
- **Academic Impact**: 64.3% report negative effects on studies
- **Mental Health Correlation**: -0.801 (strong negative correlation with usage)

### 👥 Demographic Patterns
- **Age Impact**: Younger students (18-19) show highest addiction scores
- **Gender Differences**: Female students average 5.01 hours vs male 4.83 hours
- **Education Level**: Graduate students show better usage control

### 📱 Platform Preferences
- **Instagram**: Most popular platform (25.8% of users)
- **TikTok**: Highest addiction potential
- **LinkedIn**: Professional platform with lower addiction scores

## 🛠️ Technologies Used

### Core Technologies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations (cloud-compatible)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Deployment
- **Streamlit Cloud**: Free cloud hosting
- **LocalTunnel**: Public URL for local testing
- **GitHub**: Version control and deployment source

## 📈 Visualizations

The dashboard includes 6 types of interactive visualizations:

1. **Bar Charts**: Platform usage, demographic breakdowns
2. **Pie Charts**: Gender distribution, risk levels
3. **Correlation Heatmap**: Variable relationships
4. **Line Plots**: Trends and patterns
5. **Box Plots**: Distribution analysis
6. **Scatter Plots**: Variable correlations

## 🔧 Customization

### Adding Your Data
1. Replace the sample data generation with your CSV file loading:
   ```python
   df = pd.read_csv('your_data.csv')
   ```

2. Ensure your data has the required columns or modify the column mappings

3. Update visualization functions if needed

### Extending Features
- Add new visualization types
- Include additional filtering options
- Implement machine learning predictions
- Add export capabilities

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or support:
- Create an issue in the GitHub repository
- Check the Streamlit documentation
- Review the Google Colab guide included in the project

---

**Built with ❤️ using Streamlit** | **Data Science Project** | **Educational Purpose**
