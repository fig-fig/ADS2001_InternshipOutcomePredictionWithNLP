# Virtual Internship Student Performance Prediction
### ADS2001 Data Challenges 3 | Semester 1 2023

## Introduction
This project investigates **student performance in virtual internship settings** by analysing chat communication data from the Nephrotex virtual internship platform.  
Using natural language processing and machine learning techniques, predictive models were developed to classify student outcome scores based on their communication patterns, engagement metrics, and text sentiment.

## Variables

### Target Variable
- `OutcomeScore` (Student performance score: 0-8)
- `Grade` (Categorised performance: Low, Medium, High)

### Main Features
- **Communication Metrics**
  - `wordCount` – Total words in student messages
  - `m_experimental_testing` – Mentions of experimental testing
  - `m_making_design_choices` – Mentions of design decisions
  - `m_asking_questions` – Question-asking frequency
  - `j_customer_consultants_requests` – Customer/consultant references
  - `j_performance_parameters_requirements` – Performance parameter discussions
  - `j_communication` – General communication indicators

### Additional Variables
- `content` – Student chat message text
- `RoleName` – User role (Student/Mentor)
- `roomName` – Virtual room/activity name
- `implementation` – Implementation group identifier
- `ChatGroup` – Chat group identifier

## Project Workflow

1. **Data Wrangling & Cleaning**
   - Loaded virtual internship chat data (`virtualInternshipData_ADS2001.csv`)
   - Filtered mentor messages from student data
   - Aggregated communication metrics by student and implementation
   - Created categorical `Grade` variable from `OutcomeScore`
   - Handled missing values and data type conversions

2. **Text Analysis & Feature Engineering**
   - **Sentiment Analysis**
     - Used TextBlob and spaCy for sentiment classification
     - Mapped sentiments to numerical values (Positive: 1, Neutral: 0, Negative: -1)
   - **Text Preprocessing**
     - Tokenisation using RegexpTokenizer
     - Stopword removal (standard + custom stopwords)
     - Lemmatisation using WordNetLemmatizer
     - Frequency distribution analysis
   - **Word Cloud Generation**
     - Visualised most common terms in student communications

3. **Exploratory Data Analysis**
   - Distribution of outcome scores across implementations
   - Correlation analysis between communication metrics
   - Grade distribution patterns (Low, Medium, High)
   - Word frequency analysis
   - Heatmap visualisation of feature correlations

4. **Modelling Techniques**
   
   - **Decision Tree Classifier**
     - GridSearchCV for hyperparameter tuning (`max_depth`: 1-5)
     - 5-fold cross-validation
    
   - **Random Forest Classifier**
     - Ensemble method for robust predictions
   
   - **Support Vector Machines (SVM)**
     - Ensemble techniques for improved accuracy
     - Combined multiple base estimators
     - Accuracy: 0.620
    
   - **Logistic Regression**
     - Normalised features using standardisation
     - Multi-class classification for `Grade` prediction
     - Accuracy: 0.696

5. **Evaluation**
   - **Train–Test Split** (80/20)
   - **Metrics Used:**
     - Accuracy Score
     - R² Score
     - Training vs Testing Score comparison
   - **Visualisation:**
     - Distribution plots (Actual vs. Predicted)
     - Feature correlation heatmaps
     - Decision tree visualisation

## Key Insights

- Communication patterns and engagement metrics are predictive of student performance
- Text sentiment analysis reveals emotional tone in student interactions
- Ensemble methods (Random Forest, Bagging, Voting) improve prediction accuracy
- Feature engineering from unstructured text data enhances model performance
- Word frequency and topic mentions correlate with outcome scores
<br>

---

## Built With

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/seaborn-3776AB?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Plotly](https://img.shields.io/badge/plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![TextBlob](https://img.shields.io/badge/TextBlob-4B8BBE?style=for-the-badge)

## License 
This project was developed for academic purposes 2022. <br>
If reused or distributed, please provide proper academic attribution.
