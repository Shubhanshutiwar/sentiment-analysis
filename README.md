# 🔬 SentiScope · NLP Sentiment Analyzer

A modern, interactive web application for real-time sentiment analysis powered by machine learning. Built with **Streamlit**, **Scikit-learn**, and **TF-IDF vectorization**.

Sentiment analysis (also called opinion mining) automatically classifies text as **positive**, **negative**, or **neutral** — enabling quick insights into customer feedback, social media sentiment, and more.

## ✨ Features

- **Real-time Sentiment Classification** - Instantly analyze text and get sentiment predictions
- **Confidence Scoring** - View probability distributions across all sentiment classes
- **Beautiful UI** - Modern, dark-themed interface with smooth gradients and professional typography
- **Text Statistics** - Automatically calculate word count, sentence count, and character count
- **Session History** - Track and view your recent analyses with sentiment badges
- **Class Probabilities** - Explore detailed probability breakdowns for each sentiment class
- **Session Analytics** - Real-time aggregated stats on positive, negative, and neutral sentiments

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Shubhanshutiwar/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app\ \(1\).py
```

The application will launch in your default browser at `http://localhost:8501`

## 📦 Dependencies

- **streamlit** - Interactive web app framework
- **scikit-learn** - Machine learning algorithms and utilities
- **numpy** - Numerical computing
- **nltk** - Natural language processing toolkit

See `requirements.txt` for the complete list.

## 🔧 How It Works

### Model Architecture

The sentiment analyzer uses a **Logistic Regression** classifier with **TF-IDF vectorization**:

1. **Text Preprocessing**
   - Lowercase conversion
   - URL removal
   - Special character filtering
   - Whitespace normalization

2. **Feature Extraction (TF-IDF)**
   - Unigrams and bigrams (1-2 word combinations)
   - Max 10,000 features
   - Sublinear term frequency scaling

3. **Classification**
   - Logistic Regression with LBFGS solver
   - Multi-class probability output
   - Confidence scoring based on max probability

### Training Data

The model is trained on:
- NLTK Movie Reviews corpus (when available)
- 60 hand-crafted training examples (20 positive, 20 negative, 20 neutral)

## 📊 Usage

### Basic Workflow

1. **Enter Text** - Paste or type any text (review, tweet, feedback, etc.)
2. **Click Analyze** - Get instant sentiment classification
3. **Review Results** - See sentiment, confidence, and detailed statistics
4. **Track History** - All analyses are automatically saved in your session

### Example Inputs

#### Positive
```
"This product is absolutely amazing! I love every bit of it."
```

#### Negative
```
"Terrible experience. Waste of money. Would not recommend."
```

#### Neutral
```
"The store opens at 9 AM and closes at 9 PM."
```

## 📁 Project Structure

```
sentiment-analysis/
├── app (1).py                 # Main Streamlit application
├── requirements.txt           # Python dependencies
├── sentiment_model.pkl        # Pre-trained model bundle
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🎨 UI Components

- **Hero Title** - "SentiScope" with tech stack identifier
- **Input Area** - Large text area for text input
- **Analysis Button** - Trigger sentiment analysis
- **Result Card** - Color-coded sentiment display with confidence bar
- **Stats Row** - Word, sentence, character, and confidence metrics
- **Probability Expander** - Detailed class probability breakdown
- **Session Stats** - Aggregate statistics on all analyses
- **History Panel** - Recent analyses with sentiment badges

## 🎯 Color Scheme

- **Positive**: Green (#86efac) - Optimistic and favorable
- **Negative**: Red (#fca5a5) - Critical and unfavorable
- **Neutral**: Blue (#93c5fd) - Objective and factual

## 💾 Model Caching

The trained model is cached using `@st.cache_resource` for optimal performance. On first run:
- If `sentiment_model.pkl` doesn't exist, the model is trained (~20 seconds)
- Subsequent runs load the cached model instantly

## 🔄 Session Management

- **History** - List of analyzed texts with sentiments
- **Analytics** - Counters for total analyses and sentiment breakdowns
- **State Persistence** - Session state is maintained throughout your session

Use the **"Clear History"** button to reset session data.

## 📝 Model Performance Notes

- Trained on limited examples + NLTK corpus
- Works well for clear sentiment expressions
- May struggle with sarcasm, mixed emotions, and domain-specific language
- Ideal for product reviews, social media sentiment, and general feedback

## 🛠️ Customization

You can customize the model by:

1. **Adjusting TF-IDF parameters** in `train_model()`:
   ```python
   vec = TfidfVectorizer(
       max_features=10_000,      # Increase for more features
       ngram_range=(1, 2),       # Change to (1, 3) for trigrams
       sublinear_tf=True,        # Toggle sublinear scaling
       min_df=2                  # Increase to filter rare terms
   )
   ```

2. **Tuning the classifier**:
   ```python
   clf = LogisticRegression(
       max_iter=1000,            # Training iterations
       C=1.0,                    # Regularization strength
       solver="lbfgs"            # Solver algorithm
   )
   ```

3. **Adding more training data** - Extend the `pos`, `neg`, and `neu` lists in `train_model()`

## 📜 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

## 👨‍💻 Author

Created by [Shubhanshutiwar](https://github.com/Shubhanshutiwar)

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📞 Support

For questions or issues, please open a GitHub issue in the repository.

---

**Tech Stack**: Python · Streamlit · Scikit-learn · NLP · Machine Learning
