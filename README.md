# BrandPulse AI --- Twitter Sentiment Analysis

BrandPulse AI is an NLP-based sentiment analysis project that classifies
tweet-style text into **Positive**, **Negative**, or **Neutral**
sentiment. It compares a classical machine-learning pipeline with a
deep-learning LSTM model and provides an interactive dashboard built
with Streamlit.

## Project overview

The project demonstrates the main stages of an NLP workflow:

1.  Text preprocessing and cleaning
2.  Feature extraction using TF-IDF
3.  Training and evaluating classical machine-learning models
4.  Training and evaluating a deep-learning LSTM model
5.  Saving trained models and preprocessing assets
6.  Performing custom-text predictions through a Streamlit dashboard
7.  Comparing model performance and visualizing sentiment trends

## Features

-   Classifies text into positive, negative, and neutral sentiment.
-   Compares Logistic Regression, Naive Bayes, and Linear SVM using
    TF-IDF features.
-   Includes a Bidirectional LSTM deep-learning model.
-   Provides a simulated live tweet feed and sentiment distribution
    chart.
-   Displays sentiment trends over time.
-   Allows users to test their own text with both trained approaches.
-   Shows a model-comparison table and confusion matrices.

## Repository structure

``` text
BrandPulseAI_Sentiment_Project/
├── 01_Preprocessing_and_Classical.ipynb
├── 02_Deep_Learning_LSTM.ipynb
├── app.py
├── requirements.txt
├── DEPLOYMENT_GUIDE.md
├── data/
│   ├── generate_data.py
│   └── twitter_sentiment_data.csv
├── models/
│   ├── classical_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── label_encoder.joblib
│   ├── lstm_model.keras
│   ├── tokenizer.json
│   ├── classical_metrics.json
│   └── lstm_metrics.json
└── reports/
    ├── Performance_Report.pdf
    ├── confusion_matrix_classical.png
    ├── confusion_matrix_lstm.png
    ├── lstm_training_curves.png
    ├── wordcloud_pos_neg.png
    └── BrandPulseAI_Presentation.pptx
```

## Dataset

The included `data/twitter_sentiment_data.csv` contains **6,300 labeled
text records** with sentiment labels and timestamps. The dataset was
generated using `data/generate_data.py`. It is a template-driven
**synthetic dataset**, not a downloaded or verified collection of real
tweets. It was designed to resemble tweet-style text and includes
examples with URLs, mentions, hashtags, retweet markers, emojis,
repeated characters, sarcasm, and a small amount of label noise.

The generator uses positive, negative, and neutral templates, including
sarcastic negative examples. The generated data is useful for
demonstrating the project workflow, but model scores on it should not be
interpreted as evidence of equivalent performance on real-world
Twitter/X data.

To experiment with a real dataset, prepare a CSV file with at least
these columns:

-   `text`: the tweet or text content
-   `sentiment`: its sentiment label (`positive`, `negative`, or
    `neutral`)

The dashboard also expects a `created_at` column containing timestamps.
If you replace the dataset, ensure the notebooks and dashboard use a
compatible schema and label set.

## Tech stack

-   **Language:** Python
-   **Data processing:** Pandas, NumPy
-   **NLP:** NLTK
-   **Machine learning:** Scikit-learn, TF-IDF
-   **Deep learning:** TensorFlow / Keras (LSTM)
-   **Dashboard and visualization:** Streamlit, Plotly, Matplotlib
-   **Model persistence:** Joblib, Keras model format

## Setup and installation

Use Python 3.10 or a compatible Python version supported by the
installed TensorFlow release.

1.  Clone or download this repository.
2.  Open a terminal in the project root directory
    (`BrandPulseAI_Sentiment_Project`).
3.  Create and activate a virtual environment (recommended).

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

4.  Install dependencies:

``` bash
pip install -r requirements.txt
```

NLTK resources used by the application include English stopwords,
WordNet, and tokenizer data. The application attempts to download
required resources when it starts; an internet connection may be
required the first time.

## Run the Streamlit dashboard

From the project root directory, run:

``` bash
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.
Open that address in your browser.

**Important:** Keep the supplied `data/` and `models/` folders in the
same project root as `app.py`. The application loads the saved model
files and CSV data using relative paths.

## Run the notebooks

The notebooks are intended to be executed in this order:

1.  `01_Preprocessing_and_Classical.ipynb` --- preprocessing, TF-IDF
    feature extraction, classical model training, evaluation, and saved
    classical artifacts.
2.  `02_Deep_Learning_LSTM.ipynb` --- tokenization, sequence
    preparation, LSTM training, evaluation, and saved deep-learning
    artifacts.

Run the notebooks from the project root, or update any paths if you
choose a different working directory. The existing `models/` folder
contains saved artifacts used by the dashboard, so retraining is not
required just to launch the supplied app.

## Model performance

The supplied metrics files report the following results on the project's
evaluation split:

  Model                            Accuracy   Precision   Recall   F1-score
  ------------------------------ ---------- ----------- -------- ----------
  Logistic Regression (TF-IDF)       0.9643      0.9643   0.9643     0.9643
  Naive Bayes (TF-IDF)               0.9643      0.9643   0.9643     0.9643
  Linear SVM (TF-IDF)                0.9643      0.9643   0.9643     0.9643
  LSTM                               0.9643      0.9643   0.9643     0.9643

These are the values recorded in `models/classical_metrics.json` and
`models/lstm_metrics.json`. Because the included dataset is synthetic
and template-generated, these scores are demonstration results and
should not be treated as a benchmark for real-world performance.

For additional analysis, see `reports/Performance_Report.pdf` and the
confusion-matrix and training-curve images in `reports/`.

## Dashboard sections

The Streamlit dashboard contains four main tabs:

-   **Live Stream:** displays a simulated incoming tweet feed and
    sentiment distribution.
-   **Trend Over Time:** plots sentiment counts across hourly timestamps
    in the supplied dataset.
-   **Test Custom Input:** predicts sentiment for user-entered text
    using the selected model.
-   **Model Comparison:** compares evaluation metrics and displays
    confusion matrices.

The live feed is a simulation based on the included dataset; it does not
connect to the Twitter/X API or retrieve live posts.

## Deployment

For deployment instructions, see
[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md). The guide describes
options including Streamlit Community Cloud and Hugging Face Spaces. A
public deployment URL must be created from an account with the hosting
provider; this repository does not include a pre-generated live URL.

## Limitations

-   The supplied tweet dataset is synthetic rather than a real-world
    Twitter/X data export.
-   The live feed is simulated and does not fetch real-time social-media
    posts.
-   Performance may differ substantially on unseen, naturally occurring
    text, slang, multilingual content, and context-dependent sarcasm.
-   Real-world deployment should include evaluation on representative,
    independently labeled data and appropriate monitoring.

## Future improvements

-   Integrate an authorized, policy-compliant source of live
    social-media data.
-   Evaluate on a real, representative, independently labeled dataset.
-   Improve handling of sarcasm, emojis, multilingual text, and
    domain-specific language.
-   Add model monitoring, drift detection, and periodic retraining.
-   Add exportable summaries and brand-level filtering.

## Author / project information

**Project:** BrandPulse AI --- Sentiment Analysis on Twitter Data\
**Category:** Natural Language Processing / Machine Learning\
**Interface:** Streamlit dashboard

Add your name, institution, course, internship organization, or contact
details here if required for your submission.

## License

No license is specified in the supplied project files. Add a license
file and update this section if you intend to distribute or reuse the
project publicly.
