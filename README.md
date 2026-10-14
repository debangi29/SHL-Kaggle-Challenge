# 🎙️ SHL Kaggle Challenge: Spoken Grammar Scoring Engine

A machine learning pipeline to predict English speech grammar scores from audio recordings using automatic speech recognition, acoustic feature engineering, and transformer-based regression.

## 📌 Overview

The goal of this project is to predict a continuous grammar score between **0 and 5** for English speech recordings of approximately 45 to 60 seconds.

The solution uses a two-stage architecture to handle unscorable recordings separately from recordings scored on the standard 1–5 grammar scale.

## 🚀 Methodology

### 1. Audio Preprocessing and Feature Extraction

- Resample audio recordings to 16 kHz mono.
- Process audio files in parallel to improve efficiency.
- Extract acoustic and fluency features, including:
  - Audio duration and RMS loudness.
  - Speech duration and speech-to-audio ratio.
  - Number and length of pauses longer than 0.5 seconds.
- Extract transcript-based features such as word count, speaking rate, lexical diversity, filler-word frequency, and average word length.

### 2. Automatic Speech Recognition

Use **OpenAI Whisper Medium English** (`openai/whisper-medium.en`) to convert speech recordings into text.

- Process audio in chunks and batches.
- Use available GPUs for parallel transcription.
- Cache transcripts to avoid repeated inference.
- Preserve capitalization and punctuation to retain grammatical information.

### 3. Two-Stage Prediction Architecture

The training data contains a separate group of zero-score recordings that cannot be treated as ordinary scores on the 1–5 scale. To address this, the pipeline combines classification and regression.

**Stage 1: Unscorable Clip Detection**

- Train a standardized logistic regression model using acoustic and transcript-derived features.
- Estimate the probability \(p_0\) that a recording belongs to the zero-score group.

**Stage 2: Grammar Score Regression**

- Fine-tune `microsoft/deberta-v3-base` on transcripts of recordings with scores from 1 to 5.
- Use a regression head to predict continuous grammar scores.
- Apply five-fold stratified cross-validation and select the best epoch for each fold.

**Final Prediction**

Combine the two outputs using:

\[
\hat{y}=(1-p_0)\times \hat{y}_{\text{grammar}}
\]

Clip the final prediction to the valid range \([0,5]\). This produces an expected score that accounts for both the probability of an unscorable recording and its predicted grammar score.

## 📊 Evaluation Results

The model is evaluated using Root Mean Squared Error (RMSE) and Pearson correlation.

| Evaluation | RMSE | Pearson Correlation |
|---|---:|---:|
| Regressor only, out-of-fold | 0.9345 | 0.6939 |
| Two-stage model, out-of-fold | **0.6836** | **0.8425** |
| Two-stage model, in-sample training | 0.5037 | 0.9205 |

The two-stage model substantially improves out-of-fold performance compared with using the regressor alone.

The out-of-fold results provide a more reliable estimate of generalization than the in-sample training metrics.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Speech Recognition:** OpenAI Whisper
- **Audio Processing:** Librosa, SoundFile
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Evaluation:** RMSE, Pearson correlation, ROC-AUC
- **Environment:** Kaggle Notebooks with GPU acceleration

## ⚙️ Setup and Execution

### Requirements

- Python environment with the required dependencies.
- Access to the SHL Kaggle Challenge dataset.
- GPU acceleration recommended for Whisper transcription and DeBERTa fine-tuning.
- Internet access enabled in Kaggle to download pretrained models, unless the models are already cached.

### Run the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/debangi29/SHL-Kaggle-Challenge.git
   cd SHL-Kaggle-Challenge
   ```

2. Open `shl-challenge (1).ipynb` in Kaggle or a compatible Jupyter environment.

3. Configure the competition dataset and enable GPU acceleration.

4. Execute the notebook cells sequentially to preprocess audio, extract features, generate transcripts, train the models, evaluate predictions, and create the submission file.

5. The notebook generates `submission.csv` in the configured output directory.

## 📁 Project Structure

```text
SHL-Kaggle-Challenge/
├── shl-challenge (1).ipynb
└── README.md
```

The notebook contains the complete preprocessing, feature engineering, training, evaluation, and submission workflow.

## 🔍 Key Findings

- Separating zero-score recordings from ordinary grammar scoring improves prediction quality.
- Combining acoustic signals with transcript-based features provides useful information for identifying unscorable clips.
- Transformer-based text regression captures linguistic patterns relevant to grammar assessment.
- Five-fold out-of-fold evaluation helps assess generalization on a relatively small dataset.

## ⚠️ Limitations

- The dataset is relatively small, which can limit generalization.
- Whisper may normalize or omit disfluencies and grammatical errors, potentially removing useful signals.
- Regression models can underpredict very high scores and overpredict very low scores.
- In-sample training metrics are optimistic and should not be interpreted as expected test performance.

## 👩‍💻 Author

**Debangi Ghosh**

IIT (BHU), Varanasi

GitHub: [@debangi29](https://github.com/debangi29)

---

*This project was developed as part of the SHL Kaggle Challenge.*
