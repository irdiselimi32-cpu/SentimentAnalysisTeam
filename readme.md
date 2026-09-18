---- ROADMAP FOR ALBANIAN SENTIMENT ANALYSIS PREPROCESSING ----


1. Project Setup ==================================

Create the project folder structure (data, scripts, results)
Create and activate Python virtual environment (.venv)
Install required libraries
Add project dependencies to requirements.txt
Prepare the project documentation in README.md


2. Dataset Inspection ==================================

Load the Albanian YouTube comment datasets
Inspect columns, labels and missing values
Analyze sentiment class distribution
Identify duplicate comments
Identify conflicting sentiment labels
Analyze RAW and corrected text versions


3. Master Dataset Creation ==================================

Standardize sentiment labels into:
POSITIVE
NEGATIVE
NEUTRAL

Convert Mixed labels into NEUTRAL
Remove invalid or missing observations
Remove duplicate comments with the same label
Identify and exclude conflicting annotations
Create the final Master Dataset

Final Master Dataset:
10,345 comments

POSITIVE: 5,365
NEGATIVE: 2,574
NEUTRAL: 2,406


4. Train / Validation / Test Split ==================================

Apply stratified dataset splitting

Training set: 70%
Validation set: 15%
Test set: 15%

Use random seed = 42
Preserve the same split for every experiment
Ensure no overlap between Train, Validation and Test


5. Albanian Text Analysis ==================================

Compare RAW comments with corrected versions
Analyze spelling variations
Analyze missing ë and ç characters
Identify repeated letters
Identify informal Albanian expressions
Identify abbreviations and slang
Analyze punctuation, emojis and other symbols

Use only Training data when deriving preprocessing rules
Avoid data leakage from Validation and Test


6. Preprocessing Strategy Setup ==================================

Prepare four preprocessing strategies:

RAW
Minimal
Traditional
Albanian-Aware

Apply every strategy to the exact same dataset split
Generate separate Train, Validation and Test files
Verify that IDs and labels remain identical across variants


7. RAW Pipeline ==================================

Keep the original comment unchanged
Preserve spelling variations
Preserve emojis
Preserve punctuation
Preserve capitalization
Preserve repeated letters

Use RAW as the experimental baseline


8. Minimal Preprocessing ==================================

Apply Unicode NFC normalization
Remove URLs
Remove HTML elements
Normalize unnecessary whitespace

Preserve:
Emojis
Punctuation
Capitalization
Repeated letters
Negation


9. Traditional Preprocessing ==================================

Apply Unicode normalization
Remove URLs and HTML
Convert text to lowercase
Remove emojis
Remove punctuation and symbols
Normalize whitespace

Do not remove negation
Do not apply stemming or lemmatization


10. Albanian-Aware Preprocessing ==================================

Apply basic text normalization
Normalize selected Albanian non-standard forms

Examples:

eshte -> është
shum -> shumë
qfar -> çfarë
njerz -> njerëz
flm -> faleminderit
dmth -> domethënë

Reduce excessive repeated letters
Preserve sentiment-related information
Preserve emojis
Preserve punctuation
Preserve capitalization
Preserve negation

Use conservative Albanian-specific normalization rules


11. Experimental Dataset Preparation ==================================

Generate datasets for:

RAW
Minimal
Traditional
Albanian-Aware

For each strategy create:

Train dataset
Validation dataset
Test dataset

Total experimental files: 12

Verify that all experiments use the same observations and labels


12. Model Setup ==================================

Use XLM-RoBERTa-based multilingual sentiment model
Configure classification for three classes:

NEGATIVE
NEUTRAL
POSITIVE

Use the same model for all preprocessing experiments


13. Tokenization ==================================

Use the model tokenizer
Apply subword tokenization
Set maximum sequence length to 128
Apply truncation when necessary
Use dynamic padding

Keep tokenization identical across all experiments


14. Model Training ==================================

Fine-tune the model separately for:

RAW
Minimal
Traditional
Albanian-Aware

Use identical training parameters:

Epochs: 3
Learning Rate: 2e-5
Training Batch Size: 4
Evaluation Batch Size: 8
Gradient Accumulation: 4
Weight Decay: 0.01
Random Seed: 42

Evaluate on Validation after every epoch
Select the best checkpoint using Macro F1
Use GPU acceleration for training


15. Final Evaluation ==================================

Evaluate the best checkpoint on the fixed Test set

Compute:

Accuracy
Precision
Recall
F1-score
Macro F1
Weighted F1

Generate:

Classification Report
Confusion Matrix
Test Predictions

Save results separately for every preprocessing strategy


16. Preprocessing Comparison ==================================

Compare:

RAW
Minimal
Traditional
Albanian-Aware

Analyze:

Overall Accuracy
Macro F1
Weighted F1
Negative F1
Neutral F1
Positive F1
Confusion Matrices

Determine how different preprocessing levels affect model performance


17. Error Analysis ==================================

Analyze incorrectly classified comments
Identify confusion between sentiment classes
Focus especially on Neutral classification
Analyze difficult informal comments
Analyze possible ambiguity, irony and sarcasm
Investigate whether preprocessing removes useful sentiment information


18. Final Reporting ==================================

Present results for all four preprocessing strategies
Compare performance under identical experimental conditions
Discuss the effect of minimal and aggressive preprocessing
Evaluate the Albanian-Aware strategy
Answer the research questions
Discuss limitations of the dataset and experiment
Propose an appropriate preprocessing pipeline for Albanian sentiment analysis
Suggest directions for future research