# Aspect-Based Sentiment Analysis (ABSA)

## Overview
Aspect-Based Sentiment Analysis (ABSA) is a fine-grained sentiment analysis task that identifies sentiment toward specific aspects (targets) within text, rather than assigning a single sentiment label to the whole sentence.

**Example:**
> "The battery is great but the screen is bad."

- battery → Positive  
- screen → Negative  

---

## Why ABSA?
Traditional sentiment analysis provides a single sentiment for the entire text.  
ABSA provides:
- Aspect-level understanding
- More detailed sentiment insights
- Better explainability

---

## Core Components
- **Aspect Term**: The target feature (e.g., battery, screen)
- **Opinion Term**: Words expressing sentiment (e.g., good, bad)
- **Sentiment Polarity**: Positive / Negative / Neutral
- **Context**: Sentence or review containing aspects

---

## Main Tasks in ABSA

### 1. Aspect Term Extraction (ATE)
Identify aspect terms in text.

**Input:**
> "The camera quality is amazing."

**Output:**
- camera quality

---

### 2. Aspect Sentiment Classification (ASC)
Determine sentiment for a given aspect.

**Example:**
- camera quality → Positive

---

### 3. Aspect Term Extraction + Sentiment Classification (AESC)
Jointly extract aspects and classify sentiment.

**Example:**
> "The battery is terrible."
- battery → Negative

---

### 4. Aspect Category Detection (ACD)
Map aspects into predefined categories.

**Example:**
- battery life → Performance
- screen → Display

---

### 5. Aspect Category Sentiment Classification (ACSC)
Classify sentiment at category level.

**Example:**
- Display → Negative

---

### 6. Opinion Term Extraction
Extract sentiment-bearing words.

**Example:**
- great, bad, slow

---

### 7. End-to-End ABSA
Full pipeline that performs:
- Aspect extraction
- Opinion detection
- Sentiment classification
