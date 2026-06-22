# NYT Headlines Analysis - NLP Project

**New York Times headlines analysis using Natural Language Processing (NLP)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

---

## Table of Contents
- [Overview](#overview)
- [Context](#context)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Next Steps](#next-steps)
- [License](#license)

---

## Overview

This project analyzes New York Times headlines from January to July 2020 using NLP to identify the most frequent words and how their relevance shifted over time in response to global events.

The goal is to demonstrate how textual data can be transformed into actionable insights about media coverage during periods of crisis and social transformation.

---

## Context

The analyzed period (Jan-Jul 2020) was marked by:

- **March:** WHO declares COVID-19 a pandemic
- **May:** Death of George Floyd and global protests
- **June-July:** Intensification of the US presidential campaign

Why the NYT?
- Relevance and credibility of the outlet
- Intensive pandemic coverage (NYC as one of the most affected cities)
- Accessible and well-documented API

---

## Methodology

### 1. Data Collection
- Used the **NYT Article Search API**
- Period: January 1st to July 31st, 2020
- Extracted fields: `headline`, `pub_date`, `section`

### 2. Pre-processing (NLP with NLTK)
```python
# Cleaning pipeline
1. Tokenization (word_tokenize)
2. Punctuation removal (string.punctuation)
3. Stopword removal (nltk.corpus.stopwords)
4. Filter words with < 3 characters
