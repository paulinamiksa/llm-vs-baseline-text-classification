# LLM vs classical baseline for banking text classification

Is a large language model worth using for a simple text classification task, compared with a classical machine learning baseline?

This project compares a TF-IDF + logistic regression model with a zero-shot LLM on the same set of banking customer messages, measuring quality, reliability of the output format, speed and token usage.

## Key results

| Method | Accuracy | Invalid answers | Avg time per text |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.94 | 0 | < 0.01 s |
| LLM zero-shot (`openai/gpt-oss-120b` via Groq, low reasoning effort) | 0.93 | 0 | 0.52 s |

Both methods were evaluated on the same fixed sample of 200 test messages. The LLM run used 58,080 tokens in total (about 290 tokens per message).

With n = 200, the 95% confidence interval for accuracy is about ±3 percentage points, so the 1 pp difference between the two methods is **not significant**.

## Data

- **Dataset:** [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) by PolyAI: short customer messages to a bank, each labelled with one of 77 intents.
- **Subset:** 10 intents, to keep the task focused and the LLM prompt short.
- **Train / test:** the baseline is trained on the training split; both methods are evaluated on the same fixed sample of 200 test messages (`data/test_sample_200.csv`), never used for training or prompt design.

## Methods

### 1. Baseline: TF-IDF + logistic regression

- Text is turned into TF-IDF features (word weights based on frequency in a message vs. across all messages).
- A logistic regression classifier is trained on the training split with scikit-learn.
- Accuracy on the full test set is about 0.93, and 0.94 on the 200-message sample.

### 2. LLM zero-shot classification

- Model: `openai/gpt-oss-120b`, called through the Groq API (free tier).
- The prompt lists the 10 category names and asks for exactly one category in JSON format: `{"label": "<category>"}`. No training examples are included (zero-shot).
- Settings: `temperature=0` for reproducible answers, JSON mode to enforce the output format, low reasoning effort to reduce tokens and latency.
- **Validation:** an answer that is not valid JSON or not one of the 10 categories is counted as `INVALID` and treated as an error, not skipped.
- For every message the label, response time and token usage are saved (`data/llm_zero_shot_results.csv`).

## Findings

- **No quality gain from the LLM.** The zero-shot LLM matched, but did not beat, the simple baseline. The baseline is also much faster, runs locally at no cost and is fully explainable.
- **The LLM's errors are systematic, not random.** Most of its mistakes confuse two similar intents:
  - `balance_not_updated_after_bank_transfer`: money has not appeared in the customer's own account;
  - `transfer_not_received_by_recipient`: the person the customer paid has not received the money.

  With only the label names in the prompt, the model cannot see the subtle difference that the baseline learned from labelled examples. Some messages, such as "My transfer is pending", are genuinely ambiguous.
- **The output format was reliable.** JSON mode plus validation gave 0 invalid answers out of 200.
- **Takeaway:** for a narrow task with enough labelled data, a classical model is a strong and cheap default. An LLM becomes more attractive when labelled data is scarce, categories change often, or the task needs language understanding beyond keywords. Either way, it should be compared against a baseline on the same held-out data before it is used.

## Prompt experiment: category descriptions

**Hypothesis:** the LLM confuses `balance_not_updated_after_bank_transfer` and `transfer_not_received_by_recipient` because it only sees the label names. I added a one-sentence description of every category to the prompt. The descriptions were written from the label definitions, not copied from test messages.

| Variant | Accuracy | Errors | Transfer-pair errors | Invalid | Avg tokens per text | Avg time per text |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 0.930 | 14 | 9 | 0 | 290 | 0.52 s |
| With category descriptions | 0.935 | 13 | 5 | 0 | 455 | 0.87 s |

- **The targeted confusion almost halved** (9 → 5 errors).
- **But new errors appeared elsewhere:** the descriptions fixed 4 messages and broke 3, so overall accuracy barely moved.
- **Not significant:** McNemar's exact test on the 7 messages where the two variants disagree gives p = 1.0.
- **Cost went up:** +57% tokens and +67% latency per message.

**Takeaway:** a prompt change aimed at one error type can shift errors to other classes. Evaluating only the problem being fixed would have overstated the improvement, which is why every variant is measured on the full test set.

**Caveat:** the descriptions were designed after looking at errors on this test sample, so the result should be confirmed on a fresh sample.

## Limitations

- Small test sample (200 messages), so only large differences are conclusive.
- 10 of the 77 intents, English only.
- One LLM; results may differ with other models.
- Prompt variants were designed after error analysis on the same sample (see caveat above).
- The free-tier API may change models or limits over time.

## Next steps

- Few-shot prompting: add 1–2 training examples per category and compare with the variants above.
- Confirm the prompt results on a fresh sample of test messages.
- Compare a smaller model (`openai/gpt-oss-20b`) and a model running locally.
- Estimate the cost of each approach at scale (for example, 1 million messages per month) from the measured token usage.

## Project structure

```
├── data/
│   ├── test_sample_200.csv            # fixed evaluation sample
│   ├── llm_zero_shot_results.csv      # zero-shot LLM predictions, latency and tokens
│   └── llm_descriptions_results.csv   # prompt with category descriptions
├── notebooks/
│   ├── 01_baseline.ipynb              # TF-IDF + logistic regression
│   ├── 02_llm_zero_shot.ipynb         # zero-shot LLM classification and evaluation
│   └── 03_prompt_variants.ipynb       # prompt experiments and comparison
├── requirements.txt
└── README.md
```

## How to run

1. Create and activate a virtual environment, then install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create a free API key at [console.groq.com](https://console.groq.com) and put it in a `.env` file in the project root (this file is excluded from Git):
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run the notebooks in order: `01_baseline.ipynb`, `02_llm_zero_shot.ipynb`, then `03_prompt_variants.ipynb`.

## Tools

Python, pandas, scikit-learn, Jupyter, Groq API, Git.
