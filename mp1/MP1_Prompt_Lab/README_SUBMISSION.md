# MP1 Project - Compare Different Ways to Write Instructions for AI

## What is This Project?

This project tests 4 different ways to write instructions (called "prompts") to an AI model. The goal is to see which way works best for pulling information out of job postings.

**What we're looking for:** Extract the company name, job role, and years of experience needed from job postings.

- **Input:** 10 job postings (in `data/job_snippets.jsonl`)
- **Test data:** 10 correct answers (in `data/golden_set.jsonl`) to check how well the AI did
- **Time needed:** About 5-8 hours
- **Cost:** About $0.05 to $0.20 to run the AI

---

## How to Set Up

### 1. What You Need
- Python 3.8 or newer
- An OpenAI API key (get one from openai.com)

### 2. Install Required Programs

Open your terminal and run:

```bash
pip install -r requirements.txt
```

This installs the programs we need:
- `openai` — Lets us talk to the AI
- `python-dotenv` — Helps us store secrets safely
- `pandas` — For working with data
- `jsonlines` — For reading data files

### 3. Add Your API Key

Create a new file called `.env` in the main folder and put this inside:

```
OPENAI_API_KEY=your_key_here
```

Replace `your_key_here` with your real OpenAI API key.

**Alternative:** If you use PowerShell on Windows, run:

```bash
$env:OPENAI_API_KEY = "your_key_here"
```

---

## How to Run the Project

### Best Way: Use Jupyter Notebook

Open your terminal and type:

```bash
jupyter notebook mp1_prompt_lab.ipynb
```

Then click the play button to run each section. The notebook has these parts:

1. **Setup** — Load what we need and get the API key
2. **Load Data** — Read the job postings
3. **Create Strategies** — Set up our 4 different instruction styles
4. **Run Tests** — Ask the AI 40 questions (10 jobs × 4 styles)
5. **Collect Results** — Save all the answers
6. **Grade the AI** — Check how good each style was
7. **Show Charts** — Display which style won

### Another Way: Run as Python Script

```bash
python mp1_prompt_lab.py
```

---

## Understanding Your Results

### Files You'll Get

When you run it, you'll get these files:

1. **`results/mp1_comparison.md`** — A summary of how each style did
   - Cost in dollars
   - How fast each one was
   - How accurate each one was
   - Which one performed best

2. **`results/mp1_results.jsonl`** — Detailed info for every single test

### What to Look For

#### 1. **How Much Did It Cost?**
   - Each instruction style costs a different amount
   - Simple styles are cheaper, complex ones cost more
   - Check `mp1_comparison.md` for the cost section

#### 2. **How Fast Was It?**
   - How long did each test take?
   - Simple = faster, Complex = slower
   - Check `mp1_comparison.md` for the speed section

#### 3. **How Accurate Was It?**
   - Did the AI extract the right company, role, and years?
   - We use a score called "F1" (0 = wrong, 1 = perfect)
   - Check `mp1_comparison.md` for the accuracy section

#### 4. **Which One Won?**
   - Which style was best overall?
   - It considers cost, speed, and accuracy together
   - Check `mp1_comparison.md` for the ranking

### Example Results

Let's say you got these results:

```
Best to Worst (by accuracy):
1. Long explanation      Score=0.85  Cost=$0.12  Speed=2.3 seconds
2. Structured format     Score=0.79  Cost=$0.08  Speed=1.8 seconds
3. With examples         Score=0.72  Cost=$0.06  Speed=1.5 seconds
4. Simple request        Score=0.65  Cost=$0.05  Speed=1.2 seconds
```

**What this means:**
- Long explanation gives best results but costs more and is slower
- Simple request is cheapest and fastest but less accurate
- Pick based on what matters most: speed or accuracy

---

## The 4 Instruction Styles

### 1. **Simple (No Examples)**
- Just ask the AI directly
- Fastest and cheapest
- AI guesses because it has no hints

### 2. **With Examples (Few Examples)**
- Show the AI 2-3 good examples first
- Medium speed and cost
- AI learns from examples

### 3. **Structured Format**
- Tell the AI exactly how to format the answer (JSON, specific fields)
- Medium speed and cost
- Answers are clean and easy to read

### 4. **Long Explanation (Step-by-Step)**
- Ask the AI to explain its thinking step by step
- Slowest and most expensive
- Usually most accurate - AI thinks it through

---

## Fixing Problems

### Error: Cannot Find API Key
**Fix:** 
- Make sure `.env` file is in the main folder
- Make sure it says `OPENAI_API_KEY=your_key_here`
- Restart the notebook after creating the file

### Error: Too Many API Requests
**Fix:**
- Wait a few minutes
- Check that you have money in your OpenAI account
- Try using a cheaper AI model instead

### Error: Data Files Not Found
**Fix:**
- Make sure `data/golden_set.jsonl` exists
- Make sure `data/job_snippets.jsonl` exists
- Check that the files have data in them

---

## Folder Organization

```
├── mp1_prompt_lab.ipynb       # Main notebook - run this
├── mp1_writeup.md             # What you learned
├── README.md                  # This file
├── requirements.txt           # Programs to install
├── data/
│   ├── job_snippets.jsonl     # 10 job postings
│   └── golden_set.jsonl       # Correct answers
└── results/
    ├── mp1_comparison.md      # Results and comparison
    └── mp1_results.jsonl      # Detailed test results
```

---

## Files in This Submission

| File | What It Does |
|---|---|
| `mp1_prompt_lab.ipynb` | The notebook with all the code |
| `mp1_comparison.md` | Results - which style was best |
| `mp1_writeup.md` | Your thoughts and findings |
| `requirements.txt` | List of programs to install |
| `README.md` | This file |

---

## Steps to Get Started

1. **Install programs:** Run `pip install -r requirements.txt`
2. **Add your API key:** Create `.env` file with your OpenAI key
3. **Run the notebook:** Type `jupyter notebook mp1_prompt_lab.ipynb`
4. **See results:** Open `results/mp1_comparison.md`
5. **Read findings:** Open `mp1_writeup.md`

---

## Where to Get Help

- **OpenAI Help:** https://platform.openai.com/docs/
- **AI Model Used:** gpt-4o-mini (the cheap version) and gpt-4o (the smarter version)
- **What We're Testing:** Pulling data from text
- **How We Score:** We compare to correct answers and calculate accuracy

---

*This is a learning project for Week 5*
