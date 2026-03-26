# GlucoX-challenge
Nudge Fatigue Analysis for a diabetes management app. Python pipeline that processes JSONL event logs

# Gluco-X Data Science Challenge: Nudge Effectiveness Analysis

## Context

Welcome to Gluco-X! You are the newest Data Scientist on our Product Team.

Our app helps patients manage diabetes by sending "Nudges" (push notifications) reminding them to log their glucose levels. We currently use two types of nudges:

1. **Gentle Reminder:** A soft, friendly prompt.
2. **Urgent Alert:** A more direct, medically-phrased warning.

**The Business Problem:**
Our Product Manager suspects that while "Urgent" alerts work well initially, patients might be getting "Nudge Fatigue" — ignoring them if we send too many. We need you to prove or disprove this hypothesis using raw log data.

---

## AI & LLM Policy

**We encourage the use of AI tools (ChatGPT, Claude, Copilot, etc.) to accelerate your work.** In the real world, we use them too. However, we value **transparency and understanding** over blind copy-pasting.

- **If you use AI:** Please add a short "AI Disclosure" section to your report or comments in your code explaining how you used it (e.g., *"Used ChatGPT to generate the regex for parsing the JSON logs"* or *"Used GitHub Copilot for boilerplate plotting code"*).
- **The Trap:** AI often hallucinates convenient functions that don't exist or writes inefficient loops. You are responsible for the correctness and performance of every line of code submitted.

---

## The Data

You will find two files in the `data/` directory:

### 1. `patient_registry.csv`

| Column         | Description                            |
|----------------|----------------------------------------|
| `patient_id`   | Unique patient identifier (e.g., `P_001`) |
| `age_group`    | Demographic bucket                     |
| `risk_segment` | Clinical risk score (Low / Medium / High) |

### 2. `app_logs.jsonl` (JSON Lines)

A raw stream of app events. Each line is a separate JSON object.

| Field          | Description                                              |
|----------------|----------------------------------------------------------|
| `event_id`     | Unique event identifier                                  |
| `patient_id`   | The patient associated with the event                    |
| `timestamp`    | UTC datetime string                                      |
| `event_type`   | Either `nudge_sent` or `measurement_logged`              |
| `payload`      | Metadata — contains `nudge_type` or `glucose_value`      |

---

## The Assignment

### Task 1: The "Response Time" Pipeline

We need to know how long it takes a patient to react to a nudge. Write a reusable pipeline as a Python script (`.py` file) using a class or module that:

1. **Parses** the JSON logs.
2. **Attributes** measurements to nudges using the following rules:
   - A measurement counts as a "response" if it occurs **within 4 hours** of a nudge.
   - If a patient receives multiple nudges before logging, credit the measurement to the **most recent** nudge only.

### Task 2: The Fatigue Analysis

Visualize the relationship between "Nudge Count" and "Response Rate."

- Does the probability of a patient responding drop after they have received their 10th or 20th nudge?

### Task 3: The Executive Summary

Submit a brief report (PDF) with actionable findings for the Product Manager. Use the data to tell a clear story about nudge effectiveness, and recommend what the team should do next.

---

## Deliverables

Please submit a zipped folder or GitHub link containing:

1. **Code:** A clean, well-commented Python script (`.py` file).
2. **Report:** Your findings and visualizations.
3. **Requirements:** A `requirements.txt` file listing your dependencies.

### Evaluation Criteria

| Criterion        | What We're Looking For                                      |
|------------------|-------------------------------------------------------------|
| **Correctness**  | Did you handle the attribution logic correctly?             |
| **Code Quality** | Is the code modular, readable, and robust?                  |
| **Transparency** | Did you clearly document your sources and AI usage?         |




