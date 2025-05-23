# Streamlit App Setup Guide (Windows)

This guide explains how to create a Python environment, activate it, and run a Streamlit app (`app.py`) on a Windows PC.

## 1. Install Python
Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

Check if Python is installed:
```bash
python --version
```

## 2. Create a Virtual Environment
Open Command Prompt (CMD) and run:

```bash
python -m venv env
```
This will create a new folder named `env` containing your virtual environment.

## 3. Activate the Virtual Environment
Activate it by running:

```bash
.\env\Scripts\activate
```

You should now see `(env)` at the beginning of the command line.

## 4. Install Streamlit
After activating the environment, install Streamlit:

```bash
pip install streamlit
```

## 5. Run the Streamlit App
Finally, run your `app.py`:

```bash
streamlit run app.py
```

This will open your Streamlit app in your default web browser.

## Notes
- To deactivate the environment, just type:
```bash
deactivate
```
- Make sure `app.py` is in the same folder where you activate the environment, or provide the correct path.

---
```

Would you also like a slightly more "professional" or styled version with badges or a fancier layout? 🚀#   B a s i c _ C l i e n t s _ A p p l i c a t i o n 
 
 