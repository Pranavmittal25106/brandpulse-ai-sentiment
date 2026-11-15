# Getting a Live, Shareable Link for the Dashboard

**Important — please read first:** I built and tested the full working app
(`app.py`) in this sandbox, but this sandbox cannot expose a public URL — it
has no open ports and no access to hosting/deployment services. So I can't
hand you a real "live link" myself; anything I typed here would be a fake
placeholder, which wouldn't help your submission. What I *can* do is get you
a real one in about 5 minutes with almost no extra work, using a free host.

## Option 1 — Streamlit Community Cloud (recommended, free, ~5 min)

1. Create a free GitHub account if you don't have one, and create a new
   repository (e.g. `brandpulse-ai-sentiment`).
2. Upload these files/folders to the repo, keeping the same structure:
   `app.py`, `requirements.txt`, `data/twitter_sentiment_data.csv`, `models/`.
3. Go to **share.streamlit.io** → sign in with GitHub → **New app**.
4. Select your repo, branch `main`, and set the main file path to `app.py`.
5. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app.
6. You'll get a public URL like `https://your-app-name.streamlit.app` —
   that's your live link for the submission form.

## Option 2 — Hugging Face Spaces (also free)

1. Create a Space at **huggingface.co/new-space**, choose the **Streamlit**
   SDK.
2. Upload the same files (`app.py`, `requirements.txt`, `data/`, `models/`).
3. The Space builds automatically and gives you a URL like
   `https://huggingface.co/spaces/your-username/brandpulse-ai`.

## Option 3 — Run it locally and share via a tunnel

If you just need something to demo right now on your own machine:
```
pip install -r requirements.txt
streamlit run app.py
```
Then, in a second terminal, expose it temporarily with a tool like `ngrok`
(`ngrok http 8501`) to get a short-lived public URL. This is fine for a quick
demo but not a permanent link — use Option 1 or 2 for the actual submission.

## What to paste into the submission form

Use the `https://...streamlit.app` (or Hugging Face Spaces) URL from Option 1
or 2 as your "Live Link of the working project."
