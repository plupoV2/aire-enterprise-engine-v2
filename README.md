# AIRE — AI Real Estate Underwriting (Streamlit) v5

## What’s new vs v4
- Workspace settings (custom pipeline folders + scoring profile)
- Deal collaboration: notes + tags + assignee
- Webhook integrations (optional) for CRM/BI pipelines
- Memo gallery for share links + branded PDFs
- Excel export bundle: Pipeline + Notes + Versions + Audit + Calibration + Settings
- **Robust IRR solver** (fixes OverflowError in Streamlit deploy)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
- Set main file to `app.py`.

## Optional secrets
- `RESO_BASE_URL`, `RESO_BEARER_TOKEN` (for RESO/MLS feed if you have access)

## Cleaner UI
This build swaps the top tabs for a simple sidebar navigation and a cleaner chat-first layout.


## Threads UI
Pipelines now behave like chat history. Each saved deal is a thread; open a deal to continue a conversation and version changes.


## Action chips
Assistant shows adaptive one-click action chips under the last message (ranked by flags + quick sensitivity).
