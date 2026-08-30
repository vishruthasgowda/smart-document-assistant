# ⚡ DocuMind AI — Smart Document Assistant

DocuMind AI is a conversational document assistant. Upload a PDF, DOCX, or TXT file and ask questions about it in natural language — the app retrieves the most relevant chunks of your document and generates accurate, context-aware answers using a large language model.

**🔗 Live app:** https://smart-document-assistant-46xngdgd9jykxnmukvacuz.streamlit.app/
**💻 Source code:** https://github.com/vishruthasgowda/smart-document-assistant

## Features

- **Secure Google Sign-In** — Firebase Authentication handles login, so only signed-in users can access the workspace.
- **Multi-format support** — Upload PDF, DOCX, or TXT files.
- **Semantic search** — Documents are split into chunks and embedded using HuggingFace's `all-MiniLM-L6-v2` model, then indexed with FAISS for fast, relevant retrieval.
- **Fast LLM responses** — Answers are generated using Groq's `openai/gpt-oss-20b` model, known for very low-latency inference.
- **Conversational chat interface** — Ask follow-up questions with context carried across the conversation.
- **Download chat history** — Export your full Q&A session as a `.txt` file.
- **User sidebar** — Shows your name, email, and profile photo once signed in.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App framework | Streamlit |
| Authentication | Firebase Authentication (Google Sign-In) |
| Text extraction | `pypdf`, `python-docx`, LangChain `TextLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | FAISS |
| LLM | Groq API (`openai/gpt-oss-20b`) |

## Running Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/vishruthasgowda/smart-document-assistant.git
   cd smart-document-assistant
   ```
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_api_key
   FIREBASE_API_KEY=your_firebase_api_key
   FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
   FIREBASE_SENDER_ID=your_sender_id
   FIREBASE_APP_ID=your_app_id
   ```
4. Run the app:
   ```bash
   python -m streamlit run app.py
   ```

## Using It on Your Phone

The app works fine on mobile browsers, but for the best visual experience, **switch your phone to Light Mode before opening the app**. The UI is designed with a light theme (white/light-gray surfaces with dark text), and viewing it while your phone is in system-wide Dark Mode can cause some text, backgrounds, or input fields to render with poor contrast or appear washed out, since some mobile browsers force-invert page colors in dark mode. Go to your phone's display settings, turn off Dark Mode (or use your browser's "Force light theme for websites" option if available), then open the app link for the cleanest, fully readable layout.

## Notes

- This is a **publicly accessible** app — anyone with a Google account can sign in and use it.
- All users share the same Groq API key, so heavy simultaneous usage may occasionally hit Groq's free-tier rate limits.
- Secrets (API keys) are managed via Streamlit Community Cloud's "Secrets" settings for the deployed version, and via a local `.env` file for local development — neither is committed to this repository.
