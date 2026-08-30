import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from docx import Document

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI - Smart Document Assistant", page_icon="⚡", layout="wide"
)

# ============================================================
# FIREBASE CONFIG — fill these in with your Firebase project's
# web app config (Firebase Console > Project Settings > General
# > Your apps > SDK setup and configuration).
# Also add "localhost" (and your deployed domain) under
# Authentication > Settings > Authorized domains.
# ============================================================
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyA3Km2LFfOyFq8WkELLThuEPJlJ8totO0Y"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "documind-ai-d7eef.firebaseapp.com"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "documind-ai-d7eef"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "documind-ai-d7eef.firebasestorage.app"),
    "messagingSenderId": os.getenv("FIREBASE_SENDER_ID", "957191349192"),
    "appId": os.getenv("FIREBASE_APP_ID", "1:957191349192:web:9981e92126b7f83f76d6fd"),
}

# --- Professional High-Contrast Light Theme Design System ---
st.markdown("""
    <style>
    :root {
        --background: #F8FAFC;
        --surface: #FFFFFF;
        --surface-hover: #F1F5F9;
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --border: #CBD5E1;
        --primary: #4F46E5;
        --primary-hover: #4338CA;
    }

    /* Global Background & Primary Text */
    .stApp {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }

    /* Navbar Logo */
    .logo-container {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        padding-top: 8px;
    }

    /* Hero Typography */
    .hero-title {
        font-size: 3.8rem;
        font-weight: 900;
        color: var(--text-primary);
        line-height: 1.1;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }
    .hero-desc {
        color: var(--text-secondary);
        font-size: 1.15rem;
        line-height: 1.6;
        max-width: 550px;
        margin-bottom: 2.5rem;
    }

    /* Surface Card */
    .glowing-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 100%);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }

    /* Buttons Styling */
    div.stButton > button {
        background: var(--primary) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: var(--primary-hover) !important;
        color: #FFFFFF !important;
    }

    /* Form Text Inputs & Labels Visibility */
    label, .stTextInput label, .stChatInput label, .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    div.stTextInput > div > div > input {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    div.stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    /* Chat Input Box & Typed Text Visibility */
    .stChatInputContainer {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stChatInput textarea {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }
    .stChatInput textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* File Uploader Container Surface */
    .uploadedFile, div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar user card */
    .user-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .user-card img {
        border-radius: 50%;
        width: 64px;
        height: 64px;
        margin-bottom: 0.6rem;
        border: 2px solid var(--primary);
    }
    .user-card .name {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 1.05rem;
    }
    .user-card .email {
        color: var(--text-muted);
        font-size: 0.85rem;
        word-break: break-all;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_photo" not in st.session_state:
    st.session_state.user_photo = ""
if "page" not in st.session_state:
    st.session_state.page = "home"


def check_firebase_redirect_login():
    """
    Reads the query params that our Firebase JS snippet appends to the URL
    after a successful Google sign-in, and turns them into a real
    authenticated session. This is what makes the Firebase login actually
    work in Streamlit (which can't read JS/popup results directly).
    """
    params = st.query_params
    if "fb_email" in params and not st.session_state.authenticated:
        st.session_state.authenticated = True
        st.session_state.user_email = params.get("fb_email", "")
        st.session_state.user_name = params.get("fb_name", "") or params.get("fb_email", "").split("@")[0]
        st.session_state.user_photo = params.get("fb_photo", "")
        st.session_state.page = "workspace"
        st.query_params.clear()
        st.rerun()


def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    temp_file_path = f"temp_{uploaded_file.name}"

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = ""
    try:
        if file_extension == "pdf":
            reader = PdfReader(temp_file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
        elif file_extension == "docx":
            doc = Document(temp_file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return text


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- Navbar Component ---
def render_navbar(unique_prefix=""):
    col_logo, col_space, c1, c2, c3 = st.columns([2, 2, 1, 1, 1])
    with col_logo:
        st.markdown("<p class='logo-container'>⚡ DOCUMIND AI</p>", unsafe_allow_html=True)
    with c1:
        if st.button("HOME", key=f"{unique_prefix}_home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with c2:
        if st.button("ABOUT US", key=f"{unique_prefix}_about", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()
    with c3:
        if st.button("LOGIN", key=f"{unique_prefix}_login", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()
    st.markdown("<hr style='border: 0.5px solid var(--border); margin-top: 5px; margin-bottom: 2rem;'>", unsafe_allow_html=True)


# --- Hero Landing Page ---
def show_landing_page():
    render_navbar(unique_prefix="nav_home")

    col_hero, col_visual = st.columns([1.3, 1], gap="large")

    with col_hero:
        st.markdown("<h1 class='hero-title'>ARTIFICIAL<br>INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-desc'>Upload your technical reports, research papers, or enterprise contracts. Engage in fluid conversational threads powered by high-speed vector retrieval and advanced language models.</p>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("GET STARTED", key="hero_get_started", use_container_width=True):
                st.session_state.page = "auth"
                st.rerun()
        with btn_col2:
            if st.button("EXPLORE DOCS", key="hero_explore_btn", use_container_width=True):
                st.session_state.page = "about"
                st.rerun()

    with col_visual:
        st.markdown("""
            <div class='glowing-box'>
                <h2 style='color: var(--text-primary); margin-bottom: 1rem;'>✨ Next-Gen AI Engine</h2>
                <p style='color: var(--text-secondary);'>Experience zero-latency chunk searching paired with secure cloud memory workspaces.</p>
            </div>
        """, unsafe_allow_html=True)


# --- About Us Page ---
def show_about_page():
    render_navbar(unique_prefix="nav_about")
    st.markdown("## About DocuMind AI")
    st.write("DocuMind AI is an enterprise-grade semantic parsing engine designed to extract, index, and reason across large volumes of textual files securely.")
    st.markdown("### Core Capabilities")
    st.markdown("* **High-Speed Embeddings:** Powered by HuggingFace and FAISS vector structures.")
    st.markdown("* **Secure Auth Workspaces:** Isolated memory mapping for private user sessions.")
    st.markdown("* **Instant Querying:** Real-time conversational context retrieval.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="about_back_btn"):
        st.session_state.page = "home"
        st.rerun()


# --- Authentication Screen with real Firebase Google Auth ---
def show_auth_page():
    render_navbar(unique_prefix="nav_auth")
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("## Secure Portal Access")
        st.caption("Sign in with Google via Firebase or your account credentials.")

        # NOTE: st.components.v1.html renders this inside its own embedded
        # iframe. Firebase's "unauthorized-domain" check looks at the exact
        # browser location of the page running the auth code — inside that
        # iframe it doesn't reliably see "localhost" even though your real
        # browser tab is on localhost. To fix this, the button below injects
        # the actual sign-in script into the TOP-LEVEL page (window.top)
        # instead of running it inside the iframe, so Firebase sees the
        # real address bar URL.
        firebase_google_html = f"""
        <div id="firebase-auth-container" style="text-align: center; margin-bottom: 15px;">
          <script>
            function runFirebaseSignIn() {{
              var statusEl = document.getElementById("fb-status");
              statusEl.innerText = "Opening Google sign-in...";

              var moduleCode = `
                import {{ initializeApp, getApps, getApp }} from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
                import {{ getAuth, GoogleAuthProvider, signInWithPopup }} from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";

                const firebaseConfig = {{
                  apiKey: "{FIREBASE_CONFIG['apiKey']}",
                  authDomain: "{FIREBASE_CONFIG['authDomain']}",
                  projectId: "{FIREBASE_CONFIG['projectId']}",
                  storageBucket: "{FIREBASE_CONFIG['storageBucket']}",
                  messagingSenderId: "{FIREBASE_CONFIG['messagingSenderId']}",
                  appId: "{FIREBASE_CONFIG['appId']}"
                }};

                const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
                const auth = getAuth(app);
                const provider = new GoogleAuthProvider();

                signInWithPopup(auth, provider)
                  .then((result) => {{
                    const user = result.user;
                    const topUrl = window.location.href.split("?")[0];
                    const qs = new URLSearchParams({{
                      fb_email: user.email || "",
                      fb_name: user.displayName || "",
                      fb_photo: user.photoURL || ""
                    }});
                    window.location.href = topUrl + "?" + qs.toString();
                  }})
                  .catch((error) => {{
                    window.__fbSignInError = error.message;
                    alert("Sign-in failed: " + error.message);
                  }});
              `;

              // Inject and run this code in the TOP-LEVEL page (not this
              // iframe), so Firebase sees the real localhost address.
              var s = window.top.document.createElement("script");
              s.type = "module";
              s.textContent = moduleCode;
              window.top.document.head.appendChild(s);
            }}
          </script>
          <button onclick="runFirebaseSignIn()" style="width: 100%; background-color: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; padding: 10px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 10px;">
            <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" width="18px"> Continue with Google
          </button>
          <p id="fb-status" style="color: #64748B; font-size: 0.85rem; margin-top: 8px;"></p>
        </div>
        """
        st.components.v1.html(firebase_google_html, height=110)

        st.markdown("<p style='text-align: center; color: var(--text-secondary); margin: 1rem 0;'>OR EMAIL LOGIN</p>", unsafe_allow_html=True)

        email = st.text_input("Email Address", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_pass")

        if st.button("Sign In to Workspace", key="auth_submit_btn", use_container_width=True):
            if email and len(password) >= 6:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = email.split("@")[0]
                st.session_state.user_photo = ""
                st.session_state.page = "workspace"
                st.rerun()
            else:
                st.error("Please enter a valid email and a password of at least 6 characters.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Return Home", key="auth_home_btn"):
            st.session_state.page = "home"
            st.rerun()


# --- Sidebar with logged-in user details ---
def render_user_sidebar():
    with st.sidebar:
        photo = st.session_state.user_photo or "https://www.gravatar.com/avatar/?d=mp&s=128"
        name = st.session_state.user_name or "User"
        email = st.session_state.user_email or ""
        st.markdown(
            f"""
            <div class="user-card">
                <img src="{photo}" />
                <div class="name">{name}</div>
                <div class="email">{email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Sign Out", key="sidebar_signout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_email = ""
            st.session_state.user_name = ""
            st.session_state.user_photo = ""
            st.session_state.page = "home"
            st.rerun()
        st.markdown("---")


# --- Main Workspace ---
def show_workspace():
    render_user_sidebar()

    st.markdown("## 📂 Smart Document Workspace")
    st.caption(f"Authenticated Workspace | User: **{st.session_state.user_email}**")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="doc_uploader")

    if uploaded_file is not None:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.status("Processing document into vector memory...", expanded=True) as status:
                st.write("Extracting file text payload...")
                raw_text = extract_text_from_file(uploaded_file)

                if not raw_text.strip():
                    st.error("Could not extract text from this file.")
                    return

                st.write("Splitting text chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(raw_text)

                st.write("Generating local embeddings & vector index...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)

                # Groq's llama-3.3-70b-versatile (and llama-3.1-8b-instant) were
                # deprecated in June 2026. openai/gpt-oss-20b / openai/gpt-oss-120b
                # are the current recommended replacements.
                llm = ChatGroq(
                    temperature=0.2,
                    model_name="openai/gpt-oss-20b"
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.session_state.vector_store = vector_store
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []

                status.update(label="Document indexed successfully!", state="complete", expanded=False)

    if "vector_store" in st.session_state:
        header_col, download_col = st.columns([4, 1])
        with header_col:
            st.markdown("### Ask Questions About Your Document")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with download_col:
            if st.session_state.chat_history:
                chat_text_lines = [
                    f"DocuMind AI - Chat Export",
                    f"Document: {st.session_state.get('current_file', 'N/A')}",
                    f"User: {st.session_state.user_email}",
                    "=" * 40,
                    "",
                ]
                for role, msg in st.session_state.chat_history:
                    chat_text_lines.append(f"{role}: {msg}")
                    chat_text_lines.append("")
                chat_text = "\n".join(chat_text_lines)

                st.download_button(
                    label="⬇ Download Chat",
                    data=chat_text,
                    file_name=f"chat_{st.session_state.get('current_file', 'export')}.txt",
                    mime="text/plain",
                    key="download_chat_btn",
                    use_container_width=True,
                )

        user_query = st.chat_input("Type your question here...", key="chat_input_box")

        if user_query:
            with st.spinner("Synthesizing answer..."):
                retrieved_docs = st.session_state.retriever.invoke(user_query)
                context_text = format_docs(retrieved_docs)

                history_str = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])

                formatted_prompt = f"""Answer the question accurately based only on the following context retrieved from the document:
{context_text}

Chat History:
{history_str}

Question: {user_query}
"""
                answer = st.session_state.llm.invoke(formatted_prompt).content

                st.session_state.chat_history.append(("User", user_query))
                st.session_state.chat_history.append(("Assistant", answer))

        for role, msg in st.session_state.chat_history:
            with st.chat_message("user" if role == "User" else "assistant"):
                st.write(msg)
    else:
        st.info("👆 Please upload a document above to unlock the chat interface.")


def main():
    check_firebase_redirect_login()

    if not st.session_state.authenticated:
        if st.session_state.page == "home":
            show_landing_page()
        elif st.session_state.page == "about":
            show_about_page()
        elif st.session_state.page == "auth":
            show_auth_page()
    else:
        show_workspace()


if __name__ == "__main__":
    main()