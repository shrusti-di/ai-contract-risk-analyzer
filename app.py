import json
import os
from fpdf import FPDF
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "openai/gpt-oss-120b"
MAX_CHARS = 24000

st.set_page_config(page_title="ClauseCheck", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
/* ---------- Page background ---------- */
.stApp {
    background: radial-gradient(circle at 20% 0%, #1b1f4b 0%, #0b1020 45%, #070b17 100%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1100px; }

/* ---------- Base text ---------- */
html, body, .stMarkdown p, .stMarkdown li, label, .stTextInput input {
    font-size: 20px !important;
    color: #e5e7eb;
}

/* ---------- Hero banner ---------- */
.hero {
    background: linear-gradient(120deg, #312e81 0%, #6d28d9 55%, #be185d 100%);
    padding: 48px 40px;
    border-radius: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 0 50px rgba(139, 92, 246, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.12);
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 64px !important;
    font-weight: 800 !important;
    color: white !important;
    margin: 0;
    padding: 0;
}
.hero .tagline { font-size: 24px; margin-top: 10px; opacity: 0.95; }
.hero .badges { margin-top: 18px; }
.hero .badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 6px 16px;
    border-radius: 999px;
    margin: 4px;
    font-size: 17px;
}
.hero .dev { margin-top: 18px; font-size: 18px; opacity: 0.9; }

/* ---------- Upload box ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #151b2e;
    border: 2px dashed #8b5cf6;
    border-radius: 18px;
    padding: 28px;
}
[data-testid="stFileUploaderDropzone"] * { color: #e5e7eb !important; }

/* ---------- Text input ---------- */
.stTextInput input {
    background: #151b2e !important;
    border: 1px solid #3b4270 !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] p {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #9ca3af;
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #a78bfa; }

/* ---------- Buttons ---------- */
.stButton button, .stDownloadButton button {
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.8rem !important;
    border-radius: 12px !important;
    border: none !important;
    color: white !important;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899) !important;
    box-shadow: 0 0 22px rgba(168, 85, 247, 0.45);
    transition: transform 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-2px);
    color: white !important;
}
.stButton button p, .stDownloadButton button p { color: white !important; }

/* ---------- Headings ---------- */
h2, h3 { font-size: 34px !important; color: #c4b5fd !important; font-weight: 800 !important; }

/* ---------- Risk score card ---------- */
[data-testid="stMetric"] {
    background: #151b2e;
    border-radius: 18px;
    padding: 20px;
    border-left: 8px solid #8b5cf6;
    box-shadow: 0 0 24px rgba(139, 92, 246, 0.2);
}
[data-testid="stMetricValue"] { font-size: 56px !important; color: #a78bfa; }
[data-testid="stMetricLabel"] p { font-size: 20px !important; color: #9ca3af; }

/* ---------- Clause cards ---------- */
[data-testid="stExpander"] {
    background: #151b2e;
    border-radius: 16px;
    border: 1px solid #2d3463;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.12);
    margin-bottom: 12px;
}
[data-testid="stExpander"] summary p {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #f3f4f6;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    margin-top: 50px;
    padding: 18px;
    color: #9ca3af;
    font-size: 17px;
}
/* ---------- BIGGER TEXT OVERRIDES ---------- */

/* Tagline: "Understand any contract before you sign it" */
.hero .tagline {
    font-size: 36px !important;
    font-weight: 600 !important;
    margin-top: 14px !important;
}

/* Badges: Risk score, Risky clauses, Ask questions, PDF report */
.hero .badge {
    font-size: 24px !important;
    padding: 10px 24px !important;
    margin: 6px !important;
}

/* "Developed by Shrusti Diggavi" */
.hero .dev {
    font-size: 26px !important;
    font-weight: 600 !important;
    margin-top: 24px !important;
}

/* Caption: "AI contract risk analyzer. Not legal advice..." */
[data-testid="stCaptionContainer"] p {
    font-size: 24px !important;
    color: #cbd5e1 !important;
}

/* "Upload a contract (PDF)" label */
[data-testid="stFileUploader"] label p,
[data-testid="stWidgetLabel"] p {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #e5e7eb !important;
}

/* Uploaded file name and size (AMD_Emp...pdf, 1.5MB) */
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    font-size: 24px !important;
}

/* Tabs: Risk Report, Ask Questions */
button[data-baseweb="tab"] p {
    font-size: 28px !important;
    font-weight: 800 !important;
}
button[data-baseweb="tab"] {
    padding: 14px 26px !important;
}
/* ---------- Highlighted notice ---------- */
.notice {
    background: linear-gradient(90deg, rgba(250, 204, 21, 0.18), rgba(236, 72, 153, 0.18));
    border: 2px solid #facc15;
    border-left: 10px solid #facc15;
    border-radius: 14px;
    padding: 18px 24px;
    margin: 10px 0 26px 0;
    font-size: 26px;
    line-height: 1.5;
    color: #fef9c3;
    box-shadow: 0 0 24px rgba(250, 204, 21, 0.25);
}
.notice b {
    color: #fde047;
    font-size: 30px;
}
/* ---------- My name in color ---------- */
.myname {
    font-weight: 800 !important;
    background: linear-gradient(90deg, #fde047, #fb923c, #f472b6);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: #fde047;
}
.hero .dev .myname {
    font-size: 32px;
}
/* ---------- BIGGER FOOTER ---------- */
.footer {
    font-size: 28px !important;
    font-weight: 600 !important;
    margin-top: 60px !important;
    padding: 28px !important;
    color: #cbd5e1 !important;
    border-top: 1px solid rgba(255, 255, 255, 0.15);
}
.footer b {
    font-size: 28px !important;
}
.footer .myname {
    font-size: 34px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to Streamlit secrets or env variables.")
    st.stop()

client = Groq(api_key=api_key)


# ---------- Helpers ----------
def extract_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text: str, size: int = 800, overlap: int = 150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def retrieve(question: str, chunks, k: int = 4):
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(chunks + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    top = scores.argsort()[::-1][:k]
    return [(int(i), chunks[i]) for i in top]

def make_pdf(report: dict) -> bytes:
    def clean(s):
        # built-in PDF fonts only support basic characters
        return str(s).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def write(text, style="", size=11, gap=2):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 6, clean(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    write("ClauseCheck Report", "B", 20, 4)
    write(f"Document type: {report.get('document_type', '')}")
    write(f"Risk score: {report.get('risk_score', '?')}/100", "B", 13, 4)
    write(f"Summary: {report.get('summary', '')}", gap=6)

    write("Risky clauses", "B", 14, 3)
    for i, item in enumerate(report.get("risky_clauses", []), 1):
        write(f"{i}. [{str(item.get('severity', '')).upper()}] {item.get('clause', '')}", "B")
        write(f"Why it's risky: {item.get('why_risky', '')}")
        write(f"Suggested fix: {item.get('suggested_fix', '')}", gap=5)

    if report.get("missing_protections"):
        write("Missing protections", "B", 14, 3)
        for m in report["missing_protections"]:
            write(f"- {m}")

    write("Not legal advice. Consult a lawyer for important decisions.", "I", 9)
    return bytes(pdf.output())

def analyze_contract(text: str) -> dict:
    system = (
        "You are a careful legal-document assistant for ordinary people. "
        "You are NOT a lawyer. Respond ONLY with valid JSON in this schema:\n"
        '{"document_type": str, "summary": str, "risk_score": int (0-100, higher = riskier), '
        '"risky_clauses": [{"clause": str (short quote under 25 words), '
        '"severity": "low"|"medium"|"high", "why_risky": str, "suggested_fix": str}], '
        '"missing_protections": [str]}'
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Analyze this contract:\n\n{text[:MAX_CHARS]}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def answer_question(question: str, chunks) -> str:
    hits = retrieve(question, chunks)
    context = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in hits)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY using the provided contract excerpts. "
                    "Cite chunk numbers like [Chunk 3]. "
                    "If the answer is not in the excerpts, say you could not find it."
                ),
            },
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ---------- UI ----------
st.markdown(
    """
<div class="hero">
<h1>📄 ClauseCheck</h1>
<div class="tagline">Understand any contract before you sign it</div>
<div class="badges">
<span class="badge">🔍 Risk score</span>
<span class="badge">⚠️ Risky clauses</span>
<span class="badge">💬 Ask questions</span>
<span class="badge">📥 PDF report</span>
</div>
<div class="dev">Developed by <span class="myname">Shrusti Diggavi</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice">⚖️ <b>AI contract risk analyzer.</b> Not legal advice; consult a lawyer for important decisions.</div>',
    unsafe_allow_html=True,
)

file = st.file_uploader("Upload a contract (PDF)", type="pdf")

if file:
    if st.session_state.get("file_name") != file.name:
        with st.spinner("Reading document..."):
            text = extract_text(file)
        if len(text.strip()) < 100:
            st.error("Couldn't extract text. This may be a scanned PDF.")
            st.stop()
        st.session_state.update(
            file_name=file.name, text=text, chunks=chunk_text(text), report=None
        )

    tab1, tab2 = st.tabs(["🔍 Risk Report", "💬 Ask Questions"])

    with tab1:
        if st.button("Analyze contract", type="primary"):
            with st.spinner("Analyzing clauses..."):
                try:
                    st.session_state.report = analyze_contract(st.session_state.text)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        report = st.session_state.get("report")
        if report:
            c1, c2 = st.columns([1, 3])
            c1.metric("Risk score", f"{report.get('risk_score', '?')}/100")
            c2.write(f"**{report.get('document_type', '')}**: {report.get('summary', '')}")

            icons = {"high": "🔴", "medium": "🟠", "low": "🟡"}
            st.subheader("Risky clauses")
            for item in report.get("risky_clauses", []):
                sev = item.get("severity", "low")
                with st.expander(f"{icons.get(sev, '⚪')} {sev.upper()}: {item.get('clause', '')[:80]}"):
                    st.write(f"**Why it's risky:** {item.get('why_risky')}")
                    st.write(f"**Suggested fix:** {item.get('suggested_fix')}")

            if report.get("missing_protections"):
                st.subheader("Missing protections")
                for m in report["missing_protections"]:
                    st.write(f"- {m}")

            st.download_button(
                "Download report (PDF)",
                make_pdf(report),
                file_name="clausecheck_report.pdf",
                mime="application/pdf",
            )

    with tab2:
        q = st.text_input("Ask something about this contract")
        if q:
            with st.spinner("Searching document..."):
                st.write(answer_question(q, st.session_state.chunks))

st.markdown(
    '<div class="footer">Developed by <span class="myname">Shrusti Diggavi</span> · Powered by GPT-OSS on Groq</div>',
    unsafe_allow_html=True,
)