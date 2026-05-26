import streamlit as st
import pickle
import os
import re
import numpy as np

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="SentiScope · NLP Analyzer",
    page_icon="🔬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6e1;
}
.stApp { background-color: #0d0f14; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #f0ede6;
    margin-bottom: 0.2rem;
}
.hero-title span { color: #c8f542; font-style: italic; }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
.input-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #9ca3af;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-card {
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    border: 1px solid rgba(255,255,255,0.07);
}
.result-card.positive { background: linear-gradient(135deg,#0d2818,#0f2d1c); border-color:#166534; }
.result-card.negative { background: linear-gradient(135deg,#2a0d0d,#2d0f0f); border-color:#7f1d1d; }
.result-card.neutral  { background: linear-gradient(135deg,#1a1c26,#1e2030); border-color:#374151; }
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    color: #9ca3af;
}
.result-sentiment {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.result-sentiment.positive { color: #86efac; }
.result-sentiment.negative { color: #fca5a5; }
.result-sentiment.neutral  { color: #93c5fd; }
.result-confidence { font-family:'DM Mono',monospace; font-size:0.85rem; color:#6b7280; }
.conf-value { color:#c8f542; font-weight:500; }
.conf-bar-wrap { margin-top:1.2rem; }
.conf-bar-label {
    font-family:'DM Mono',monospace; font-size:0.65rem;
    letter-spacing:0.1em; text-transform:uppercase; color:#4b5563; margin-bottom:4px;
}
.conf-bar-bg { background:rgba(255,255,255,0.06); border-radius:99px; height:6px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:99px; }
.conf-bar-fill.positive { background:linear-gradient(90deg,#4ade80,#86efac); }
.conf-bar-fill.negative { background:linear-gradient(90deg,#f87171,#fca5a5); }
.conf-bar-fill.neutral  { background:linear-gradient(90deg,#60a5fa,#93c5fd); }
.stats-row { display:flex; gap:1rem; margin-top:1.5rem; }
.stat-box {
    flex:1; background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:1rem; text-align:center;
}
.stat-num { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#c8f542; line-height:1; }
.stat-desc {
    font-family:'DM Mono',monospace; font-size:0.62rem; color:#4b5563;
    letter-spacing:0.08em; text-transform:uppercase; margin-top:4px;
}
.history-item {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem;
    display:flex; justify-content:space-between; align-items:center;
}
.hist-text {
    font-size:0.85rem; color:#9ca3af; max-width:78%;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.hist-badge {
    font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.08em;
    padding:3px 10px; border-radius:99px; text-transform:uppercase; font-weight:500;
}
.hist-badge.positive { background:#14532d; color:#86efac; }
.hist-badge.negative { background:#450a0a; color:#fca5a5; }
.hist-badge.neutral  { background:#1e3a5f; color:#93c5fd; }
.section-divider { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:2rem 0; }
.stButton > button {
    background:#c8f542 !important; color:#0d0f14 !important;
    border:none !important; border-radius:8px !important;
    font-family:'DM Mono',monospace !important; font-size:0.78rem !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    padding:0.6rem 1.5rem !important; font-weight:500 !important;
}
.stTextArea textarea {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:#e8e6e1 !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.95rem !important;
}
.stTextArea textarea:focus {
    border-color:#c8f542 !important;
    box-shadow:0 0 0 2px rgba(200,245,66,0.12) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Train & save model if not present ────────────────────────────────────────
def train_model():
    """Train a Logistic Regression sentiment model and return the bundle."""
    import re as _re
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    def _clean(text):
        text = text.lower()
        text = _re.sub(r"http\S+|www\S+", "", text)
        text = _re.sub(r"[^a-z\s']", " ", text)
        return _re.sub(r"\s+", " ", text).strip()

    # Try NLTK corpus first
    texts, labels = [], []
    try:
        import nltk
        nltk.download("movie_reviews", quiet=True)
        from nltk.corpus import movie_reviews
        for cat in movie_reviews.categories():
            for fid in movie_reviews.fileids(cat):
                texts.append(" ".join(movie_reviews.words(fid)))
                labels.append("positive" if cat == "pos" else "negative")
    except Exception:
        pass

    # Built-in fallback sentences
    pos = [
        "This is absolutely amazing and I love every part of it",
        "Fantastic experience, highly recommend to everyone",
        "The quality exceeded my expectations by a wide margin",
        "I am so happy with this purchase, truly outstanding",
        "Best thing I have ever bought, works perfectly",
        "Brilliant product, great value for money",
        "Really impressed with the performance and design",
        "Everything worked flawlessly from the very first day",
        "Outstanding results, I could not be more pleased",
        "Wonderful experience from start to finish",
        "Very happy with the outcome, will definitely return",
        "Absolutely thrilled with this, five stars from me",
        "Highly satisfied and would recommend without hesitation",
        "Great investment, saved me a lot of time and effort",
        "Phenomenal quality and outstanding value",
        "I am genuinely impressed by how well this works",
        "Perfect for my needs, could not be happier",
        "Superb craftsmanship and beautiful design",
        "Amazing customer support and a fantastic product",
        "Absolutely perfect, no complaints whatsoever",
    ]
    neg = [
        "This is terrible and I am extremely disappointed",
        "Worst purchase I have ever made, complete waste of money",
        "Broke after two days, very poor quality",
        "Completely useless, does not work as advertised",
        "Terrible customer service, they ignored my complaints",
        "Very disappointed with the quality, feels very cheap",
        "Would not recommend this to anyone, waste of time",
        "Awful experience from beginning to end",
        "Total disappointment, expected so much better",
        "Shipping took forever and the item arrived damaged",
        "Not worth the price at all, very overpriced",
        "Really bad experience, regret buying this",
        "Absolute garbage, broke on the first day",
        "Waste of money, does not do what it claims",
        "Would give zero stars if I could, dreadful",
        "Cheap materials and terrible finish, very ugly",
        "Stopped working immediately, very poor durability",
        "Took three weeks to arrive and was broken",
        "The worst product I have encountered in years",
        "Horrible quality, unusable, and extremely overpriced",
    ]
    neu = [
        "The product was released on Tuesday in three different sizes",
        "The company announced new features for its platform today",
        "The report covers data from January to December last year",
        "The film runs for approximately one hundred and twelve minutes",
        "The store is open from nine in the morning until nine at night",
        "The university offers courses in science, arts, and engineering",
        "The event will take place at the city convention centre",
        "The team consists of twelve members from various departments",
        "The document outlines the procedures for submitting applications",
        "The results of the survey will be published next quarter",
        "The laboratory is equipped with modern scientific instruments",
        "The regulations require all participants to register in advance",
        "The population of the city has grown steadily over the past decade",
        "The study involved participants from five different countries",
        "The database is updated automatically every twenty four hours",
        "The hospital has a capacity of five hundred beds across eight floors",
        "The investigation is ongoing and no conclusions have been reached",
        "The factory produces approximately two thousand units per day",
        "The policy applies to all employees regardless of their department",
        "The new model comes with an extended warranty of three years",
    ]
    texts  += pos + neg + neu
    labels += ["positive"]*len(pos) + ["negative"]*len(neg) + ["neutral"]*len(neu)

    cleaned = [_clean(t) for t in texts]
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    vec = TfidfVectorizer(max_features=10_000, ngram_range=(1,2),
                          sublinear_tf=True, min_df=2)
    X = vec.fit_transform(cleaned)

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                         random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)

    bundle = {"model": clf, "vectorizer": vec, "label_encoder": le}
    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return bundle


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if os.path.exists("sentiment_model.pkl"):
        with open("sentiment_model.pkl", "rb") as f:
            return pickle.load(f)
    return train_model()


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("history",[]),("total",0),("pos",0),("neg",0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model — first run may take ~20 s…"):
    pipeline = load_pipeline()

EMOJI   = {"positive":"✦","negative":"✕","neutral":"◎"}
LABEL   = {"positive":"Positive","negative":"Negative","neutral":"Neutral"}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Senti<span>Scope</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ML · NLP · Scikit-learn · Logistic Regression</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-label">Enter your text below</div>', unsafe_allow_html=True)
user_text = st.text_area(
    label="text_input", label_visibility="collapsed",
    placeholder="Type or paste any text — a review, tweet, sentence, or paragraph…",
    height=140,
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("→ Analyze")
with col2:
    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.total   = 0
        st.session_state.pos     = 0
        st.session_state.neg     = 0

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze_btn and user_text.strip():
    model      = pipeline["model"]
    vectorizer = pipeline["vectorizer"]
    le         = pipeline["label_encoder"]

    X          = vectorizer.transform([user_text])
    pred_idx   = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    confidence = float(np.max(proba))
    sentiment  = le.inverse_transform([pred_idx])[0]

    st.session_state.total += 1
    if sentiment == "positive": st.session_state.pos += 1
    elif sentiment == "negative": st.session_state.neg += 1
    st.session_state.history.insert(0, {"text": user_text,
                                         "sentiment": sentiment,
                                         "conf": confidence})

    conf_pct = f"{confidence*100:.1f}%"
    bar_w    = f"{confidence*100:.0f}%"

    st.markdown(f"""
    <div class="result-card {sentiment}">
        <div class="result-label">Sentiment Detected</div>
        <div class="result-sentiment {sentiment}">{EMOJI[sentiment]} {LABEL[sentiment]}</div>
        <div class="result-confidence">Confidence: <span class="conf-value">{conf_pct}</span></div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-label">Confidence</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill {sentiment}" style="width:{bar_w}"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    wc = len(user_text.split())
    sc = max(1, len(re.split(r'[.!?]+', user_text.strip())))
    cc = len(user_text)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><div class="stat-num">{wc}</div><div class="stat-desc">Words</div></div>
        <div class="stat-box"><div class="stat-num">{sc}</div><div class="stat-desc">Sentences</div></div>
        <div class="stat-box"><div class="stat-num">{cc}</div><div class="stat-desc">Chars</div></div>
        <div class="stat-box"><div class="stat-num">{conf_pct}</div><div class="stat-desc">Confidence</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("▸ Class Probabilities"):
        classes = le.classes_
        for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
            pct = prob * 100
            clr = "#86efac" if cls=="positive" else "#fca5a5" if cls=="negative" else "#93c5fd"
            st.markdown(f"""
            <div style="margin-bottom:.6rem">
                <div style="font-family:'DM Mono',monospace;font-size:.7rem;
                            text-transform:uppercase;letter-spacing:.08em;color:#6b7280;margin-bottom:3px">{cls}</div>
                <div style="background:rgba(255,255,255,.06);border-radius:99px;height:5px;overflow:hidden">
                    <div style="width:{pct:.0f}%;height:100%;border-radius:99px;background:{clr}"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:#4b5563;margin-top:2px">{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

elif analyze_btn:
    st.warning("Please enter some text first.")

# ── Session stats ─────────────────────────────────────────────────────────────
if st.session_state.total > 0:
    neu = st.session_state.total - st.session_state.pos - st.session_state.neg
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><div class="stat-num">{st.session_state.total}</div><div class="stat-desc">Analyzed</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#86efac">{st.session_state.pos}</div><div class="stat-desc">Positive</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#fca5a5">{st.session_state.neg}</div><div class="stat-desc">Negative</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#93c5fd">{neu}</div><div class="stat-desc">Neutral</div></div>
    </div>""", unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="input-label">Recent Analyses</div>', unsafe_allow_html=True)
    for item in st.session_state.history[:8]:
        preview = item["text"][:80] + ("…" if len(item["text"])>80 else "")
        snt = item["sentiment"]
        st.markdown(f"""
        <div class="history-item">
            <div class="hist-text">{preview}</div>
            <span class="hist-badge {snt}">{LABEL[snt]}</span>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;text-align:center;font-family:'DM Mono',monospace;
            font-size:.62rem;color:#2d3748;letter-spacing:.1em;text-transform:uppercase">
    NLP Mini Project · Logistic Regression + TF-IDF · Scikit-learn
</div>""", unsafe_allow_html=True)
