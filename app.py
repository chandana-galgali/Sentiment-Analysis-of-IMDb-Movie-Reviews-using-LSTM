"""CineSentiment — Streamlit IMDb LSTM demo."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import streamlit as st

st.set_page_config(page_title="CineSentiment · Movie Review AI", page_icon="🎬", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Oswald:wght@500;600;700&display=swap');
:root {color-scheme:dark}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse 85% 45% at 10% 2%,rgba(149,38,64,.25),transparent 65%),radial-gradient(ellipse 60% 45% at 94% 17%,rgba(246,181,54,.10),transparent 70%),#0b0b10;color:#f6f0e8}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:860px;padding-top:2.3rem;padding-bottom:3rem}
.cine-hero {position:relative;overflow:hidden;border:1px solid rgba(255,211,146,.26);border-radius:23px;min-height:245px;padding:36px 34px;background:radial-gradient(circle at 87% 35%,rgba(255,198,83,.22) 0%,transparent 35%),linear-gradient(120deg,#26101f 2%,#11111a 65%,#18121b);box-shadow:0 28px 65px rgba(0,0,0,.35)}
.cine-hero::before {content:'';position:absolute;right:10%;top:-85px;width:170px;height:360px;transform:rotate(30deg);background:linear-gradient(90deg,transparent,rgba(251,207,137,.16),transparent);filter:blur(20px);pointer-events:none}
.cine-hero::after {content:'◉';position:absolute;right:28px;top:0px;font-size:185px;line-height:1;color:rgba(255,224,160,.085);font-family:serif;pointer-events:none}
.eyebrow {letter-spacing:.24em;text-transform:uppercase;color:#f4bd79;font-size:12px;font-weight:700;margin-bottom:14px}
.cine-hero h1 {font-family:'Oswald',sans-serif!important;font-weight:700;letter-spacing:.018em;color:#fff7ea!important;font-size:3.4rem!important;line-height:1.05;margin:0 0 14px}
.cine-hero p {color:#ddd0cd;font-size:15px;line-height:1.7;max-width:470px;margin:0}
.filmstrip {height:24px;border-top:2px solid #d99f55;border-bottom:2px solid #d99f55;opacity:.6;background:repeating-linear-gradient(90deg,transparent 0,transparent 16px,#d99f55 16px,#d99f55 28px,transparent 28px,transparent 48px);background-size:auto 10px;background-position:center;background-repeat:repeat-x;margin:18px 0 26px}
.section-title {font-family:'Oswald',sans-serif;letter-spacing:.04em;font-size:1.45rem;color:#fff1dd;margin:15px 0 5px}
.muted {font-size:13px;color:#ab9ca2;margin-bottom:16px}
.result {padding:24px;border:1px solid #5a4947;border-radius:16px;background:linear-gradient(125deg,rgba(48,32,43,.7),rgba(19,22,32,.8));margin-top:20px}
.result .result-kicker {letter-spacing:.2em;text-transform:uppercase;color:#f0ba76;font-weight:700;font-size:11px}
.result .result-label {font-family:'Oswald',sans-serif;font-size:2.2rem;color:#fff4e6}
[data-testid="stTextArea"] textarea {background:#17151d!important;border:1px solid #56404b!important;color:#fff5e9!important;border-radius:13px!important}
div.stButton > button[kind="primary"] {background:linear-gradient(100deg,#b64253,#dc985d)!important;border:0!important;color:#fff!important;font-weight:700;border-radius:10px!important}
div.stButton > button[kind="secondary"] {border:1px solid #61505b!important;color:#f5e7da!important;background:#231e2a!important;border-radius:10px!important}
[data-testid="stMetric"] {padding:12px;border:1px solid #4b3b42;border-radius:12px;background:#1a151d}
.footer-note {text-align:center;color:#90858b;font-size:12px;padding-top:12px}
@media(max-width:600px){.cine-hero{min-height:200px;padding:25px 20px}.cine-hero h1{font-size:2.5rem!important}.cine-hero::after{font-size:120px}}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="cine-hero"><div class="eyebrow">★ Now showing · NLP lab</div><h1>CINE<br>SENTIMENT.</h1><p>Every review tells a story. Find out whether this LSTM model reads it as a hit or a miss.</p></div><div class="filmstrip"></div>""", unsafe_allow_html=True)
st.markdown('<div class="section-title">THE REVIEW ROOM</div><div class="muted">Write a movie review or load a sample, then run the trained model.</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the trained LSTM model…")
def get_model():
    from imdb_sentiment.model import load_artifacts
    return load_artifacts()

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

a, b = st.columns(2)
with a:
    if st.button("🍿 Positive sample", use_container_width=True):
        st.session_state.review_text = "An excellent movie with thoughtful storytelling and wonderful performances. I loved every minute."
with b:
    if st.button("🎭 Negative sample", use_container_width=True):
        st.session_state.review_text = "A dull and disappointing film with an incoherent plot and weak acting. I regret watching it."

review = st.text_area("Your movie review", key="review_text", placeholder="The cinematography was stunning, but the story was hard to follow…", height=170, max_chars=5000)
if st.button("🎬 REVEAL THE SENTIMENT", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Add a review first — every movie needs a script!")
    else:
        try:
            model, metadata = get_model()
            from imdb_sentiment.model import predict
            result = predict(review, model, metadata)
            sentiment = result["label"]
            emoji = "🍿" if sentiment == "Positive" else "🥀"
            st.markdown(f'<div class="result"><div class="result-kicker">THE VERDICT</div><div class="result-label">{emoji} {sentiment.upper()}</div></div>', unsafe_allow_html=True)
            st.metric("Positive sentiment score", f'{result["positive_probability"]:.1%}')
            st.progress(result["positive_probability"])
            st.caption("Score is a model output, not calibrated certainty. Sarcasm, negation and mixed opinions may be misclassified.")
        except FileNotFoundError:
            st.error("Model files are missing. Add models/sentiment_lstm.keras and models/metadata.json from your trained run (see README).")
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}. Check that the model artifacts and package versions match your training environment.")

st.markdown('<div class="filmstrip"></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: st.metric("Test accuracy", "85.28%")
with c2: st.metric("Test loss", "0.3494")
with c3: st.metric("Training", "5 epochs")
st.markdown('<div class="footer-note">IMDb review sentiment · TensorFlow/Keras LSTM · Educational demonstration</div>', unsafe_allow_html=True)
