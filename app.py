import streamlit as st
from sentence_transformers import SentenceTransformer
from math_engine import analyze_summary

st.set_page_config(page_title="AI Hallucination & Fact-Checker", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

if "source_input" not in st.session_state:
    st.session_state.source_input = ""
if "summary_input" not in st.session_state:
    st.session_state.summary_input = ""

st.title("🔍 No-Hype AI Hallucination & Fact-Checker")
st.markdown("Audit AI-generated summaries against source text using **cosine similarity** and **symbolic fact verification**.")

#sidebar parameter control
penalty = st.sidebar.slider("Unverified Fact Penalty Factor", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.05, help = "Multiplier applied to a sentence's similarity score when it introduces numbers of proper nouns missing from the source.")

# One-click sample button
if st.button("Load Sample Demo"):
    st.session_state.source_input = (
        "Mount Lyell is a historic copper mine located in Queenstown, Tasmania. "
        "Mining operations began in 1893 and produced over 1.8 million tonnes of copper."
    )
    st.session_state.summary_input = (
        "Mount Lyell is a copper mine in Queenstown, Tasmania that opened in 1893. "
        "It produced over 50 million tonnes of gold during its operation."
    )

#Side-by-side text areas linked to session_state via 'key'
col1, col2 = st.columns(2)
with col1:
    st.text_area("Source Document", height=220, key="source_input")
with col2:
    st.text_area("AI-Generated summary", height=220, key="summary_input")


if st.button("Analyze Summary", type="primary"):
    with st.spinner("Computing embeddings and auditing facts...."):
        results = analyze_summary(
            st.session_state.source_input,
            st.session_state.summary_input,
            load_model(),
            penalty_factor=penalty, 
        )

    if "error" in results:
        st.error(results["error"])
    else:
        st.divider()
        #Top KPI metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Trust Score",f"{results['overall_score']}%")
        m2.metric("Unverified Facts Flagged", results["total_unverified_facts"])
        m3.metric("Sentences Analyzed", len(results["sentence_breakdown"]))

        st.subheader("Sentence-by-Sentence Audit")
        for item in results["sentence_breakdown"]:
            header = f"***[{item['status']}]** (score: {item['final_score']} | Raw Sim: {item['raw_similarity']}) - {item['sentence']}"
            if item ["status"] == "High":
                st.success(header)
            elif item["status"] == "Moderate":
                st.warning(header)
            else:
                st.error(header)

            with st.expander("View Evidence & Fact Audit"):
                st.write(f"**Closest Source Match:** {item['best_match_source']}")
                if item["unverified_facts"]:
                    st.write(f"**⚠️ Unverified Numbers/Entities:** {', '.join(item['unverified_facts'])}")
                else:
                    st.write("**✅ Hard Facts:** All numbers and proper nouns verified in source.")
