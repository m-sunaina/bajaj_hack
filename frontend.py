# ===================================================
# Streamlit Frontend
# ===================================================

import streamlit as st
import requests

API_URL = "http://localhost:8000"


st.set_page_config(page_title="📄 Insurance Assistant", layout="centered")
st.title("📄 Insurance Assistant (PDF + Gemini + Qdrant)")

# -----------------------
# File Upload Section
# -----------------------
st.subheader("📤 Upload Insurance Document")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload & Process"):
        with st.spinner("Uploading and processing..."):
            try:
                res = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=300
                )
                if res.ok:
                    st.success("✅ File uploaded and processed!")
                    st.write(f"Chunks created: {res.json().get('chunks', 0)}")
                else:
                    st.error("❌ Upload failed")
                    st.text(res.text)
            except Exception as e:
                st.error(f"🚨 Error: {e}")

# -----------------------
# Query Section
# -----------------------
st.subheader("🧠 Ask a Question")
query = st.text_area("Enter your insurance-related question:")

if st.button("Get Decision"):
    if not query.strip():
        st.warning("⚠️ Please enter a query first.")
    else:
        with st.spinner("Querying backend..."):
            try:
                res = requests.post(f"{API_URL}/query", params={"query": query}, timeout=300)
                if res.ok:
                    data = res.json()

                    st.markdown("### ✅ Parsed Query")
                    st.json(data.get("parsed_query", {}))

                    st.markdown("### 🧾 Decision Result")
                    st.write(data.get("decision_result", "No decision returned"))

                else:
                    st.error("❌ Error in backend response")
                    st.text(res.text)
            except Exception as e:
                st.error(f"🚨 Error: {e}")

#st.subheader("📚 Bulk Query")
bulk_file = st.file_uploader("Upload PDF for bulk query (optional)", type=["pdf"])
bulk_url = st.text_input("Or enter document URL (optional)")
bulk_questions = st.text_area("Enter one or more questions as JSON array or plain text", height=150)

if st.button("Run Bulk Query"):
    if not bulk_questions.strip():
        st.warning("⚠️ Please enter questions for bulk query.")
    else:
        with st.spinner("Processing bulk query..."):
            try:
                files = {"file": (bulk_file.name, bulk_file, "application/pdf")} if bulk_file else None
                data = {"questions": bulk_questions}
                if bulk_url:
                    data["document_url"] = bulk_url

                res = requests.post(
                    f"{API_URL}/bulk_query",
                    files=files,
                    data=data,
                    timeout=600
                )
                if res.ok:
                    answers = res.json().get("answers", [])
                    for i, ans in enumerate(answers, 1):
                        st.markdown(f"**Q{i}:**")
                        st.write(ans)
                else:
                    st.error("❌ Bulk query failed")
                    st.text(res.text)
            except Exception as e:
                st.error(f"🚨 Error: {e}")

