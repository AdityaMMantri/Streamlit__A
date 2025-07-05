import streamlit as st

# Configure page FIRST
st.set_page_config(page_title="Fake Job Detector", layout="centered")

# Track which model is currently shown
def load_distillbert():
    try:
        import distill_app
        distill_app.run()
    except Exception as e:
        st.error(f"❌ Failed to load DistilBERT: {e}")
        st.stop()

def load_bert_uncased():
    try:
        import unased_app  # Make sure the filename is correct
        unased_app.run()
    except ImportError as e:
        st.error(f"❌ Could not import the selected model module: {str(e)}")
    except AttributeError as e:
        st.error(f"❌ The selected module doesn't have a 'run()' function: {str(e)}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

def main():
    st.title(" Fake Job Posting Detector")

    # Initialize state variables on first run
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model_selector = "DistilBERT (Default)"
        st.session_state.first_load_complete = False

    # Only load DistilBERT on first launch
    if not st.session_state.first_load_complete:
        with st.spinner("🔄 Loading DistilBERT model..."):
            load_distillbert()
        st.session_state.first_load_complete = True  # ✅ Let Streamlit remember this
        st.rerun()  # 🔁 Now safely rerun the app to show model selector

    # After first DistilBERT load, allow model switching
    model_switch = st.radio(
        "Choose a model to use:",
        ["DistilBERT (Default)", "BERT-uncased"],
        horizontal=True,
        key="model_selector"
    )

    st.markdown("---")

    # Load selected model
    if model_switch == "DistilBERT (Default)":
        load_distillbert()
    else:
        load_bert_uncased()

if __name__ == "__main__":
    main()