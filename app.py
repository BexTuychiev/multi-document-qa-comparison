"""
Multi-Document Research Assistant - Streamlit App
Compares long-context performance across different LLM providers.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from document_loader import load_documents, count_tokens
from query_handler import query_model
from model_config import check_deepseek_balance
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Multi-Document Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Multi-Document Research Assistant")
st.markdown("Compare long-context performance across GPT-5, Claude Sonnet 4.5, DeepSeek v3.2, and DeepSeek v3.1")

# Sidebar - Document info
with st.sidebar:
    st.header("📄 Documents Loaded")

    if st.button("Load Documents"):
        with st.spinner("Loading documents..."):
            context, token_count, doc_names = load_documents("documents")
            st.session_state.context = context
            st.session_state.token_count = token_count
            st.session_state.doc_names = doc_names

    if "token_count" in st.session_state:
        st.success(f"✅ Loaded {len(st.session_state.doc_names)} documents")
        st.metric("Total Tokens", f"{st.session_state.token_count:,}")

        st.write("**Documents:**")
        for name in st.session_state.doc_names:
            st.write(f"• {name}")

    # Check DeepSeek balance
    st.divider()
    st.header("💰 API Status")

    if st.button("Check DeepSeek Balance"):
        with st.spinner("Checking balance..."):
            balance_info = check_deepseek_balance()
            if balance_info:
                is_available = balance_info.get('is_available', False)
                balances = balance_info.get('balance_infos', [])

                if balances and len(balances) > 0:
                    total = balances[0].get('total_balance', '0.00')
                    if is_available and float(total) > 0:
                        st.success(f"✅ Balance: ${total}")
                    else:
                        st.error(f"⚠️ Insufficient balance: ${total}")
                        st.info("💡 Add credits at https://platform.deepseek.com")
            else:
                st.warning("Could not check balance")

# Main area
if "context" not in st.session_state:
    st.info("👈 Click 'Load Documents' in the sidebar to begin")
else:
    # Model selection
    st.subheader("Select Models to Compare")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        use_gpt5 = st.checkbox("GPT-5", value=True)
    with col2:
        use_claude = st.checkbox("Claude Sonnet 4.5", value=True)
    with col3:
        use_deepseek_v32 = st.checkbox("DeepSeek v3.2-Exp", value=True)
    with col4:
        use_deepseek_v31 = st.checkbox("DeepSeek v3.1-Terminus", value=True)

    # Question input
    st.subheader("Ask a Question")

    # Pre-defined questions
    sample_questions = [
        "Compare the main approaches to attention mechanisms described in these documents",
        "What are the key differences between sparse and dense attention?",
        "Summarize the common themes across all documents"
    ]

    question_choice = st.selectbox(
        "Select a sample question or write your own:",
        ["Custom"] + sample_questions
    )

    if question_choice == "Custom":
        question = st.text_area("Enter your question:", height=100)
    else:
        question = st.text_area("Enter your question:", value=question_choice, height=100)

    # Query button
    if st.button("🚀 Query Models", type="primary"):
        if not question:
            st.error("Please enter a question")
        else:
            selected_models = []
            if use_gpt5:
                selected_models.append("gpt-5")
            if use_claude:
                selected_models.append("claude-sonnet-4-5-20250929")
            if use_deepseek_v32:
                selected_models.append("deepseek-chat")
            if use_deepseek_v31:
                selected_models.append("deepseek-chat-v3.1")

            if not selected_models:
                st.error("Please select at least one model")
            else:
                results = []

                # Query each model
                for model_name in selected_models:
                    with st.spinner(f"Querying {model_name}..."):
                        result = query_model(
                            model_name,
                            st.session_state.context,
                            question
                        )
                        results.append(result)

                # Store results
                st.session_state.results = results
                st.session_state.question = question

# Display results
if "results" in st.session_state:
    st.divider()
    st.subheader("📊 Results")

    results = st.session_state.results

    # Display responses
    st.markdown("### Responses")
    for result in results:
        with st.expander(f"**{result['model']}** - ${result['cost']:.4f} | {result['time']:.2f}s"):
            if result['error']:
                st.error(f"Error: {result['error']}")
            else:
                st.write(result['response'])

    # Metrics comparison
    st.markdown("### Metrics Comparison")

    # Create metrics dataframe
    metrics_df = pd.DataFrame([
        {
            "Model": r['model'],
            "Input Tokens": r['input_tokens'],
            "Output Tokens": r['output_tokens'],
            "Total Tokens": r['total_tokens'],
            "Cost ($)": f"${r['cost']:.4f}",
            "Time (s)": f"{r['time']:.2f}"
        }
        for r in results
    ])

    st.dataframe(metrics_df, use_container_width=True)

    # Charts
    st.markdown("### Visual Comparison")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Cost comparison
    axes[0, 0].bar([r['model'] for r in results], [r['cost'] for r in results], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(results)])
    axes[0, 0].set_title('Cost Comparison')
    axes[0, 0].set_ylabel('Cost ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Time comparison
    axes[0, 1].bar([r['model'] for r in results], [r['time'] for r in results], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(results)])
    axes[0, 1].set_title('Response Time Comparison')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Token usage
    models = [r['model'] for r in results]
    input_tokens = [r['input_tokens'] for r in results]
    output_tokens = [r['output_tokens'] for r in results]

    x = range(len(models))
    width = 0.35
    axes[1, 0].bar([i - width/2 for i in x], input_tokens, width, label='Input', color='#1f77b4')
    axes[1, 0].bar([i + width/2 for i in x], output_tokens, width, label='Output', color='#ff7f0e')
    axes[1, 0].set_title('Token Usage Comparison')
    axes[1, 0].set_ylabel('Tokens')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].legend()

    # Cost vs Time scatter
    axes[1, 1].scatter([r['cost'] for r in results], [r['time'] for r in results], s=100, alpha=0.6, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(results)])
    for i, r in enumerate(results):
        axes[1, 1].annotate(r['model'], (r['cost'], r['time']), fontsize=8, ha='right')
    axes[1, 1].set_title('Cost vs Time Trade-off')
    axes[1, 1].set_xlabel('Cost ($)')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Key findings
    st.markdown("### 🔍 Key Findings")

    if len(results) > 1:
        cheapest = min(results, key=lambda x: x['cost'])
        fastest = min(results, key=lambda x: x['time'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Cost-Effective", cheapest['model'], f"${cheapest['cost']:.4f}")
        with col2:
            st.metric("Fastest Response", fastest['model'], f"{fastest['time']:.2f}s")
