import streamlit as st
import requests
import json
import os
from datetime import datetime

# Page config
st.set_page_config(page_title="AI Support Agent", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 Agentforce Customer Support Agent")

def _resolve_api_url() -> str:
    """Resolve the backend URL from secrets/env and reject known bad values."""
    fallback = "https://customer-support-agent-ygnl.onrender.com"
    candidate = os.getenv("API_URL", fallback)
    try:
        if "API_URL" in st.secrets:
            candidate = st.secrets["API_URL"]
    except Exception:
        pass

    bad_markers = ("127.0.0.1", "localhost", "console.groq.com", "groq.com/keys")
    if not isinstance(candidate, str) or not candidate.strip():
        return fallback
    if any(marker in candidate for marker in bad_markers):
        return fallback
    return candidate.strip().rstrip("/")


DEFAULT_API_URL = _resolve_api_url()

# Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, help="Backend FastAPI endpoint")
    mode = st.radio("Agent Mode", ["Hybrid (92% accuracy)", "Pure ReAct (22% accuracy)"])
    st.markdown("---")
    st.markdown("**Metrics:**")
    st.metric("Tool Accuracy (Hybrid)", "92%")
    st.metric("Tool Accuracy (Pure ReAct)", "22%")
    st.metric("Avg Latency", "1.9s")

# Session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().isoformat()

# Main chat interface
st.subheader("Chat with Support Agent")

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if "tool_calls" in msg and msg["tool_calls"]:
                st.caption(f"🔧 Tools used: {', '.join(msg['tool_calls'])}")
            if "escalated" in msg and msg["escalated"]:
                st.warning(f"⚠️ Escalated to human support: {msg.get('escalation_id', 'N/A')}")

# Input and send
col1, col2 = st.columns([0.9, 0.1])
with col1:
    user_query = st.text_input("Your question:", placeholder="e.g., My email is user@example.com, check my billing")
with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

if send_button and user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Call backend API
    try:
        with st.spinner("🔄 Processing..."):
            # Determine PURE_REACT setting
            pure_react = "1" if "Pure ReAct" in mode else "0"
            
            response = requests.post(
                f"{api_url}/chat",
                json={"query": user_query, "session_id": st.session_state.session_id},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Add assistant message to history with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data.get("response_text", "No response"),
                    "tool_calls": data.get("tool_calls_made", []),
                    "escalated": data.get("escalated", False),
                    "escalation_id": data.get("escalation_id"),
                    "latency": data.get("latency"),
                    "tokens": data.get("tokens")
                })
                
                # Display response with metrics
                with st.chat_message("assistant"):
                    st.write(data.get("response_text", "No response"))
                    
                    # Show tool calls if any
                    if data.get("tool_calls_made"):
                        st.caption(f"🔧 Tools used: {', '.join(data['tool_calls_made'])}")
                    
                    # Show escalation if triggered
                    if data.get("escalated"):
                        st.warning(f"⚠️ Escalated to human support")
                    
                    # Show metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Latency", f"{data.get('latency', 0):.2f}s")
                    with col_m2:
                        st.metric("Tokens", data.get("tokens", 0))
                    with col_m3:
                        st.metric("Hallucinated", "✓" if data.get("hallucinated") else "✗")
                
                st.rerun()
            else:
                st.error(f"❌ Backend error: {response.status_code}")
                st.error(response.text)
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        st.info(f"Make sure backend is running at: {api_url}")

# Sidebar: Conversation history & escalations
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📊 View Metrics"):
        st.info("Metrics stored in backend database")
    
    if st.button("📞 View Escalations"):
        try:
            esc_response = requests.get(f"{api_url}/escalations", timeout=5)
            if esc_response.status_code == 200:
                escalations = esc_response.json()
                if escalations:
                    st.write("Recent escalations:")
                    for esc in escalations[:5]:
                        st.write(f"- {esc.get('ticket_number')}: {esc.get('reason')[:50]}...")
                else:
                    st.write("No escalations yet")
        except Exception as e:
            st.error(f"Could not load escalations: {e}")
