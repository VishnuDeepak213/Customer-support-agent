"""Set up the LLM, LangChain tools, and agent wrapper.
Configuration expects environment variable: GROQ_API_KEY.
"""
import os
import re
import json
import time
from pathlib import Path
import logging

from langchain.agents import create_agent as lc_create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from app import tools as crm_tools

# set up basic logging for raw LLM responses
LOG_PATH = Path("llm_raw_responses.log")
logging.basicConfig(level=logging.INFO)


@tool
def customer_lookup(email: str) -> str:
    """Lookup customer details by email address."""
    return crm_tools.customer_lookup(email)


@tool
def order_history(customer_id: int) -> str:
    """Return the most recent orders for a customer_id."""
    return crm_tools.order_history(customer_id)


@tool
def ticket_status(customer_id: int) -> str:
    """Return open tickets for a customer_id."""
    return crm_tools.ticket_status(customer_id)


@tool
def escalate_to_human(reason: str, priority: str = "high") -> str:
    """Create a human escalation ticket."""
    return crm_tools.escalate_to_human(reason, priority)


def _make_tools():
    return [customer_lookup, order_history, ticket_status, escalate_to_human]


def create_agent():
    # Make ChatGroq the primary model. LocalSupportAgent is only used if an
    # LLM invocation fails at runtime (network error or API error).
    api_key = os.getenv("GROQ_API_KEY")
    llm = None
    try:
        # Deterministic params: temperature=0 for consistency, max_tokens=256 for speed
        llm = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),  # Deterministic
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "256")),    # Faster responses
        )
    except Exception:
        # If initialization fails, we'll still return a wrapper that will
        # attempt to call the LLM and fallback to LocalSupportAgent on invoke.
        llm = None

    # Try to load the canonical hwchase17/react ReAct prompt from a local file
    # If you want the exact hub prompt, download it to prompts/hwchase17_react.txt
    prompt_path = Path("prompts/hwchase17_react.txt")
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
        logging.info("Using hwchase17/react prompt from %s", prompt_path)
    else:
        # Fallback prompt: encourage ReAct-style tool use and explicit tool annotation
        system_prompt = (
            "You are a customer support assistant. Use the CRM tools when needed. "
            "Prefer factual answers over creativity. Escalate when the query indicates a human handoff is needed. "
            "When using tools, explicitly state `tool:TOOL_NAME(args)` in your reasoning and return the final answer under 'Final Answer:'."
        )

        # Get max iterations for agent loop (default 3 to allow multi-step ReAct)
        max_iters = int(os.getenv("AGENT_MAX_ITERATIONS", "3"))

    agent_graph = lc_create_agent(
        model=llm,
        tools=_make_tools(),
        system_prompt=system_prompt,
    )
    return LangChainAgentWrapper(agent_graph, max_iterations=max_iters)


class LangChainAgentWrapper:
    def __init__(self, agent_graph, max_iterations=1):
        self.agent_graph = agent_graph
        self.max_iterations = max_iterations

    def run(self, query: str):
        # Invoke the LangChain agent and capture timing and token usage.
        import time
        t0 = time.time()
        try:
            # Pass max_iterations via config dict with recursion_limit (for LangGraph)
            config = {"recursion_limit": max(10, self.max_iterations * 5)}
            result = self.agent_graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        except Exception as e:
            # On any invoke error (network/API), fall back to deterministic agent
            local = LocalSupportAgent()
            return local.run(query)
        latency = time.time() - t0

        # Log the full raw LLM/agent response for debugging (usage, errors, etc.)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as lf:
                lf.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} QUERY: {query}\n")
                lf.write(json.dumps(result, default=str, indent=2))
                lf.write("\n---\n")
        except Exception:
            logging.exception("Failed to write LLM raw response to log")

        # Also attempt to print the usage block to stdout for quick visibility
        try:
            if isinstance(result, dict):
                raw = result.get("raw_response") or result.get("llm_response") or result.get("response")
                if isinstance(raw, dict):
                    usage = raw.get("usage") or raw.get("usage", {})
                    logging.info("LLM usage for query: %s", json.dumps(usage))
                else:
                    logging.info("LLM raw response type: %s", type(raw))
        except Exception:
            logging.exception("Failed to extract usage from LLM response")

        # Extract message content
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            content = getattr(messages[-1], "content", str(messages[-1]))
        else:
            # sometimes the invoke returns a simple string
            content = str(result)

        # Extract tool calls from structured message metadata and fallback regex parsing.
        tool_calls_made = []
        try:
            for msg in messages:
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    for item in tc:
                        if isinstance(item, dict) and item.get("name"):
                            tool_calls_made.append(str(item["name"]))
        except Exception:
            pass

        if not tool_calls_made:
            try:
                # ReAct text fallback: Action: tool_name
                regex_calls = re.findall(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)", content)
                tool_calls_made.extend(regex_calls)
            except Exception:
                pass

        # De-duplicate while preserving order.
        seen = set()
        tool_calls_made = [c for c in tool_calls_made if not (c in seen or seen.add(c))]

        # Token usage: try common keys used by LLM wrappers
        tokens = None
        try:
            # result may contain raw_response or llm_response
            if isinstance(result, dict):
                raw = result.get("raw_response") or result.get("llm_response") or result.get("response")
                if raw and isinstance(raw, dict):
                    usage = raw.get("usage") or raw.get("usage", {})
                    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            # fallback: some wrappers embed usage under result['response'].usage.total_tokens
            if tokens is None and hasattr(result, "raw_response"):
                rr = getattr(result, "raw_response")
                try:
                    tokens = rr.get("usage", {}).get("total_tokens")
                except Exception:
                    tokens = None
        except Exception:
            tokens = None

        # Return a structured dict so /chat can record tokens and latency
        return {
            "response_text": content,
            "tool_calls_made": tool_calls_made,
            "escalated": False,
            "latency": latency,
            "tokens": tokens,
        }


class LocalSupportAgent:
    """Fallback agent so the project can launch without a Groq API key."""

    def run(self, query: str):
        """Deterministic dispatcher that calls the right CRM tools and returns a structured dict.

        Returns a dict with keys: response_text (str), tool_calls_made (list), escalated (bool).
        """
        lowered = query.lower()
        tool_calls = []
        escalated = False
        response_text = ""

        # Check for policy/capability questions that don't need CRM lookups
        capability_keywords = ["change my email", "verify", "password", "security", "policy", "can you", "how do i", "how to"]
        account_keywords = ["account", "profile", "settings", "preference", "information"]
        if any(kw in lowered for kw in capability_keywords):
            if any(kw in lowered for kw in account_keywords):
                response_text = "For account settings and email changes, we recommend contacting our support team for security verification. "
                response_text += "Would you like me to escalate this to a human agent?"
                return {"response_text": response_text.strip(), "tool_calls_made": [], "escalated": False}

        # extract email or customer id or order id
        email_match = re.search(r"[\w\.-]+@[\w\.-]+", query)
        customer_id = None
        order_id = None

        if email_match:
            email = email_match.group(0)
            tool_calls.append("customer_lookup")
            cust_info = crm_tools.customer_lookup(email)
            response_text += f"Customer details:\n{cust_info}\n"
            # try to parse customer id
            m = re.search(r"id:(\d+)", cust_info)
            if m:
                customer_id = int(m.group(1))

        id_match = re.search(r"customer id[:\s]*(\d+)", lowered)
        if id_match and not customer_id:
            customer_id = int(id_match.group(1))

        oid_match = re.search(r"order id[:\s]*(\d+)", lowered)
        if oid_match:
            order_id = int(oid_match.group(1))
            # try to resolve order -> customer
            try:
                from app.db import get_connection
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT customer_id FROM orders WHERE id = ? LIMIT 1", (order_id,))
                row = cur.fetchone()
                conn.close()
                if row:
                    customer_id = int(row[0])
            except Exception:
                pass
            # if not resolved, try crm_tools.order_lookup
            if customer_id is None:
                try:
                    ol = crm_tools.order_lookup(order_id)
                    m = re.search(r"customer_id:(\d+)", ol)
                    if m:
                        customer_id = int(m.group(1))
                except Exception:
                    pass

        # Escalation intent
        if any(word in lowered for word in ["refund", "cancel", "escalate", "human", "speak to a human", "angry", "complaint"]):
            tool_calls.append("escalate_to_human")
            ticket = crm_tools.escalate_to_human("User requested human review via web UI", "high")
            escalated = True
            response_text += f"Escalated to human support. Ticket: {ticket}\n"

        # Order/status intent
        if any(word in lowered for word in ["order", "shipping", "delivery", "status"]) or order_id is not None:
            if customer_id is not None:
                tool_calls.append("order_history")
                orders = crm_tools.order_history(customer_id)
                response_text += f"Order history:\n{orders}\n"
            else:
                # If we saw an order-related query but couldn't resolve a customer, try order lookup
                if order_id is not None:
                    try:
                        ol = crm_tools.order_lookup(order_id)
                        response_text += f"Order lookup:\n{ol}\n"
                        m = re.search(r"customer_id:(\d+)", ol)
                        if m:
                            customer_id = int(m.group(1))
                            tool_calls.append("order_history")
                            orders = crm_tools.order_history(customer_id)
                            response_text += f"Order history:\n{orders}\n"
                    except Exception:
                        pass
                # best-effort: include order_history in tool_calls to signal intent
                if "order_history" not in tool_calls:
                    tool_calls.append("order_history")

        # Ticket/technical intent
        if any(word in lowered for word in ["ticket", "issue", "problem", "bug", "login", "broken", "troubleshoot"]) and customer_id is not None:
            tool_calls.append("ticket_status")
            tickets = crm_tools.ticket_status(customer_id)
            response_text += f"Open tickets:\n{tickets}\n"

        # If order id was provided and we resolved a customer, include order history
        if order_id is not None and customer_id is not None:
            # ensure we call order_history for order-linked tech queries
            if "order_history" not in tool_calls:
                tool_calls.append("order_history")
                orders = crm_tools.order_history(customer_id)
                response_text += f"Order history for order id {order_id}:\n{orders}\n"

        if not response_text:
            response_text = "I need a customer email or customer ID or order ID to look up details. Please provide this information so I can help you better."

        return {"response_text": response_text.strip(), "tool_calls_made": tool_calls, "escalated": escalated}


_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = create_agent()
    return _AGENT
