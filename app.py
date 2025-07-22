import os
import json
import re
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
import plotly.graph_objects as go
import plotly.express as px
import requests
from requests.exceptions import HTTPError
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ——— Monkey‐patch & env loading —————————————————————————————————————————
ChatOpenAI.bind_tools = lambda self, tools: self
load_dotenv(find_dotenv(), override=True)
def load_env_var(name: str) -> str:
    v = os.getenv(name)
    if not v:
        st.error(f"Missing `{name}`")
        st.stop()
    return v

OPENAI_API_KEY = load_env_var("OPENAI_API_KEY")
INE_API_URL    = os.getenv("INE_NEO4J_API_URL","http://localhost:8080/neo4j/query")
parts          = urlparse(INE_API_URL)
BASE_PROXY     = f"{parts.scheme}://{parts.netloc}"
DIAGNO_URL     = f"{BASE_PROXY}/diagnostic"

# ——— HTTP client ————————————————————————————————————————————————————————
class IngenuityClient:
    def __init__(self,url:str): self.url = url.rstrip("/")
    def run_query(self,cypher:str)->List[Dict]:
        try:
            resp = requests.post(self.url, json={"q":cypher}, timeout=10); resp.raise_for_status()
        except HTTPError:
            if cypher.strip().upper().startswith("EXPLAIN"):
                return []
            resp = requests.get(self.url, params={"q":cypher}, timeout=10); resp.raise_for_status()
        data = resp.json()
        if isinstance(data.get("data"), list):
            return data["data"]
        results = data.get("results", [])
        if not results:
            return []
        cols = results[0]["columns"]
        recs = results[0]["data"]
        return [dict(zip(cols, r["row"])) for r in recs]

client     = IngenuityClient(INE_API_URL)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ——— Page setup ————————————————————————————————————————————————————————
st.set_page_config(page_title="Gas & Oil Engineer Assistant", layout="wide")
st.markdown("""
<style>
  body { font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;}
  h1,h2,h3,h4,h5,h6 { color:#333;}
</style>
""", unsafe_allow_html=True)

# Health check
try:
    r = requests.get(DIAGNO_URL, timeout=2); r.raise_for_status()
    st.sidebar.success("🟢 Ingenuity API OK")
except Exception as e:
    st.sidebar.error(f"🔴 Ingenuity API Down: {e}")

# ——— Sidebar LLM controls —————————————————————————————————————————————————
st.sidebar.header("LLM Settings")
model_name_cypher        = st.sidebar.selectbox("Model for Cypher", ["gpt-4","gpt-3.5-turbo"], 0)
model_name_context       = st.sidebar.selectbox("Model for Context/Report", ["gpt-3.5-turbo","gpt-4"], 0)
temperature              = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)
enable_cot               = st.sidebar.checkbox("🧠 Chain-of-Thought", True)
enable_fs                = st.sidebar.checkbox("📝 Few-Shot", True)
show_versions            = st.sidebar.checkbox("🔢 Explain versions?", False)
aggressive_schema_shrink = st.sidebar.checkbox("Aggressively Shrink Schema", True)

model_cypher  = ChatOpenAI(model_name=model_name_cypher,  temperature=temperature, openai_api_key=OPENAI_API_KEY)
model_context = ChatOpenAI(model_name=model_name_context, temperature=temperature, openai_api_key=OPENAI_API_KEY)

# ——— Session state ————————————————————————————————————————————————————————
for key in ("question_history","timestamp_history","cost_history","answer_history","model_history","temp_history"):
    if key not in st.session_state:
        st.session_state[key] = []

# ——— Token & caching helpers —————————————————————————————————————————————————
def count_tokens(text:str,model:str)->int:
    try: enc = tiktoken.encoding_for_model(model)
    except: enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def call_model(prompt,model_obj,model_name):
    in_toks = count_tokens(prompt, model_name)
    resp    = model_obj.invoke(prompt)
    out     = resp.content.strip()
    out_toks= count_tokens(out, model_name)
    ic, oc  = (0.01/1000, 0.03/1000) if "gpt-4" in model_name else (0.0005/1000, 0.0015/1000)
    return out, (in_toks*ic + out_toks*oc)

def get_cache_dir():
    d = os.path.join(os.path.dirname(__file__), ".llm_cache")
    os.makedirs(d, exist_ok=True)
    return d

def hash_key(*args):
    return hashlib.sha256("|||".join(str(a) for a in args).encode()).hexdigest()

def cached_llm_call(prompt,model_obj,model_name,tag):
    cdir = get_cache_dir()
    key  = hash_key(tag, prompt, model_name)
    path = os.path.join(cdir, f"{key}.txt")
    if os.path.exists(path):
        return open(path, "r").read(), 0.0
    out, cost = call_model(prompt, model_obj, model_name)
    with open(path, "w") as f: f.write(out)
    return out, cost

# ——— Load prompts & examples —————————————————————————————————————————————————
PROMPT_DIR = "prompts"
def load_prompt(p): return open(p, encoding="utf-8").read()
def_prompt_context  = load_prompt(f"{PROMPT_DIR}/context_prompt.txt")
def_prompt_semantic = load_prompt(f"{PROMPT_DIR}/cypher_prompt.txt")
def_prompt_engineer = load_prompt(f"{PROMPT_DIR}/engineer_prompt.txt")
FEW_COT            = load_prompt(f"{PROMPT_DIR}/few_shot_examples_cot.txt")
FEW_CYP            = load_prompt(f"{PROMPT_DIR}/few_shot_examples_cypher.txt")

# ——— Schema extraction ——————————————————————————————————————————————————————
@st.cache_data
def extract_database_schema():
    nodes = [r["label"] for r in client.run_query("CALL db.labels()") if r.get("label")]
    rels  = [r["relationshipType"] for r in client.run_query("CALL db.relationshipTypes()") if r.get("relationshipType")]
    props = [r["propertyKey"] for r in client.run_query("CALL db.propertyKeys()") if r.get("propertyKey")]
    return json.dumps({"nodes":nodes,"relationships":rels,"properties":props}, indent=2)

@st.cache_data
def extract_relationship_patterns():
    q = "MATCH (a)-[r]->(b) RETURN DISTINCT type(r) AS rel, labels(a)[0] AS from, labels(b)[0] AS to"
    return pd.DataFrame(client.run_query(q))

FULL_SCHEMA_JSON = extract_database_schema()
RELS_DF         = extract_relationship_patterns()
RELS_SNIPPET    = "\n".join(f"  - ({r['from']})-[:{r['rel']}]->({r['to']})" for _,r in RELS_DF.iterrows())

def smart_shrink(s:str)->str:
    ess = {
        "nodes": ["Well","Formation","Facility","Document","WellTestResults"],
        "relationships": ["has_well","has_formation","hasdocument"],
        "properties": ["name","code","active","startUpDate"]
    }
    try:
        d = json.loads(s)
        return json.dumps({
            "nodes":[n for n in d["nodes"] if n in ess["nodes"]],
            "relationships":[r for r in d["relationships"] if r in ess["relationships"]],
            "properties":[p for p in d["properties"] if p in ess["properties"]],
        }, indent=2)
    except:
        return s

# ——— Semantic pipeline ————————————————————————————————————————————————————
def semantic_method(question, cp, sp, ep):
    schema = smart_shrink(FULL_SCHEMA_JSON) if aggressive_schema_shrink else FULL_SCHEMA_JSON

    # Context analysis
    ctx_prompt     = cp.replace("{relationship_snippets}", RELS_SNIPPET).replace("{schema}", schema).replace("{question}", question)
    context_out, cost_ctx = cached_llm_call(ctx_prompt, model_context, model_name_context, "context")

    # Cypher generation
    sem_prompt     = sp.replace("{examples}", (FEW_CYP if enable_fs else "") + ("\n\n"+FEW_COT if enable_cot else ""))\
                       .replace("{schema}", schema).replace("{context}", context_out).replace("{question}", question)
    max_ctx        = 16385 if "gpt-4" in model_name_cypher else 4096
    if count_tokens(sem_prompt, model_name_cypher) > max_ctx - 500:
        sem_prompt = sp.format(examples="", schema=json.dumps(json.loads(FULL_SCHEMA_JSON)[:12]), context=context_out, question=question)
    raw_cy, cost_cy = cached_llm_call(sem_prompt, model_cypher, model_name_cypher, "cypher")
    cleaned       = re.sub(r"```(?:\s*cypher)?", "", raw_cy, flags=re.IGNORECASE).strip()
    if not cleaned.endswith(";"):
        cleaned += ";"
    matches = re.findall(r"(?i)MATCH\s*\([\s\S]*?;", cleaned)
    cypher  = (matches[-1].strip() if matches else cleaned).rstrip(";") + ";"

    client.run_query(f"EXPLAIN {cypher}")
    try:
        records = client.run_query(cypher)
    except HTTPError as e:
        return {"error": f"Query failed: {e}"}

    # Insights
    snippet, cost_ins = json.dumps(records[:5], indent=2, default=str), 0.0
    eng_prompt        = ep.format(results=snippet, question=question)
    if not show_versions:
        eng_prompt = eng_prompt.replace("explain their validity windows", "omit version details")
    insights_out, cost_ins = cached_llm_call(eng_prompt, model_context, model_name_context, "insights")

    return {
        "schema":        schema,
        "context":       context_out,
        "cypher_query":  cypher,
        "query_results": records,
        "insights":      insights_out,
        "costs": {
            "context_cost": cost_ctx,
            "cypher_cost":  cost_cy,
            "insights_cost": cost_ins,
            "total_cost":   cost_ctx + cost_cy + cost_ins
        }
    }

# ——— UI —————————————————————————————————————————————————————————————————
st.title("Gas & Oil Engineer Assistant")
tabs = st.tabs(["Semantic Retriever","Vectorise & Retrieve"])

with tabs[0]:
    st.subheader("Semantic Retriever")
    with st.expander("Prompt Configuration", expanded=False):
        custom_cp = st.text_area("Context Analyzer Prompt", def_prompt_context, height=200)
        custom_sp = st.text_area("Semantic Query Prompt",  def_prompt_semantic, height=200)
        custom_ep = st.text_area("Engineer Assistant Prompt", def_prompt_engineer, height=200)
        st.markdown("**Few-Shot Examples (Cypher)**")
        st.text_area("few_shot_examples_cypher.txt", FEW_CYP, height=200)
        st.markdown("**Chain-of-Thought Templates**")
        st.text_area("few_shot_examples_cot.txt", FEW_COT, height=200)

 

   
    demos = [
        "List formations at 'Brage'.",
        "List injection wells.",
        "List Water Production wells and their formations"
    ]
    # dropdown to pick a demo
    demo_choice = st.selectbox("Pick a demo question", demos)
    # pre-fill the text box with that choice (but still editable)
    user_q = st.text_input("Your question", value=demo_choice)


    if st.button("Run Semantic Retriever"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out = semantic_method(user_q, custom_cp, custom_sp, custom_ep)

            if "error" in out:
                st.error(out["error"])
            else:
                # Record histories
                st.session_state.question_history.append(user_q)
                st.session_state.timestamp_history.append(now)
                st.session_state.cost_history.append(out["costs"]["total_cost"])
                st.session_state.model_history.append(model_name_cypher)
                st.session_state.temp_history.append(temperature)

                # Prompt Analyses
                with st.expander("Prompt Analyses", expanded=False):
                    st.markdown("**Database Schema (JSON)**")
                    st.code(out["schema"], language="json")
                    st.markdown("**Relationship Patterns**")
                    st.dataframe(RELS_DF)
                    st.markdown("**Context Analysis**")
                    st.markdown(out["context"])
                    st.markdown("**Generated Cypher Query**")
                    st.code(out["cypher_query"], language="cypher")

                # Results & Insights
                st.subheader("Results & Insights")
                if out["query_results"]:
                    df = pd.DataFrame(out["query_results"])
                    bools = df.select_dtypes(include="bool").columns.tolist()
                    if bools:
                        with st.expander("Filter boolean columns"):
                            mask = pd.Series(True, index=df.index)
                            for c in bools:
                                sel = st.multiselect(c, [True, False], default=[True, False])
                                mask &= df[c].isin(sel)
                            df = df[mask]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No records returned.")
                st.markdown("### 💡 Engineer Report")
                st.markdown(out["insights"])

    # correctness radio
    if st.session_state.question_history:
        idx = len(st.session_state.question_history) - 1
        prev = st.session_state.answer_history[idx] if idx < len(st.session_state.answer_history) else 1
        choice = st.radio("Was the last answer correct?", ("Yes","No"),
                          index=0 if prev==1 else 1, key=f"correct_flag_{idx}")
        val = 1 if choice=="Yes" else 0
        if idx < len(st.session_state.answer_history):
            st.session_state.answer_history[idx] = val
        else:
            st.session_state.answer_history.append(val)

    # Persistency Chart expander
    if st.session_state.cost_history:
        # build df_cost
        qs    = st.session_state.question_history
        ts    = st.session_state.timestamp_history
        cs    = st.session_state.cost_history
        ans   = st.session_state.answer_history.copy()
        mods  = st.session_state.model_history.copy()
        temps = st.session_state.temp_history.copy()
        n     = len(qs)
        if len(ans)   < n: ans   += [None]*(n-len(ans))
        if len(mods)  < n: mods  += [None]*(n-len(mods))
        if len(temps) < n: temps += [None]*(n-len(temps))

        df_cost = pd.DataFrame({
            "timestamp":    ts,
            "question":     qs,
            "cost":         cs,
            "answer":       ans,
            "model":        mods,
            "temperature":  temps
        })
        df_cost["correctness"] = df_cost["answer"].map({1:"Correct",0:"Incorrect"})
        df_cost["question_timestamp"] = (
            df_cost["question"]
            + " | "
            + pd.to_datetime(df_cost["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        # flag cached runs (cost==0)
        df_cost["cached"] = df_cost["cost"] == 0

        with st.expander("Persistency Chart", expanded=True):
            fig = px.bar(
                df_cost,
                x="cost",
                y="question_timestamp",
                color="correctness",
                orientation="h",
                custom_data=["temperature","model","cached"],
                labels={
                    "cost": "Cost (USD)",
                    "question_timestamp": "Question & Timestamp",
                    "correctness": "Answer Correctness"
                },
                title="Persistency Chart: Cost per Request by Correctness",
                height=600,
                barmode="group",
                color_discrete_map={"Correct":"green","Incorrect":"red"}
            )
            fig.update_traces(
                hovertemplate=(
                    "Temperature: %{customdata[0]}<br>"
                    "Model: %{customdata[1]}<br>"
                    "Cached: %{customdata[2]}<extra></extra>"
                )
            )
            st.plotly_chart(fig, use_container_width=True)



with tabs[1]:
    st.header("Vectorise & Retrieve")
    st.markdown("Coming soon: embedding write & k-NN retrieval.")

