#!/usr/bin/env python3
import re
from typing import Dict, List, Tuple
import streamlit as st

# ------------------ Prompt ------------------
PROMPT = (
    "Identify the tone of this statement. Only choose from the options – "
    "'positive', 'negative', or 'neutral'. Provide the response as a single label. "
    "Pay attention to the slang usage of words too. Bear in mind negative words can "
    "have positive intention based on context. For eg: Shut up! Did you really buy me lunch?"
)

# ------------------ Lexicons & Helpers ------------------
POS_WORDS = {
    # standard
    "good","great","excellent","awesome","amazing","fantastic","love","like","liked","likes",
    "enjoy","enjoyed","wonderful","perfect","nice","sweet","thanks","thank","appreciate",
    "glad","happy","joy","thrilled","delighted","cool","wow",
    # slang/short
    "lit","fire","dope","goated","pog","based","legit","clutch","win","winning","w",
    "sick","insane","crazy","ridiculous","wild",  # context-sensitive, handled below
    "chef's","kiss","slaps","banger",
    # emojis/shortcuts handled separately but keep for completeness
}

NEG_WORDS = {
    # standard
    "bad","awful","terrible","horrible","hate","hated","dislike","disliked","worse","worst",
    "annoying","stupid","dumb","useless","broken","buggy","angry","mad","upset","sad",
    "disappointed","disappointing","cringe","gross","sucks","sucked","sucky","lame","meh",
    # slang
    "trash","mid","cap","salty","ratio","cope","downbad","flake","flaky","weak","fail","l",
    "wtf","smh","bs",
}

# words that *invert* polarity when nearby (and special idioms handled below)
NEGATORS = {
    "not","no","never","hardly","scarcely","barely","without","ain't","isn't","wasn't","don't",
    "doesn't","didn't","can't","couldn't","won't","wouldn't","shouldn't","lack","lacking"
}

# intensifiers / diminishers
INTENSIFIERS = {"very","really","so","super","totally","absolutely","insanely","crazy","hella","deadass","fr","for","real"}
DIMINISHERS = {"kinda","kind","of","sorta","sort","of","slightly","somewhat","a","bit","lowkey"}

# obvious positive emojis / neg emojis
POS_EMOJI = {"🙂","😊","😄","😃","😁","😍","🥰","❤️","💖","🔥","✨","👍","👏","🙌","😁","😂","🤣","😎","✅","💯"}
NEG_EMOJI = {"☹","🙁","😞","😠","😡","💔","👎","😢","😭","🤮","🤢","😤","😒","😓","😕","❌"}

# context words that usually indicate a positive event
POS_EVENT_WORDS = {
    "bought","buy","got","gift","gifting","surprise","treat","treated",
    "helped","promoted","passed","win","won","upgrade","raise","lunch","dinner","coffee"
}

WORD_RE = re.compile(r"[A-Za-z']+")
ELLIPSIS_RE = re.compile(r"\.\.\.$|…$")

def tokenize(text: str) -> List[str]:
    # keep punctuation to check for exclamations and ellipses later
    return re.findall(r"[A-Za-z']+|[!?…]|[:;]-?\)|\.\.\.|[^\sA-Za-z]", text)

def is_emoji(tok: str) -> bool:
    return any(ch in POS_EMOJI or ch in NEG_EMOJI for ch in tok)

def contains_any(s: str, vocab: set) -> bool:
    return any(w in s for w in vocab)

# ------------------ Core Heuristic Classifier ------------------
def classify_sentiment(text: str, return_evidence: bool=False):
    """
    Returns 'positive', 'negative', or 'neutral'.
    Heuristics: lexicon + negation + intensifiers + emoji + slang/context rules.
    """
    if not text or not text.strip():
        return ("neutral", {"score":0.0}) if return_evidence else "neutral"

    original = text
    lowered = text.lower()
    toks = tokenize(text)
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", lowered)]

    score = 0.0
    pos_hits, neg_hits, flipped_hits, intens_hits, dim_hits, emo_hits = [], [], [], [], [], []

    # --- Special idiom: "Shut up!" used as excited disbelief can be positive ---
    # Example: "Shut up! Did you really buy me lunch?"
    if "shut up" in lowered:
        after = lowered.split("shut up", 1)[1]
        # look for excited disbelief + positive event soon after
        if re.search(r"(did you really|for real|no way|fr)\b", after) and contains_any(after, POS_EVENT_WORDS):
            score += 1.5
            pos_hits.append("shut up! (excited disbelief)")
        else:
            # if used alone with insult, treat mildly negative
            if not contains_any(after, POS_EVENT_WORDS):
                score -= 0.2
                neg_hits.append("shut up (harsh)")

    # --- Token-level scoring with negation & modifiers in a small window ---
    window = 3
    for i, w in enumerate(words):
        base = 0
        w_is_pos = w in POS_WORDS
        w_is_neg = w in NEG_WORDS

        # treat "sick/crazy/insane/ridiculous" as positive if followed by positive noun/event or exclamation
        if w in {"sick","crazy","insane","ridiculous","wild"}:
            lookahead = " ".join(words[i+1:i+6])
            if contains_any(lookahead, POS_EVENT_WORDS|POS_WORDS) or "!" in original[i: i+80]:
                w_is_pos, w_is_neg = True, False

        if w_is_pos:
            base = 1.0
        elif w_is_neg:
            base = -1.0
        else:
            continue

        # check nearby negators
        window_start = max(0, i - window)
        window_slice = words[window_start:i]  # words before w
        has_negator = any(t in NEGATORS for t in window_slice)
        if has_negator:
            base *= -1
            flipped_hits.append(f"negated:{w}")

        # intensifiers/diminishers nearby
        has_intens = any(t in INTENSIFIERS for t in window_slice)
        has_dim = any(t in DIMINISHERS for t in window_slice)

        if has_intens:
            base *= 1.5
            intens_hits.append(w)
        if has_dim:
            base *= 0.65
            dim_hits.append(w)

        score += base
        (pos_hits if base > 0 else neg_hits).append(w)

    # --- emojis ---
    for ch in original:
        if ch in POS_EMOJI:
            score += 0.6
            emo_hits.append(ch)
        elif ch in NEG_EMOJI:
            score -= 0.6
            emo_hits.append(ch)

    # --- exclamation boosts (cap at 3) ---
    exclamations = original.count("!")
    if exclamations:
        score += min(exclamations, 3) * 0.2

    # --- ellipsis after a positive word can imply sarcasm → dampen positive ---
    if ELLIPSIS_RE.search(original.strip()):
        if any(w in pos_hits for w in ("great","awesome","amazing","perfect","nice")):
            score -= 0.4

    # --- common polarity flips: "not bad" (positive), "not good" (negative) ---
    if re.search(r"\bnot\s+bad\b", lowered):
        score += 1.0
        pos_hits.append("not bad")
    if re.search(r"\bnot\s+(so\s+)?good\b", lowered):
        score -= 1.0
        neg_hits.append("not good")

    # --- Label from score with small neutral band ---
    label = "neutral"
    if score >= 0.5:
        label = "positive"
    elif score <= -0.5:
        label = "negative"

    if return_evidence:
        return label, {
            "score": round(score, 3),
            "positive_hits": sorted(set(pos_hits)),
            "negative_hits": sorted(set(neg_hits)),
            "flipped_by_negation": sorted(set(flipped_hits)),
            "intensified": sorted(set(intens_hits)),
            "diminished": sorted(set(dim_hits)),
            "emoji": emo_hits,
        }
    return label

# ------------------ Streamlit UI ------------------
DEFAULT_TEXT = "Shut up! Did you really buy me lunch?"

st.set_page_config(page_title="Sentiment Classifier (Single Label)", page_icon="🧠", layout="centered")
st.title("🧠 Sentiment Classifier (Single Label)")
st.caption(PROMPT)

col1, col2 = st.columns([3, 2])
with col1:
    show_evidence = st.checkbox("Show evidence (debug)", value=False)
with col2:
    use_sample = st.toggle("Use sample", value=True)

uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])
if uploaded is not None:
    text = uploaded.read().decode("utf-8", errors="ignore")
else:
    text = DEFAULT_TEXT if use_sample else ""

text = st.text_area("Statement", value=text, height=180, placeholder="Type or paste a statement…")

if st.button("Classify", type="primary", use_container_width=True):
    if show_evidence:
        label, evidence = classify_sentiment(text, return_evidence=True)
    else:
        label = classify_sentiment(text, return_evidence=False)

    st.subheader("Result")
    st.success(label)

    if show_evidence:
        with st.expander("Evidence & scoring"):
            st.write(evidence)
