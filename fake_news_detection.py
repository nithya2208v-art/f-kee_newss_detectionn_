"""
Fake News / Misinformation Detection using a Deterministic Finite Automaton (DFA)
==============================================================================

A college mini-project (Theory of Computation / Automata Theory) that shows how a
DFA can be used to flag suspicious news headlines.

Pipeline
--------
    headline --> normalise --> tokenise --> map words to DFA symbols
             --> run the DFA --> final state --> Fake / Real

The DFA is the heart of the project. It is defined formally as the 5-tuple

    M = (Q, Sigma, delta, q0, F)

    Q     = {q0, q1, q2, q3, q4}          (set of states)
    Sigma = {S, C, U, R, N}               (input alphabet)
    delta = transition function (see TRANSITION_TABLE below)
    q0    = start state
    F     = {q3}                          (accepting state = "FAKE")

Usage
-----
    python fake_news_detection.py                    # full demo + interactive mode
    python fake_news_detection.py --headline "..."   # classify one headline
    python fake_news_detection.py --demo             # run built-in sample headlines
    python fake_news_detection.py --diagram images/dfa_diagram.png
    python fake_news_detection.py --help             # all options

Note: the file is organised in numbered SECTIONs. The Google Colab notebook
re-uses exactly these sections, so keep the "# SECTION n:" banners intact.
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# SECTION 2: DFA DEFINITION DATA (alphabet, states, transitions, lexicon)
# =============================================================================

# --- Input alphabet (Sigma) --------------------------------------------------
# Every word (or phrase) of a headline is converted into exactly one of these
# symbols before it is fed to the DFA.
SYM_SENSATIONAL = "S"   # exaggerated / emotional wording
SYM_CLICKBAIT = "C"     # clickbait or conspiracy phrase
SYM_URGENCY = "U"       # urgency / "share this now" pressure
SYM_CREDIBLE = "R"      # credibility marker (attribution to a source)
SYM_NEUTRAL = "N"       # any other, ordinary word

ALPHABET: Tuple[str, ...] = (
    SYM_SENSATIONAL,
    SYM_CLICKBAIT,
    SYM_URGENCY,
    SYM_CREDIBLE,
    SYM_NEUTRAL,
)

SYMBOL_DESCRIPTIONS: Dict[str, str] = {
    SYM_SENSATIONAL: "Sensational / exaggerated wording",
    SYM_CLICKBAIT: "Clickbait / conspiracy phrase",
    SYM_URGENCY: "Urgency / share-bait",
    SYM_CREDIBLE: "Credibility marker (source / attribution)",
    SYM_NEUTRAL: "Neutral word (no signal)",
}

# --- States (Q) --------------------------------------------------------------
STATE_START = "q0"      # start state, no net suspicious signal
STATE_LOW = "q1"        # one suspicious signal
STATE_HIGH = "q2"       # two suspicious signals (borderline)
STATE_FAKE = "q3"       # accepting state: suspicion threshold reached
STATE_CREDIBLE = "q4"   # a credibility marker was seen

STATE_DESCRIPTIONS: Dict[str, str] = {
    STATE_START: "START / NEUTRAL - no net suspicious signal",
    STATE_LOW: "LOW SUSPICION - one suspicious signal",
    STATE_HIGH: "HIGH SUSPICION - two signals (borderline, still Real)",
    STATE_FAKE: "FAKE - suspicion threshold reached (accepting, absorbing)",
    STATE_CREDIBLE: "CREDIBLE - credibility marker seen",
}

ACCEPTING_STATES = frozenset({STATE_FAKE})

# --- Transition function (delta) --------------------------------------------
# TRANSITION_TABLE[current_state][input_symbol] = next_state
#
# Intuition:
#   * S / C push the automaton one step towards FAKE,
#   * U (urgency) is a stronger signal and pushes it two steps,
#   * R (credibility) pulls it back one step (or into the CREDIBLE state),
#   * N (neutral) never changes the state (self-loop),
#   * q3 (FAKE) is absorbing: once reached it can never be left.
TRANSITION_TABLE: Dict[str, Dict[str, str]] = {
    #        S      C      U      R      N
    "q0": {"S": "q1", "C": "q1", "U": "q2", "R": "q4", "N": "q0"},
    "q1": {"S": "q2", "C": "q2", "U": "q3", "R": "q0", "N": "q1"},
    "q2": {"S": "q3", "C": "q3", "U": "q3", "R": "q1", "N": "q2"},
    "q3": {"S": "q3", "C": "q3", "U": "q3", "R": "q3", "N": "q3"},
    "q4": {"S": "q0", "C": "q0", "U": "q1", "R": "q4", "N": "q4"},
}

# --- Lexicon: words / phrases that map to each non-neutral symbol -----------
# Phrases are matched word-by-word (longest phrase first). Apostrophes and
# hyphens are ignored, so "don't" is written "dont" and "cover-up" "cover up".
LEXICON: Dict[str, Tuple[str, ...]] = {
    SYM_SENSATIONAL: (
        "shocking", "unbelievable", "miracle", "miraculous", "secret", "secrets",
        "exposed", "exposes", "bombshell", "stunning", "horrifying",
        "mind blowing", "jaw dropping", "100%", "guaranteed", "instantly",
        "cures", "cure all", "overnight", "!",
    ),
    SYM_CLICKBAIT: (
        "hoax", "conspiracy", "cover up", "coverup", "covered up", "covering up",
        "covering it up", "illuminati", "chemtrails", "big pharma",
        "mainstream media", "deep state", "new world order", "mind control",
        "doctors hate", "you wont believe", "dont want you to know",
        "wont tell you", "hidden truth", "hiding the truth", "the truth about",
        "wake up",
    ),
    SYM_URGENCY: (
        "urgent", "share this", "share now", "share with everyone",
        "forward this", "spread the word", "tell everyone", "must watch",
        "must share", "before its deleted", "before they delete",
        "before it gets deleted", "send this to", "retweet",
    ),
    SYM_CREDIBLE: (
        "according to", "officials", "official", "ministry", "announced",
        "announces", "announce", "reported", "reports", "report", "researchers",
        "study", "published", "confirmed", "confirms", "spokesperson",
        "statement", "survey", "university", "court", "parliament",
        "commission", "statistics", "percent", "data shows", "said", "says",
    ),
}

# Sample headlines used by the demo mode / notebook (not part of dataset.csv).
SAMPLE_HEADLINES: Tuple[str, ...] = (
    "SHOCKING secret cure exposed! Share this before it's deleted",
    "Government announces new metro line, officials said",
    "Researchers publish study on air pollution in Indian cities",
    "Miracle weight loss pill guaranteed to work, doctors hate it",
    "Shocking rise in onion prices worries shoppers, ministry reports",
    "Local school wins state level football tournament",
)

LABELS: Tuple[str, str] = ("Fake", "Real")
MAX_HEADLINE_LENGTH = 300
MIN_HEADLINE_WORDS = 3
QUIT_COMMANDS = ("quit", "exit", "q")


# =============================================================================
# SECTION 3: TEXT PROCESSING (validation, normalisation, tokenisation)
# =============================================================================
class InvalidHeadlineError(ValueError):
    """Raised when a headline cannot be processed (empty, too short, etc.)."""


@dataclass(frozen=True)
class Token:
    """A word or phrase of the headline together with its DFA input symbol."""

    text: str     # the (normalised) word or phrase that was matched
    symbol: str   # one of the symbols in ALPHABET


def normalize_text(text: str) -> str:
    """Convert a headline to a clean, lower-case form that is easy to tokenise.

    * lower-cases everything,
    * removes apostrophes ("don't" -> "dont"),
    * turns runs of "!" into a single stand-alone "!" token,
    * replaces every other punctuation mark (including hyphens) with a space,
    * collapses repeated whitespace.
    """
    text = text.lower()
    text = re.sub(r"['\u2018\u2019`]", "", text)    # drop apostrophes
    text = re.sub(r"!+", " ! ", text)               # "!!!" -> single "!" token
    text = re.sub(r"[^a-z0-9%!\s]", " ", text)      # punctuation / hyphens -> space
    return re.sub(r"\s+", " ", text).strip()


def validate_headline(headline: object) -> str:
    """Check that the user input is usable and return the stripped headline.

    Raises:
        InvalidHeadlineError: if the input is not text, is empty, too long,
            has no English letters, or has fewer than MIN_HEADLINE_WORDS words.
    """
    if not isinstance(headline, str):
        raise InvalidHeadlineError("The headline must be text.")

    cleaned = headline.strip()
    if not cleaned:
        raise InvalidHeadlineError("The headline is empty. Please type a news headline.")
    if len(cleaned) > MAX_HEADLINE_LENGTH:
        raise InvalidHeadlineError(
            f"The headline is too long ({len(cleaned)} characters). "
            f"Maximum allowed is {MAX_HEADLINE_LENGTH}."
        )
    if not re.search(r"[A-Za-z]", cleaned):
        raise InvalidHeadlineError("The headline must contain English letters, not only digits or symbols.")

    word_count = len(re.findall(r"[a-z0-9%]+", normalize_text(cleaned)))
    if word_count < MIN_HEADLINE_WORDS:
        raise InvalidHeadlineError(
            f"The headline is too short ({word_count} word(s)). "
            f"Please enter at least {MIN_HEADLINE_WORDS} words."
        )
    return cleaned


def build_phrase_index(lexicon: Dict[str, Tuple[str, ...]]) -> List[Tuple[Tuple[str, ...], str]]:
    """Turn the lexicon into a list of (phrase_words, symbol), longest phrase first.

    Sorting by length makes the tokenizer prefer "share this" over a shorter
    match, i.e. it performs *longest-match* scanning.
    """
    index: List[Tuple[Tuple[str, ...], str]] = []
    seen: Dict[Tuple[str, ...], str] = {}
    for symbol, phrases in lexicon.items():
        for phrase in phrases:
            words = tuple(normalize_text(phrase).split())
            if words in seen:
                raise ValueError(f"Lexicon conflict: '{phrase}' appears under both {seen[words]} and {symbol}.")
            seen[words] = symbol
            index.append((words, symbol))
    index.sort(key=lambda item: len(item[0]), reverse=True)
    return index


# Built once at import time and re-used for every headline.
PHRASE_INDEX = build_phrase_index(LEXICON)


def tokenize(headline: str, phrase_index: Optional[List[Tuple[Tuple[str, ...], str]]] = None) -> List[Token]:
    """Split a headline into tokens and assign each one a DFA input symbol.

    Words that are not found in the lexicon become neutral tokens (symbol N).
    """
    phrase_index = PHRASE_INDEX if phrase_index is None else phrase_index
    words = normalize_text(headline).split()

    tokens: List[Token] = []
    position = 0
    while position < len(words):
        for phrase_words, symbol in phrase_index:
            end = position + len(phrase_words)
            if tuple(words[position:end]) == phrase_words:
                tokens.append(Token(" ".join(phrase_words), symbol))
                position = end
                break
        else:  # no lexicon phrase starts here -> ordinary word
            tokens.append(Token(words[position], SYM_NEUTRAL))
            position += 1
    return tokens


# =============================================================================
# SECTION 4: THE DFA ENGINE
# =============================================================================
class DFA:
    """A generic Deterministic Finite Automaton M = (Q, Sigma, delta, q0, F).

    The class knows nothing about news: it only stores the five components and
    runs a sequence of symbols through the transition function.
    """

    def __init__(
        self,
        states: Sequence[str],
        alphabet: Sequence[str],
        transitions: Dict[Tuple[str, str], str],
        start_state: str,
        accepting_states: Sequence[str],
        state_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.states: Tuple[str, ...] = tuple(states)
        self.alphabet: Tuple[str, ...] = tuple(alphabet)
        self.transitions: Dict[Tuple[str, str], str] = dict(transitions)
        self.start_state: str = start_state
        self.accepting_states: frozenset = frozenset(accepting_states)
        self.state_descriptions: Dict[str, str] = dict(state_descriptions or {})
        self.validate()

    def validate(self) -> None:
        """Make sure the five components really form a valid (complete) DFA."""
        if len(set(self.states)) != len(self.states):
            raise ValueError("DFA states must be unique.")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("DFA alphabet symbols must be unique.")
        if self.start_state not in self.states:
            raise ValueError(f"Start state '{self.start_state}' is not in the set of states.")
        if not self.accepting_states.issubset(self.states):
            raise ValueError("Every accepting state must be a member of the set of states.")

        # A DFA needs exactly ONE transition for every (state, symbol) pair.
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transitions:
                    raise ValueError(f"Missing transition for ({state}, {symbol}).")
        for (state, symbol), target in self.transitions.items():
            if state not in self.states or symbol not in self.alphabet or target not in self.states:
                raise ValueError(f"Invalid transition ({state}, {symbol}) -> {target}.")

    def step(self, state: str, symbol: str) -> str:
        """Apply the transition function delta(state, symbol)."""
        if symbol not in self.alphabet:
            raise ValueError(f"Symbol '{symbol}' is not in the DFA alphabet {self.alphabet}.")
        return self.transitions[(state, symbol)]

    def run(self, symbols: Sequence[str]) -> Tuple[str, List[Tuple[str, str, str]]]:
        """Feed all symbols to the DFA.

        Returns:
            (final_state, path) where path is a list of
            (source_state, symbol, target_state) triples, one per symbol.
        """
        state = self.start_state
        path: List[Tuple[str, str, str]] = []
        for symbol in symbols:
            next_state = self.step(state, symbol)
            path.append((state, symbol, next_state))
            state = next_state
        return state, path

    def accepts(self, symbols: Sequence[str]) -> bool:
        """True if the DFA ends in an accepting state after reading symbols."""
        final_state, _ = self.run(symbols)
        return final_state in self.accepting_states


def build_news_dfa() -> DFA:
    """Create the fake-news DFA from the constants in SECTION 2."""
    transitions = {
        (state, symbol): target
        for state, row in TRANSITION_TABLE.items()
        for symbol, target in row.items()
    }
    return DFA(
        states=sorted(TRANSITION_TABLE),
        alphabet=ALPHABET,
        transitions=transitions,
        start_state=STATE_START,
        accepting_states=ACCEPTING_STATES,
        state_descriptions=STATE_DESCRIPTIONS,
    )


def transition_table(dfa: DFA) -> pd.DataFrame:
    """Return delta as a pandas DataFrame (rows = states, columns = symbols)."""
    data = {
        symbol: [dfa.transitions[(state, symbol)] for state in dfa.states]
        for symbol in dfa.alphabet
    }
    table = pd.DataFrame(data, index=list(dfa.states))
    table.index.name = "state"
    return table


# =============================================================================
# SECTION 5: DETECTION (headline -> prediction)
# =============================================================================
@dataclass(frozen=True)
class Transition:
    """One step of the DFA run, kept for the explanation shown to the user."""

    step: int       # 1-based position in the token stream
    token: str      # word / phrase that was read
    symbol: str     # DFA input symbol
    source: str     # state before reading the symbol
    target: str     # state after reading the symbol


@dataclass(frozen=True)
class Prediction:
    """Everything produced while classifying one headline."""

    headline: str
    tokens: Tuple[Token, ...]
    transitions: Tuple[Transition, ...]
    final_state: str
    is_fake: bool

    @property
    def label(self) -> str:
        """'Fake' if the DFA stopped in an accepting state, otherwise 'Real'."""
        return "Fake" if self.is_fake else "Real"

    @property
    def keywords(self) -> List[Token]:
        """Only the tokens that carry a signal (everything except neutral words)."""
        return [token for token in self.tokens if token.symbol != SYM_NEUTRAL]


def detect(headline: str, dfa: Optional[DFA] = None) -> Prediction:
    """Classify a headline as Fake or Real using the DFA.

    Raises:
        InvalidHeadlineError: if the headline fails validation.
    """
    dfa = dfa or build_news_dfa()
    cleaned = validate_headline(headline)

    tokens = tokenize(cleaned)
    final_state, path = dfa.run([token.symbol for token in tokens])

    transitions = tuple(
        Transition(step=index, token=token.text, symbol=symbol, source=source, target=target)
        for index, (token, (source, symbol, target)) in enumerate(zip(tokens, path), start=1)
    )
    return Prediction(
        headline=cleaned,
        tokens=tuple(tokens),
        transitions=transitions,
        final_state=final_state,
        is_fake=final_state in dfa.accepting_states,
    )


# =============================================================================
# SECTION 6: REPORTING (pretty text output)
# =============================================================================
REPORT_WIDTH = 72


def format_dfa_definition(dfa: DFA) -> str:
    """Return a readable description of the DFA (states, alphabet, delta ...)."""
    lines = [
        "=" * REPORT_WIDTH,
        "DFA DEFINITION   M = (Q, Sigma, delta, q0, F)",
        "=" * REPORT_WIDTH,
        f"Q (states)          : {{{', '.join(dfa.states)}}}",
        f"Sigma (alphabet)    : {{{', '.join(dfa.alphabet)}}}",
        f"q0 (start state)    : {dfa.start_state}",
        f"F (accepting states): {{{', '.join(sorted(dfa.accepting_states))}}}",
        "",
        "States:",
    ]
    for state in dfa.states:
        lines.append(f"  {state} : {dfa.state_descriptions.get(state, '')}")
    lines.append("")
    lines.append("Alphabet symbols:")
    for symbol in dfa.alphabet:
        lines.append(f"  {symbol} : {SYMBOL_DESCRIPTIONS.get(symbol, '')}")
    lines.append("")
    lines.append("Transition function delta (row = current state, column = input symbol):")
    lines.append(transition_table(dfa).to_string())
    lines.append("=" * REPORT_WIDTH)
    return "\n".join(lines)


def format_report(prediction: Prediction, dfa: DFA, show_neutral: bool = False) -> str:
    """Build the explanation for one headline.

    Order of the report (as required by the project specification):
        1. detected keywords
        2. DFA state transitions
        3. final DFA state
        4. the prediction
    """
    lines = [
        "=" * REPORT_WIDTH,
        f"HEADLINE: {prediction.headline}",
        "=" * REPORT_WIDTH,
        "STEP 1 - Detected keywords",
    ]

    # --- Step 1: keywords ---------------------------------------------------
    if prediction.keywords:
        for number, token in enumerate(prediction.keywords, start=1):
            meaning = SYMBOL_DESCRIPTIONS[token.symbol]
            lines.append(f'  {number}. "{token.text}"  ->  {token.symbol}  ({meaning})')
    else:
        lines.append("  (none - every word in this headline is neutral)")

    # --- Step 2: state transitions -----------------------------------------
    lines.append("")
    lines.append(f"STEP 2 - DFA state transitions (start state = {dfa.start_state})")
    hidden_neutral = 0
    for move in prediction.transitions:
        if move.symbol == SYM_NEUTRAL and not show_neutral:
            hidden_neutral += 1   # neutral words only cause self-loops
            continue
        lines.append(
            f'  [{move.step:>2}] {move.source} --{move.symbol}--> {move.target}'
            f'    (read "{move.token}")'
        )
    if hidden_neutral:
        lines.append(f"  ...  {hidden_neutral} neutral word(s) caused self-loops (hidden; use verbose mode to show)")

    # --- Step 3: final state -----------------------------------------------
    is_accepting = prediction.final_state in dfa.accepting_states
    lines.append("")
    lines.append("STEP 3 - Final DFA state")
    lines.append(f"  Final state     : {prediction.final_state}")
    lines.append(f"  Meaning         : {dfa.state_descriptions.get(prediction.final_state, '')}")
    lines.append(f"  Accepting state : {'YES' if is_accepting else 'NO'}")

    # --- Step 4: prediction -------------------------------------------------
    lines.append("-" * REPORT_WIDTH)
    reason = "accepting" if is_accepting else "non-accepting"
    lines.append(f"PREDICTION: {prediction.label.upper()}   (DFA ended in {reason} state {prediction.final_state})")
    lines.append("=" * REPORT_WIDTH)
    return "\n".join(lines)


def print_report(prediction: Prediction, dfa: DFA, show_neutral: bool = False) -> None:
    """Print the report returned by format_report()."""
    print(format_report(prediction, dfa, show_neutral=show_neutral))


def interactive_mode(dfa: DFA, show_neutral: bool = False) -> None:
    """Keep asking the user for headlines until they type 'quit' (or press Ctrl+C)."""
    print("\nINTERACTIVE MODE - type a news headline and press Enter.")
    print("Type 'quit' to stop.")
    while True:
        try:
            text = input("\nEnter a news headline: ")
        except (EOFError, KeyboardInterrupt):
            print("\nInput closed. Goodbye!")
            return
        if text.strip().lower() in QUIT_COMMANDS:
            print("Goodbye!")
            return
        try:
            prediction = detect(text, dfa)
        except InvalidHeadlineError as error:
            print(f"[Invalid input] {error}")
            continue
        print_report(prediction, dfa, show_neutral=show_neutral)


# =============================================================================
# SECTION 7: DATASET LOADING AND EVALUATION (pandas)
# =============================================================================
def load_dataset(path) -> pd.DataFrame:
    """Read the labelled headlines CSV with pandas and validate it.

    The file must contain the columns 'headline' and 'label' (Fake / Real).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty, malformed or has wrong labels.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: '{path}'.")

    try:
        data = pd.read_csv(path, encoding="utf-8")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read '{path}' as a CSV file: {error}") from error

    data.columns = data.columns.str.strip().str.lower()
    missing = {"headline", "label"} - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}.")

    data = data[["headline", "label"]].dropna()
    data["headline"] = data["headline"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.capitalize()

    bad_labels = sorted(set(data["label"]) - set(LABELS))
    if bad_labels:
        raise ValueError(f"Unknown label(s) in dataset: {bad_labels}. Allowed labels: {list(LABELS)}.")

    data = data.drop_duplicates(subset="headline").reset_index(drop=True)
    if data.empty:
        raise ValueError("The dataset contains no usable rows.")
    return data


def evaluate_dataset(data: pd.DataFrame, dfa: DFA) -> pd.DataFrame:
    """Run the DFA on every dataset headline and compare with the true label.

    Returns a copy of the data with the extra columns:
    'predicted', 'final_state' and 'correct'.
    """
    predicted: List[str] = []
    final_states: List[str] = []
    for headline in data["headline"]:
        try:
            result = detect(headline, dfa)
            predicted.append(result.label)
            final_states.append(result.final_state)
        except InvalidHeadlineError:
            predicted.append("Invalid")    # counted as a wrong prediction
            final_states.append("-")

    results = data.copy()
    results["predicted"] = predicted
    results["final_state"] = final_states
    results["correct"] = results["label"] == results["predicted"]
    return results


def _safe_divide(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator else 0.0


def format_evaluation(results: pd.DataFrame) -> str:
    """Summarise accuracy, confusion matrix and precision / recall for 'Fake'."""
    total = len(results)
    correct = int(results["correct"].sum())

    matrix = (
        pd.crosstab(results["label"], results["predicted"])
        .reindex(index=list(LABELS), columns=list(LABELS), fill_value=0)
    )
    true_pos = int(matrix.loc["Fake", "Fake"])
    false_pos = int(matrix.loc["Real", "Fake"])
    false_neg = int(matrix.loc["Fake", "Real"])
    precision = _safe_divide(true_pos, true_pos + false_pos)
    recall = _safe_divide(true_pos, true_pos + false_neg)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)

    lines = [
        "=" * REPORT_WIDTH,
        "DATASET EVALUATION",
        "=" * REPORT_WIDTH,
        f"Total headlines : {total}  ({int((results['label'] == 'Fake').sum())} Fake, "
        f"{int((results['label'] == 'Real').sum())} Real)",
        f"Correct         : {correct}",
        f"Accuracy        : {_safe_divide(correct, total) * 100:.2f} %",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        matrix.to_string(),
        "",
        f"Fake class -> precision: {precision:.2f}   recall: {recall:.2f}   F1-score: {f1_score:.2f}",
    ]

    wrong = results[~results["correct"]]
    if not wrong.empty:
        lines.append("")
        lines.append("Misclassified headlines:")
        for _, row in wrong.iterrows():
            lines.append(f"  - [{row['label']} -> {row['predicted']}, state {row['final_state']}] {row['headline']}")
    lines.append("=" * REPORT_WIDTH)
    return "\n".join(lines)


# =============================================================================
# SECTION 8: DFA DIAGRAM (matplotlib, optional)
# =============================================================================
# Hand-tuned node positions for the 5-state fake-news DFA (x, y).
_NODE_POSITIONS: Dict[str, Tuple[float, float]] = {
    "q0": (0.0, 0.0),
    "q1": (4.6, 0.0),
    "q2": (9.2, 0.0),
    "q3": (13.8, 0.0),
    "q4": (0.0, -3.6),
}
# Short caption written under every state in the diagram.
_NODE_CAPTIONS: Dict[str, str] = {
    "q0": "START", "q1": "LOW", "q2": "HIGH", "q3": "FAKE", "q4": "CREDIBLE",
}
# Direction (degrees) in which each self-loop is drawn.
_LOOP_ANGLES: Dict[str, float] = {"q0": 90, "q1": 90, "q2": 90, "q3": 90, "q4": 180}
# Curvature of the arrows between states (positive bends below for left->right).
_EDGE_CURVES: Dict[Tuple[str, str], float] = {
    ("q0", "q2"): -0.55,   # long arc above the row
    ("q1", "q3"): 0.40,    # long arc below the row
    ("q4", "q1"): 0.0,     # straight diagonal
}


def draw_dfa_diagram(dfa: DFA, output_path: Optional[str] = None, show: bool = True) -> None:
    """Draw the state diagram of the fake-news DFA with matplotlib.

    Args:
        dfa: the automaton to draw (must be the 5-state news DFA).
        output_path: if given, the figure is saved to this PNG file.
        show: display the figure (inline in a notebook / window in a script).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
    except ImportError as error:
        raise RuntimeError("matplotlib is required for the diagram. Install it with: pip install matplotlib") from error

    if set(dfa.states) != set(_NODE_POSITIONS):
        raise ValueError("draw_dfa_diagram() only supports the 5-state news DFA (q0 - q4).")

    radius = 0.62
    fig, axis = plt.subplots(figsize=(15, 8))
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_xlim(-3.2, 16.6)
    axis.set_ylim(-6.2, 3.7)

    # --- draw the states ----------------------------------------------------
    fill_colours = {"q0": "#dbeafe", "q1": "#fef9c3", "q2": "#fed7aa", "q3": "#fecaca", "q4": "#bbf7d0"}
    circles: Dict[str, Circle] = {}
    for state, (x, y) in _NODE_POSITIONS.items():
        circle = Circle((x, y), radius, facecolor=fill_colours[state], edgecolor="#1f2937", linewidth=2, zorder=3)
        axis.add_patch(circle)
        circles[state] = circle
        if state in dfa.accepting_states:   # accepting state = double circle
            axis.add_patch(Circle((x, y), radius - 0.1, fill=False, edgecolor="#1f2937", linewidth=2, zorder=4))
        axis.text(x, y, state, ha="center", va="center", fontsize=15, fontweight="bold", zorder=5)
        axis.text(x, y - radius - 0.28, _NODE_CAPTIONS[state], ha="center", va="center",
                  fontsize=10, color="#374151", zorder=5)

    # --- start arrow --------------------------------------------------------
    start_x, start_y = _NODE_POSITIONS[dfa.start_state]
    axis.add_patch(FancyArrowPatch((start_x - 2.6, start_y), (start_x - radius, start_y), arrowstyle="-|>",
                                   mutation_scale=22, linewidth=2, color="#111827", zorder=2))
    axis.text(start_x - 2.6, start_y + 0.35, "start", ha="left", fontsize=11, color="#111827")

    # --- group the transitions: several symbols can share one arrow ----------
    edges: Dict[Tuple[str, str], List[str]] = {}
    for (source, symbol), target in dfa.transitions.items():
        edges.setdefault((source, target), []).append(symbol)

    for (source, target), symbols in edges.items():
        ordered = [symbol for symbol in dfa.alphabet if symbol in symbols]
        label = "any symbol" if len(ordered) == len(dfa.alphabet) else ", ".join(ordered)
        colour = "#b91c1c" if target == "q3" else "#374151"

        if source == target:                       # self-loop
            angle = _LOOP_ANGLES[source]
            x, y = _NODE_POSITIONS[source]
            theta_a, theta_b = math.radians(angle + 32), math.radians(angle - 32)
            start = (x + radius * math.cos(theta_a), y + radius * math.sin(theta_a))
            end = (x + radius * math.cos(theta_b), y + radius * math.sin(theta_b))
            axis.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=-2.3", arrowstyle="-|>",
                                           mutation_scale=16, linewidth=1.6, color=colour, zorder=2))
            label_x = x + (radius + 1.0) * math.cos(math.radians(angle))
            label_y = y + (radius + 1.0) * math.sin(math.radians(angle))
        else:                                      # arrow between two different states
            has_reverse = (target, source) in edges
            curve = _EDGE_CURVES.get((source, target), 0.25 if has_reverse else 0.0)
            (x1, y1), (x2, y2) = _NODE_POSITIONS[source], _NODE_POSITIONS[target]
            axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={curve}",
                                           arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=colour,
                                           patchA=circles[source], patchB=circles[target],
                                           shrinkA=0, shrinkB=0, zorder=2))
            # mid-point of the quadratic Bezier curve used by "arc3"
            label_x = (x1 + x2) / 2 + 0.5 * curve * (y2 - y1)
            label_y = (y1 + y2) / 2 - 0.5 * curve * (x2 - x1)

        axis.text(label_x, label_y, label, ha="center", va="center", fontsize=11, color=colour,
                  fontweight="bold", zorder=6,
                  bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})

    # --- title and legend ---------------------------------------------------
    axis.set_title("Fake News Detection DFA  -  state diagram", fontsize=17, fontweight="bold", pad=10)
    legend = "   ".join(f"{symbol} = {SYMBOL_DESCRIPTIONS[symbol]}" for symbol in dfa.alphabet[:3])
    legend2 = "   ".join(f"{symbol} = {SYMBOL_DESCRIPTIONS[symbol]}" for symbol in dfa.alphabet[3:])
    axis.text(6.7, -5.35, legend, ha="center", fontsize=10, color="#111827")
    axis.text(6.7, -5.85, legend2 + "      (double circle = accepting state)", ha="center", fontsize=10,
              color="#111827")

    fig.tight_layout()
    if output_path:
        target_file = Path(output_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_file, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# SECTION 9: COMMAND-LINE INTERFACE (VS Code / terminal only)
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line options."""
    default_dataset = Path(__file__).resolve().parent / "dataset.csv"
    parser = argparse.ArgumentParser(
        description="Fake news / misinformation detection using a Deterministic Finite Automaton (DFA).",
    )
    parser.add_argument("--dataset", default=str(default_dataset), help="path to the labelled CSV dataset")
    parser.add_argument("--headline", help="classify a single headline and exit")
    parser.add_argument("--demo", action="store_true", help="classify the built-in sample headlines and exit")
    parser.add_argument("--diagram", metavar="PNG_PATH", help="save the DFA state diagram to a PNG file and exit")
    parser.add_argument("--no-interactive", action="store_true", help="do not ask for headlines after the evaluation")
    parser.add_argument("--verbose", action="store_true", help="also show the neutral-word self-loop transitions")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point. Returns a process exit code (0 = success)."""
    args = build_arg_parser().parse_args(argv)
    dfa = build_news_dfa()

    # --- single-purpose modes ----------------------------------------------
    if args.diagram:
        try:
            draw_dfa_diagram(dfa, output_path=args.diagram, show=False)
        except (RuntimeError, ValueError, OSError) as error:
            print(f"[Error] Could not create the diagram: {error}")
            return 1
        print(f"DFA diagram saved to: {args.diagram}")
        return 0

    if args.headline is not None:
        try:
            print_report(detect(args.headline, dfa), dfa, show_neutral=args.verbose)
        except InvalidHeadlineError as error:
            print(f"[Invalid input] {error}")
            return 1
        return 0

    if args.demo:
        for headline in SAMPLE_HEADLINES:
            print_report(detect(headline, dfa), dfa, show_neutral=args.verbose)
            print()
        return 0

    # --- full run: DFA definition, dataset evaluation, interactive mode ------
    print("FAKE NEWS / MISINFORMATION DETECTION USING A DFA")
    print(format_dfa_definition(dfa))

    try:
        data = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as error:
        print(f"\n[Warning] {error}\nSkipping the dataset evaluation.")
    else:
        print(f"\nLoaded {len(data)} headlines from '{args.dataset}'.")
        print(format_evaluation(evaluate_dataset(data, dfa)))

    if not args.no_interactive:
        interactive_mode(dfa, show_neutral=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
