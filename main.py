from __future__ import annotations

import argparse
import difflib
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "DATA"
ATP_DATA_DIR = DATA_ROOT / "tennis_atp-master"
ODDS_MEN_DIR = DATA_ROOT / "TennisDataMan"
ODDS_WOMEN_DIR = DATA_ROOT / "TennisDataWomen"
BASE_ELO = 1500.0
DEFAULT_REST_DAYS = 30.0
DEFAULT_HISTORY_YEARS = 20
DEFAULT_BACKTEST_YEARS = 5
FORM_ALPHA = 0.25
H2H_PRIOR = 1.0
ROUND_DAY_OFFSETS = {
    "RR": 0,
    "BR": 0,
    "ER": 0,
    "R128": 0,
    "R64": 1,
    "R32": 2,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}
GRAND_SLAM_ROUND_DAY_OFFSETS = {
    "R128": 0,
    "R64": 2,
    "R32": 4,
    "R16": 6,
    "QF": 8,
    "SF": 10,
    "F": 13,
}

MATCH_COLUMNS = [
    "tourney_name",
    "tourney_date",
    "match_num",
    "surface",
    "tourney_level",
    "best_of",
    "round",
    "winner_id",
    "winner_name",
    "winner_ht",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_name",
    "loser_ht",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "score",
    "w_svpt",
    "w_ace",
    "w_df",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_bpSaved",
    "w_bpFaced",
    "l_svpt",
    "l_ace",
    "l_df",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_bpSaved",
    "l_bpFaced",
]

NUMERIC_FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "height_diff",
    "form_ewm_diff",
    "h2h_diff",
    "days_since_last_match_diff",
    "ace_rate_ewm_diff",
    "df_rate_ewm_diff",
    "first_in_rate_ewm_diff",
    "first_serve_win_rate_ewm_diff",
    "second_serve_win_rate_ewm_diff",
    "break_save_rate_ewm_diff",
    "recent_retirement_ewm_diff",
    "market_prob_diff",
    "market_overround",
]

CATEGORICAL_FEATURES = ["surface", "tourney_level", "best_of"]
MODEL_NAMES = ["logistic", "random_forest", "hist_gradient_boosting"]


@dataclass
class PlayerState:
    elo: float = BASE_ELO
    surface_elos: dict[str, float] = field(default_factory=dict)
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    form_ewm: float = 0.5
    ace_rate_ewm: float = 0.0
    df_rate_ewm: float = 0.0
    first_in_rate_ewm: float = 0.0
    first_serve_win_rate_ewm: float = 0.0
    second_serve_win_rate_ewm: float = 0.0
    break_save_rate_ewm: float = 0.0
    recent_retirement_ewm: float = 0.0
    match_count: int = 0
    last_match_date: pd.Timestamp | None = None
    last_rank: float | None = None
    last_rank_points: float | None = None
    last_age: float | None = None
    last_height: float | None = None
    display_name: str | None = None

    def surface_elo(self, surface: str) -> float:
        return self.surface_elos.get(surface, BASE_ELO)

    def set_surface_elo(self, surface: str, value: float) -> None:
        self.surface_elos[surface] = value

    def form_last_5(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    def form_value(self) -> float:
        return self.form_ewm

    def rest_days(self, match_date: pd.Timestamp) -> float:
        if self.last_match_date is None:
            return DEFAULT_REST_DAYS
        return float((match_date - self.last_match_date).days)

    def age_on(self, match_date: pd.Timestamp) -> float | None:
        if self.last_age is None:
            return None
        if self.last_match_date is None:
            return self.last_age
        delta_days = max(0, (match_date - self.last_match_date).days)
        return self.last_age + delta_days / 365.25

    def update_profile(
        self,
        *,
        display_name: str,
        rank: float | None,
        rank_points: float | None,
        age: float | None,
        height: float | None,
        match_date: pd.Timestamp,
    ) -> None:
        self.display_name = display_name
        if pd.notna(rank):
            self.last_rank = float(rank)
        if pd.notna(rank_points):
            self.last_rank_points = float(rank_points)
        if pd.notna(age):
            self.last_age = float(age)
        if pd.notna(height):
            self.last_height = float(height)
        self.last_match_date = match_date

    def update_form(self, result: int) -> None:
        self.form_ewm = FORM_ALPHA * result + (1.0 - FORM_ALPHA) * self.form_ewm
        self.recent_results.append(result)
        self.match_count += 1

    def update_serve_stats(
        self,
        *,
        ace_rate: float | None,
        df_rate: float | None,
        first_in_rate: float | None,
        first_serve_win_rate: float | None,
        second_serve_win_rate: float | None,
        break_save_rate: float | None,
        retired: float,
    ) -> None:
        for attr, value in [
            ("ace_rate_ewm", ace_rate),
            ("df_rate_ewm", df_rate),
            ("first_in_rate_ewm", first_in_rate),
            ("first_serve_win_rate_ewm", first_serve_win_rate),
            ("second_serve_win_rate_ewm", second_serve_win_rate),
            ("break_save_rate_ewm", break_save_rate),
        ]:
            if value is None:
                continue
            current = getattr(self, attr)
            setattr(self, attr, FORM_ALPHA * value + (1.0 - FORM_ALPHA) * current)
        self.recent_retirement_ewm = FORM_ALPHA * retired + (1.0 - FORM_ALPHA) * self.recent_retirement_ewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and use a tennis match prediction model on ATP match history."
    )
    subparsers = parser.add_subparsers(dest="command")

    interactive_parser = subparsers.add_parser("interactive", help="Start an interactive prompt UI.")
    add_common_history_args(interactive_parser)

    backtest_parser = subparsers.add_parser(
        "backtest",
        aliases=["evaluate"],
        help="Backtest the model on the most recent years.",
    )
    add_common_history_args(backtest_parser)
    backtest_parser.add_argument("--test-years", type=int, default=DEFAULT_BACKTEST_YEARS)
    backtest_parser.add_argument("--model", choices=MODEL_NAMES, default="random_forest")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare multiple models on the last years and report the best hit rate.",
    )
    add_common_history_args(benchmark_parser)
    benchmark_parser.add_argument("--test-years", type=int, default=DEFAULT_BACKTEST_YEARS)

    predict_parser = subparsers.add_parser("predict", help="Predict a single match.")
    add_common_history_args(predict_parser)
    predict_parser.add_argument("--player-a", required=True, help="First player name.")
    predict_parser.add_argument("--player-b", required=True, help="Second player name.")
    predict_parser.add_argument("--surface", required=True, help="Hard, Clay, Grass, Carpet, etc.")
    predict_parser.add_argument("--tourney-level", default="A", help="ATP level, e.g. G, M, A.")
    predict_parser.add_argument("--best-of", default="3", help="Usually 3 or 5.")
    predict_parser.add_argument("--date", help="Match date in YYYY-MM-DD. Defaults to latest history + 1 day.")
    predict_parser.add_argument("--tournament", default="", help="Optional tournament name for display only.")
    predict_parser.add_argument("--odds-a", type=float, help="Optional decimal odds for player A.")
    predict_parser.add_argument("--odds-b", type=float, help="Optional decimal odds for player B.")
    predict_parser.add_argument("--model", choices=MODEL_NAMES, default="hist_gradient_boosting")
    predict_parser.add_argument("--show-features", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        args.command = "interactive"
        args.start_year = None
        args.end_year = None
        args.history_years = DEFAULT_HISTORY_YEARS
    return args


def add_common_history_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--history-years", type=int, default=DEFAULT_HISTORY_YEARS)


def available_year_bounds() -> tuple[int, int]:
    years = sorted(
        int(path.stem.rsplit("_", 1)[-1])
        for pattern in [
            "atp_matches_[0-9][0-9][0-9][0-9].csv",
            "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv",
            "atp_matches_futures_[0-9][0-9][0-9][0-9].csv",
        ]
        for path in ATP_DATA_DIR.glob(pattern)
    )
    if not years:
        raise FileNotFoundError(f"No ATP match files found in {ATP_DATA_DIR}.")
    return years[0], years[-1]


def resolve_year_window(
    *,
    start_year: int | None,
    end_year: int | None,
    history_years: int | None = None,
    test_years: int | None = None,
    match_date: pd.Timestamp | None = None,
) -> tuple[int | None, int | None]:
    if start_year is not None or end_year is not None:
        return start_year, end_year

    _, max_year = available_year_bounds()
    if match_date is not None and history_years is not None:
        return match_date.year - history_years - 1, match_date.year
    if history_years is not None and test_years is not None:
        return max_year - history_years - test_years - 1, max_year
    return start_year, end_year


def list_match_files(start_year: int | None, end_year: int | None) -> list[Path]:
    min_year, max_year = available_year_bounds()
    start = min_year if start_year is None else start_year
    end = max_year if end_year is None else end_year
    files = []
    patterns = [
        "atp_matches_[0-9][0-9][0-9][0-9].csv",
        "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv",
        "atp_matches_futures_[0-9][0-9][0-9][0-9].csv",
    ]
    for pattern in patterns:
        for path in sorted(ATP_DATA_DIR.glob(pattern)):
            year = int(path.stem.rsplit("_", 1)[-1])
            if start <= year <= end:
                files.append(path)
    amateur_file = ATP_DATA_DIR / "atp_matches_amateur.csv"
    if amateur_file.exists() and start <= 1967 <= end:
        files.append(amateur_file)
    if not files:
        raise FileNotFoundError("No ATP match files found for the selected year range.")
    return sorted(files)


def normalize_surface(surface: str) -> str:
    cleaned = surface.strip().lower()
    mapping = {
        "hard": "Hard",
        "clay": "Clay",
        "grass": "Grass",
        "carpet": "Carpet",
    }
    return mapping.get(cleaned, surface.strip().title() or "Unknown")


def normalize_round(round_name: object) -> str:
    if pd.isna(round_name):
        return "Unknown"
    return str(round_name).strip().upper()


def estimated_match_date(
    tourney_date: pd.Timestamp, round_name: object, tourney_level: object
) -> pd.Timestamp:
    round_key = normalize_round(round_name)
    if str(tourney_level).strip().upper() == "G":
        offset = GRAND_SLAM_ROUND_DAY_OFFSETS.get(round_key, ROUND_DAY_OFFSETS.get(round_key, 0))
    else:
        offset = ROUND_DAY_OFFSETS.get(round_key, 0)
    return pd.Timestamp(tourney_date).normalize() + pd.Timedelta(days=int(offset))


def load_matches(start_year: int | None, end_year: int | None) -> pd.DataFrame:
    frames = []
    for path in list_match_files(start_year, end_year):
        frame = pd.read_csv(path, usecols=MATCH_COLUMNS)
        frame["source_file"] = path.name
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    matches["surface"] = matches["surface"].fillna("Unknown").astype(str).map(normalize_surface)
    matches["tourney_level"] = matches["tourney_level"].fillna("Unknown").astype(str)
    matches["best_of"] = matches["best_of"].fillna(-1).astype(int).astype(str)
    matches["round"] = matches["round"].map(normalize_round)
    matches["match_num"] = matches["match_num"].fillna(0)
    matches = matches.dropna(subset=["tourney_date", "winner_name", "loser_name"])
    matches["match_date"] = matches.apply(
        lambda row: estimated_match_date(row["tourney_date"], row["round"], row["tourney_level"]),
        axis=1,
    )
    matches = matches.sort_values(["match_date", "tourney_date", "source_file", "match_num"]).reset_index(
        drop=True
    )
    odds_df = load_men_odds(start_year, end_year)
    return attach_market_odds(matches, odds_df)


def normalize_text(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(value))
    return " ".join(cleaned.split())


def name_key_variants(name: str) -> set[tuple[str, str]]:
    tokens = normalize_text(name).split()
    if not tokens:
        return set()
    variants: set[tuple[str, str]] = set()
    if len(tokens) == 1:
        variants.add((tokens[0], ""))
        return variants

    if len(tokens[-1]) == 1:
        surname = " ".join(tokens[:-1])
        initials = tokens[-1]
        variants.add((surname, initials))
        variants.add((tokens[:-1][-1], initials))
        return variants

    variants.add((" ".join(tokens[1:]), tokens[0][0]))
    variants.add((tokens[-1], "".join(token[0] for token in tokens[:-1])))
    return variants


def choose_market_probs(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    sources = [("AvgW", "AvgL"), ("PSW", "PSL"), ("B365W", "B365L"), ("MaxW", "MaxL")]
    for winner_col, loser_col in sources:
        winner_odds = row.get(winner_col)
        loser_odds = row.get(loser_col)
        if pd.isna(winner_odds) or pd.isna(loser_odds):
            continue
        winner_odds = float(winner_odds)
        loser_odds = float(loser_odds)
        if winner_odds <= 1.0 or loser_odds <= 1.0:
            continue
        winner_raw = 1.0 / winner_odds
        loser_raw = 1.0 / loser_odds
        overround = winner_raw + loser_raw
        if overround <= 0:
            continue
        return winner_raw / overround, loser_raw / overround, overround
    return None, None, None


def week_start(date_value: pd.Timestamp) -> pd.Timestamp:
    normalized = pd.Timestamp(date_value).normalize()
    return normalized - pd.Timedelta(days=normalized.weekday())


def load_men_odds(start_year: int | None, end_year: int | None) -> pd.DataFrame:
    min_year, max_year = available_year_bounds()
    start = min_year if start_year is None else start_year
    end = max_year if end_year is None else end_year
    frames = []

    for path in sorted(ODDS_MEN_DIR.glob("*")):
        if path.suffix.lower() not in {".xls", ".xlsx"}:
            continue
        try:
            year = int(path.stem)
        except ValueError:
            continue
        if not (start <= year <= end + 1):
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported and will be removed",
                category=UserWarning,
            )
            frame = pd.read_excel(path)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Winner", "Loser"])
        frame["week_start"] = frame["Date"].map(week_start)
        frame["Surface"] = frame.get("Surface", pd.Series(index=frame.index)).fillna("Unknown").astype(str).map(
            normalize_surface
        )
        frame["Round"] = frame.get("Round", pd.Series(index=frame.index)).map(normalize_round)
        frame["Tournament"] = frame.get("Tournament", pd.Series(index=frame.index)).fillna("").astype(str)
        market_probs = frame.apply(choose_market_probs, axis=1, result_type="expand")
        market_probs.columns = ["winner_market_prob", "loser_market_prob", "market_overround"]
        frame["winner_variants"] = frame["Winner"].map(name_key_variants)
        frame["loser_variants"] = frame["Loser"].map(name_key_variants)
        frame = pd.concat([frame, market_probs], axis=1)
        frames.append(
            frame[
                [
                    "Date",
                    "week_start",
                    "Surface",
                    "Round",
                    "Tournament",
                    "winner_variants",
                    "loser_variants",
                    "winner_market_prob",
                    "loser_market_prob",
                    "market_overround",
                ]
            ].copy()
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "Date",
                "week_start",
                "Surface",
                "Round",
                "Tournament",
                "winner_variants",
                "loser_variants",
                "winner_market_prob",
                "loser_market_prob",
                "market_overround",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def text_similarity(value_a: str, value_b: str) -> float:
    if not value_a and not value_b:
        return 0.0
    return difflib.SequenceMatcher(a=normalize_text(value_a), b=normalize_text(value_b)).ratio()


def attach_market_odds(matches: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        matches["winner_market_prob"] = np.nan
        matches["loser_market_prob"] = np.nan
        matches["market_overround"] = np.nan
        return matches

    odds_by_week: dict[pd.Timestamp, list[dict[str, object]]] = defaultdict(list)
    for row in odds_df.itertuples(index=False):
        odds_by_week[pd.Timestamp(row.week_start).normalize()].append(
            {
                "winner_variants": row.winner_variants,
                "loser_variants": row.loser_variants,
                "winner_market_prob": row.winner_market_prob,
                "loser_market_prob": row.loser_market_prob,
                "market_overround": row.market_overround,
                "surface": normalize_surface(str(row.Surface)),
                "round": normalize_round(row.Round),
                "tournament": str(row.Tournament),
                "date": pd.Timestamp(row.Date).normalize(),
            }
        )

    winner_probs: list[float | None] = []
    loser_probs: list[float | None] = []
    market_overrounds: list[float | None] = []
    for row in matches.itertuples(index=False):
        week_candidates = {
            week_start(row.match_date),
            week_start(row.match_date - pd.Timedelta(days=7)),
            week_start(row.match_date + pd.Timedelta(days=7)),
        }
        winner_keys = name_key_variants(row.winner_name)
        loser_keys = name_key_variants(row.loser_name)
        winner_prob = None
        loser_prob = None
        market_overround = None
        best_score = float("-inf")
        for week_key in week_candidates:
            for candidate in odds_by_week.get(week_key, []):
                if not (winner_keys & candidate["winner_variants"] and loser_keys & candidate["loser_variants"]):
                    continue

                score = 0.0
                if candidate["surface"] == row.surface:
                    score += 2.0
                if candidate["round"] == row.round:
                    score += 1.5
                score += 3.0 * text_similarity(row.tourney_name, candidate["tournament"])
                score -= abs((candidate["date"] - row.match_date).days) * 0.05

                if score > best_score:
                    best_score = score
                    winner_prob = candidate["winner_market_prob"]
                    loser_prob = candidate["loser_market_prob"]
                    market_overround = candidate["market_overround"]
        winner_probs.append(winner_prob)
        loser_probs.append(loser_prob)
        market_overrounds.append(market_overround)

    enriched = matches.copy()
    enriched["winner_market_prob"] = winner_probs
    enriched["loser_market_prob"] = loser_probs
    enriched["market_overround"] = market_overrounds
    return enriched


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_k_factor(match_count: int) -> float:
    return 250.0 / ((match_count + 5) ** 0.4)


def update_elo(
    rating_a: float,
    rating_b: float,
    outcome_a: float,
    match_count_a: int,
    match_count_b: int,
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    k = (elo_k_factor(match_count_a) + elo_k_factor(match_count_b)) / 2.0
    delta = k * (outcome_a - expected_a)
    return rating_a + delta, rating_b - delta


def get_h2h_diff(
    h2h_records: DefaultDict[tuple[str, str], dict[str, int]], player_a: str, player_b: str
) -> float:
    key = tuple(sorted((player_a, player_b)))
    record = h2h_records[key]
    wins_a = record[player_a]
    wins_b = record[player_b]
    total = wins_a + wins_b
    shrunk_win_rate = (wins_a + H2H_PRIOR) / (total + 2 * H2H_PRIOR)
    return 2.0 * (shrunk_win_rate - 0.5)


def update_h2h(
    h2h_records: DefaultDict[tuple[str, str], dict[str, int]], winner: str, loser: str
) -> None:
    key = tuple(sorted((winner, loser)))
    h2h_records[key][winner] += 1


def build_samples(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, PlayerState], DefaultDict[tuple[str, str], dict[str, int]]]:
    player_states: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h_records: DefaultDict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    samples: list[dict[str, object]] = []

    for row in matches.itertuples(index=False):
        match_date = row.match_date
        surface = row.surface
        winner = row.winner_name
        loser = row.loser_name

        winner_state = player_states[winner]
        loser_state = player_states[loser]

        player_a_is_winner = normalize_name(winner) <= normalize_name(loser)
        player_a = winner if player_a_is_winner else loser
        player_b = loser if player_a_is_winner else winner
        state_a = winner_state if player_a_is_winner else loser_state
        state_b = loser_state if player_a_is_winner else winner_state
        rank_a = row.winner_rank if player_a_is_winner else row.loser_rank
        rank_b = row.loser_rank if player_a_is_winner else row.winner_rank
        rank_points_a = row.winner_rank_points if player_a_is_winner else row.loser_rank_points
        rank_points_b = row.loser_rank_points if player_a_is_winner else row.winner_rank_points
        age_a = row.winner_age if player_a_is_winner else row.loser_age
        age_b = row.loser_age if player_a_is_winner else row.winner_age
        height_a = row.winner_ht if player_a_is_winner else row.loser_ht
        height_b = row.loser_ht if player_a_is_winner else row.winner_ht
        market_prob_a = row.winner_market_prob if player_a_is_winner else row.loser_market_prob
        market_prob_b = row.loser_market_prob if player_a_is_winner else row.winner_market_prob

        sample = {
            "tourney_date": match_date,
            "surface": surface,
            "tourney_level": row.tourney_level,
            "best_of": row.best_of,
            "elo_diff": state_a.elo - state_b.elo,
            "surface_elo_diff": state_a.surface_elo(surface) - state_b.surface_elo(surface),
            "rank_diff": safe_diff(rank_b, rank_a),
            "rank_points_diff": safe_diff(rank_points_a, rank_points_b),
            "age_diff": safe_diff(age_a, age_b),
            "height_diff": safe_diff(height_a, height_b),
            "form_ewm_diff": state_a.form_value() - state_b.form_value(),
            "h2h_diff": get_h2h_diff(h2h_records, player_a, player_b),
            "days_since_last_match_diff": state_a.rest_days(match_date) - state_b.rest_days(match_date),
            "ace_rate_ewm_diff": state_a.ace_rate_ewm - state_b.ace_rate_ewm,
            "df_rate_ewm_diff": state_a.df_rate_ewm - state_b.df_rate_ewm,
            "first_in_rate_ewm_diff": state_a.first_in_rate_ewm - state_b.first_in_rate_ewm,
            "first_serve_win_rate_ewm_diff": state_a.first_serve_win_rate_ewm
            - state_b.first_serve_win_rate_ewm,
            "second_serve_win_rate_ewm_diff": state_a.second_serve_win_rate_ewm
            - state_b.second_serve_win_rate_ewm,
            "break_save_rate_ewm_diff": state_a.break_save_rate_ewm - state_b.break_save_rate_ewm,
            "recent_retirement_ewm_diff": state_a.recent_retirement_ewm - state_b.recent_retirement_ewm,
            "market_prob_diff": safe_diff(market_prob_a, market_prob_b),
            "market_overround": row.market_overround if pd.notna(row.market_overround) else None,
            "player_a_wins": 1 if player_a_is_winner else 0,
        }

        samples.append(sample)

        winner_state.elo, loser_state.elo = update_elo(
            winner_state.elo,
            loser_state.elo,
            1.0,
            winner_state.match_count,
            loser_state.match_count,
        )
        winner_surface_elo, loser_surface_elo = update_elo(
            winner_state.surface_elo(surface),
            loser_state.surface_elo(surface),
            1.0,
            winner_state.match_count,
            loser_state.match_count,
        )
        winner_state.set_surface_elo(surface, winner_surface_elo)
        loser_state.set_surface_elo(surface, loser_surface_elo)

        winner_state.update_form(1)
        loser_state.update_form(0)
        match_retired = retired_in_match(row.score)
        winner_state.update_serve_stats(
            ace_rate=safe_ratio(row.w_ace, row.w_svpt),
            df_rate=safe_ratio(row.w_df, row.w_svpt),
            first_in_rate=safe_ratio(row.w_1stIn, row.w_svpt),
            first_serve_win_rate=safe_ratio(row.w_1stWon, row.w_1stIn),
            second_serve_win_rate=safe_ratio(row.w_2ndWon, row.w_svpt - row.w_1stIn if pd.notna(row.w_svpt) and pd.notna(row.w_1stIn) else np.nan),
            break_save_rate=safe_ratio(row.w_bpSaved, row.w_bpFaced),
            retired=0.0,
        )
        loser_state.update_serve_stats(
            ace_rate=safe_ratio(row.l_ace, row.l_svpt),
            df_rate=safe_ratio(row.l_df, row.l_svpt),
            first_in_rate=safe_ratio(row.l_1stIn, row.l_svpt),
            first_serve_win_rate=safe_ratio(row.l_1stWon, row.l_1stIn),
            second_serve_win_rate=safe_ratio(row.l_2ndWon, row.l_svpt - row.l_1stIn if pd.notna(row.l_svpt) and pd.notna(row.l_1stIn) else np.nan),
            break_save_rate=safe_ratio(row.l_bpSaved, row.l_bpFaced),
            retired=match_retired,
        )
        winner_state.update_profile(
            display_name=winner,
            rank=row.winner_rank,
            rank_points=row.winner_rank_points,
            age=row.winner_age,
            height=row.winner_ht,
            match_date=match_date,
        )
        loser_state.update_profile(
            display_name=loser,
            rank=row.loser_rank,
            rank_points=row.loser_rank_points,
            age=row.loser_age,
            height=row.loser_ht,
            match_date=match_date,
        )

        update_h2h(h2h_records, winner, loser)

    return pd.DataFrame(samples), player_states, h2h_records


def safe_diff(value_a: object, value_b: object) -> float | None:
    if pd.isna(value_a) or pd.isna(value_b):
        return None
    return float(value_a) - float(value_b)


def safe_ratio(numerator: object, denominator: object) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0:
        return None
    return float(numerator) / denominator_value


def retired_in_match(score: object) -> float:
    text = str(score).upper()
    return 1.0 if "RET" in text or "W/O" in text or "DEF" in text else 0.0


def de_vig_from_odds(odds_a: float | None, odds_b: float | None) -> tuple[float | None, float | None, float | None]:
    if odds_a is None or odds_b is None:
        return None, None, None
    if odds_a <= 1.0 or odds_b <= 1.0:
        return None, None, None
    raw_a = 1.0 / float(odds_a)
    raw_b = 1.0 / float(odds_b)
    overround = raw_a + raw_b
    if overround <= 0:
        return None, None, None
    return raw_a / overround, raw_b / overround, overround


def filter_history_window(
    matches: pd.DataFrame, match_date: pd.Timestamp, history_years: int
) -> pd.DataFrame:
    window_start = match_date - pd.DateOffset(years=history_years)
    return matches[(matches["match_date"] < match_date) & (matches["match_date"] >= window_start)].copy()


def test_year_sequence(matches: pd.DataFrame, test_years: int) -> list[int]:
    date_column = "match_date" if "match_date" in matches.columns else "tourney_date"
    latest_year = int(matches[date_column].dt.year.max())
    first_year = latest_year - test_years + 1
    return [year for year in range(first_year, latest_year + 1)]


def to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_sigmoid_calibrator(probabilities: np.ndarray, labels: pd.Series) -> LogisticRegression | None:
    if len(np.unique(labels.to_numpy())) < 2:
        return None
    calibrator = LogisticRegression(max_iter=1000, C=1e6)
    calibrator.fit(to_logit(probabilities), labels.to_numpy())
    return calibrator


def predict_with_calibration(
    model: Pipeline, X: pd.DataFrame, calibrator: LogisticRegression | None
) -> np.ndarray:
    probabilities = model.predict_proba(X)[:, 1]
    if calibrator is None:
        return probabilities
    return calibrator.predict_proba(to_logit(probabilities))[:, 1]


def fit_prediction_model(
    samples: pd.DataFrame, model_name: str, match_date: pd.Timestamp
) -> tuple[Pipeline, LogisticRegression | None]:
    validation_start = match_date - pd.DateOffset(years=1)
    train_df = samples[samples["tourney_date"] < validation_start].copy()
    validation_df = samples[samples["tourney_date"] >= validation_start].copy()

    if train_df.empty:
        train_df = samples.copy()
        validation_df = pd.DataFrame(columns=samples.columns)

    model = make_pipeline(model_name)
    model.fit(
        train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        train_df["player_a_wins"],
    )

    calibrator = None
    if not validation_df.empty:
        validation_probs = model.predict_proba(
            validation_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )[:, 1]
        calibrator = fit_sigmoid_calibrator(validation_probs, validation_df["player_a_wins"])

    return model, calibrator


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic":
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", make_one_hot_encoder()),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        estimator = LogisticRegression(max_iter=1000, C=1.0)
    elif model_name == "random_forest":
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", make_one_hot_encoder()),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        estimator = RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=100,
            max_features=0.4,
            max_samples=0.7,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ]
                    ),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", make_one_hot_encoder()),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        estimator = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.03,
            max_iter=700,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def print_top_importances(model: Pipeline) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "coef_"):
        scores = pd.Series(estimator.coef_[0], index=feature_names).abs()
        label = "Top coefficient magnitudes"
    else:
        scores = pd.Series(estimator.feature_importances_, index=feature_names)
        label = "Top feature importances"

    top = scores.sort_values(ascending=False).head(10)
    print(f"\n{label}:")
    for name, value in top.items():
        print(f"  {name}: {value:.3f}")


def build_player_index(player_states: dict[str, PlayerState]) -> dict[str, str]:
    index: dict[str, str] = {}
    for key, state in player_states.items():
        display_name = state.display_name or key
        index[normalize_name(display_name)] = key
    return index


def normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def resolve_player(name: str, player_index: dict[str, str]) -> str:
    normalized = normalize_name(name)
    if normalized in player_index:
        return player_index[normalized]

    matches = difflib.get_close_matches(normalized, player_index.keys(), n=5, cutoff=0.6)
    if matches:
        suggestions = ", ".join(player_index[m] for m in matches)
        raise ValueError(f"Player '{name}' not found. Similar names: {suggestions}")
    raise ValueError(f"Player '{name}' not found in the selected history range.")


def build_prediction_row(
    *,
    player_a: str,
    player_b: str,
    surface: str,
    tourney_level: str,
    best_of: str,
    match_date: pd.Timestamp,
    player_states: dict[str, PlayerState],
    h2h_records: DefaultDict[tuple[str, str], dict[str, int]],
    market_prob_a: float | None = None,
    market_prob_b: float | None = None,
    market_overround: float | None = None,
) -> pd.DataFrame:
    state_a = player_states[player_a]
    state_b = player_states[player_b]
    row = {
        "surface": surface,
        "tourney_level": tourney_level,
        "best_of": str(best_of),
        "elo_diff": state_a.elo - state_b.elo,
        "surface_elo_diff": state_a.surface_elo(surface) - state_b.surface_elo(surface),
        "rank_diff": safe_diff(state_b.last_rank, state_a.last_rank),
        "rank_points_diff": safe_diff(state_a.last_rank_points, state_b.last_rank_points),
        "age_diff": safe_diff(state_a.age_on(match_date), state_b.age_on(match_date)),
        "height_diff": safe_diff(state_a.last_height, state_b.last_height),
        "form_ewm_diff": state_a.form_value() - state_b.form_value(),
        "h2h_diff": get_h2h_diff(h2h_records, player_a, player_b),
        "days_since_last_match_diff": state_a.rest_days(match_date) - state_b.rest_days(match_date),
        "ace_rate_ewm_diff": state_a.ace_rate_ewm - state_b.ace_rate_ewm,
        "df_rate_ewm_diff": state_a.df_rate_ewm - state_b.df_rate_ewm,
        "first_in_rate_ewm_diff": state_a.first_in_rate_ewm - state_b.first_in_rate_ewm,
        "first_serve_win_rate_ewm_diff": state_a.first_serve_win_rate_ewm - state_b.first_serve_win_rate_ewm,
        "second_serve_win_rate_ewm_diff": state_a.second_serve_win_rate_ewm - state_b.second_serve_win_rate_ewm,
        "break_save_rate_ewm_diff": state_a.break_save_rate_ewm - state_b.break_save_rate_ewm,
        "recent_retirement_ewm_diff": state_a.recent_retirement_ewm - state_b.recent_retirement_ewm,
        "market_prob_diff": safe_diff(market_prob_a, market_prob_b),
        "market_overround": market_overround,
    }
    return pd.DataFrame([row])


def walk_forward_evaluation(
    samples: pd.DataFrame,
    model_name: str,
    history_years: int,
    test_years: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    yearly_results: list[dict[str, float | int | str]] = []
    all_probabilities: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for year in test_year_sequence(samples, test_years):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        next_year_start = pd.Timestamp(year=year + 1, month=1, day=1)
        train_start = year_start - pd.DateOffset(years=history_years)
        validation_start = max(train_start, year_start - pd.DateOffset(years=1))

        train_df = samples[
            (samples["tourney_date"] >= train_start) & (samples["tourney_date"] < validation_start)
        ].copy()
        validation_df = samples[
            (samples["tourney_date"] >= validation_start) & (samples["tourney_date"] < year_start)
        ].copy()
        test_df = samples[
            (samples["tourney_date"] >= year_start) & (samples["tourney_date"] < next_year_start)
        ].copy()

        if train_df.empty or validation_df.empty or test_df.empty:
            continue

        model = make_pipeline(model_name)
        X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_train = train_df["player_a_wins"]
        X_validation = validation_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_validation = validation_df["player_a_wins"]
        X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_test = test_df["player_a_wins"]

        model.fit(X_train, y_train)
        validation_probs = model.predict_proba(X_validation)[:, 1]
        calibrator = fit_sigmoid_calibrator(validation_probs, y_validation)
        test_probs = predict_with_calibration(model, X_test, calibrator)
        test_preds = (test_probs >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, test_preds)

        yearly_results.append(
            {
                "year": year,
                "train": int(len(train_df)),
                "validation": int(len(validation_df)),
                "test": int(len(test_df)),
                "correct": int((test_preds == y_test).sum()),
                "accuracy": float(accuracy),
            }
        )
        all_probabilities.append(test_probs)
        all_predictions.append(test_preds)
        all_labels.append(y_test.to_numpy())

    if not yearly_results:
        raise ValueError("Walk-forward evaluation failed because no valid yearly splits were found.")

    probabilities = np.concatenate(all_probabilities)
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    summary = {
        "model": model_name,
        "years": len(yearly_results),
        "correct": int((predictions == labels).sum()),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }
    return summary, yearly_results


def evaluate_model(args: argparse.Namespace) -> None:
    start_year, end_year = resolve_year_window(
        start_year=args.start_year,
        end_year=args.end_year,
        history_years=args.history_years,
        test_years=args.test_years,
    )
    matches = load_matches(start_year, end_year)
    print(
        f"Loaded {len(matches):,} ATP matches from {matches['match_date'].min().date()} "
        f"to {matches['match_date'].max().date()}."
    )

    samples, _, _ = build_samples(matches)
    summary, yearly_results = walk_forward_evaluation(
        samples=samples,
        model_name=args.model,
        history_years=args.history_years,
        test_years=args.test_years,
    )

    print(f"Model: {args.model}")
    print(f"Backtest period: last {args.test_years} calendar years")
    print(f"Walk-forward years used: {summary['years']}")
    print(f"Correct predictions: {summary['correct']:,} / {summary['total']:,}")
    print(f"Trefferquote: {summary['accuracy']:.2%}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"ROC-AUC:  {summary['roc_auc']:.4f}")
    print(f"Log loss: {summary['log_loss']:.4f}")
    print(f"Brier:    {summary['brier']:.4f}")
    print("Decision threshold: 0.500 (nach Kalibrierung)")
    print("\nPer-year backtest:")
    for row in yearly_results:
        print(
            f"  {row['year']}: Trefferquote {row['accuracy']:.2%} | "
            f"richtig {row['correct']:,}/{row['test']:,}"
        )


def predict_match(args: argparse.Namespace) -> None:
    if args.start_year is not None or args.end_year is not None:
        initial_matches = load_matches(args.start_year, args.end_year)
        latest_history_date = initial_matches["match_date"].max()
    else:
        _, max_year = available_year_bounds()
        initial_matches = None
        latest_history_date = load_matches(max_year, max_year)["match_date"].max()
    if args.date:
        match_date = pd.Timestamp(args.date)
    else:
        match_date = latest_history_date + pd.Timedelta(days=1)

    start_year, end_year = resolve_year_window(
        start_year=args.start_year,
        end_year=args.end_year,
        history_years=args.history_years,
        match_date=match_date,
    )
    matches = initial_matches if initial_matches is not None and (start_year == args.start_year and end_year == args.end_year) else load_matches(start_year, end_year)

    history_matches = filter_history_window(matches, match_date, args.history_years)
    if history_matches.empty:
        raise ValueError("No historical matches available before the requested match date.")

    samples, player_states, h2h_records = build_samples(history_matches)
    player_index = build_player_index(player_states)
    player_a_key = resolve_player(args.player_a, player_index)
    player_b_key = resolve_player(args.player_b, player_index)

    model, calibrator = fit_prediction_model(samples, args.model, match_date)
    market_prob_a, market_prob_b, market_overround = de_vig_from_odds(args.odds_a, args.odds_b)

    prediction_row = build_prediction_row(
        player_a=player_a_key,
        player_b=player_b_key,
        surface=normalize_surface(args.surface),
        tourney_level=args.tourney_level,
        best_of=str(args.best_of),
        match_date=match_date,
        player_states=player_states,
        h2h_records=h2h_records,
        market_prob_a=market_prob_a,
        market_prob_b=market_prob_b,
        market_overround=market_overround,
    )
    probability_a = predict_with_calibration(model, prediction_row, calibrator)[0]
    probability_b = 1.0 - probability_a

    player_a_name = player_states[player_a_key].display_name or player_a_key
    player_b_name = player_states[player_b_key].display_name or player_b_key
    predicted_winner = player_a_name if probability_a >= 0.5 else player_b_name

    print(f"Model: {args.model}")
    print(
        f"History used: {len(history_matches):,} matches up to "
        f"{history_matches['match_date'].max().date()} "
        f"from the last {args.history_years} years"
    )
    if args.tournament:
        print(f"Tournament: {args.tournament}")
    print(
        f"Context: {player_a_name} vs {player_b_name} | "
        f"{normalize_surface(args.surface)} | level {args.tourney_level} | best-of {args.best_of}"
    )
    print(f"Match date: {match_date.date()}")
    print(f"Win probability {player_a_name}: {probability_a:.2%}")
    print(f"Win probability {player_b_name}: {probability_b:.2%}")
    print("Decision threshold: 0.500")
    print(f"Predicted winner: {predicted_winner}")

    if args.show_features:
        print("\nFeature snapshot:")
        for feature in NUMERIC_FEATURES:
            value = prediction_row.iloc[0][feature]
            print(f"  {feature}: {value}")


def benchmark_models(args: argparse.Namespace) -> None:
    start_year, end_year = resolve_year_window(
        start_year=args.start_year,
        end_year=args.end_year,
        history_years=args.history_years,
        test_years=args.test_years,
    )
    matches = load_matches(start_year, end_year)
    print(
        f"Loaded {len(matches):,} ATP matches from {matches['match_date'].min().date()} "
        f"to {matches['match_date'].max().date()}."
    )
    samples, _, _ = build_samples(matches)

    results: list[dict[str, float | int | str]] = []
    for model_name in MODEL_NAMES:
        print(f"Training {model_name} ...")
        summary, _ = walk_forward_evaluation(
            samples=samples,
            model_name=model_name,
            history_years=args.history_years,
            test_years=args.test_years,
        )
        results.append(summary)

    ranked = sorted(results, key=lambda row: (row["log_loss"], row["brier"], -row["accuracy"]))

    print("")
    print("Model comparison")
    print("----------------")
    for row in ranked:
        print(
            f"{row['model']}: Trefferquote {row['accuracy']:.2%} | "
            f"richtig {row['correct']:,}/{row['total']:,} | "
            f"ROC-AUC {row['roc_auc']:.4f} | Log loss {row['log_loss']:.4f} | "
            f"Brier {row['brier']:.4f}"
        )

    best = ranked[0]
    print("")
    print(
        f"Beste Methode: {best['model']} mit Log loss {best['log_loss']:.4f}, "
        f"Brier {best['brier']:.4f} und Trefferquote {best['accuracy']:.2%} "
        f"({best['correct']:,}/{best['total']:,} richtig)."
    )


def prompt_text(label: str, default: str | None = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("Bitte etwas eingeben.")


def prompt_choice(label: str, options: list[tuple[str, str]], default_key: str) -> str:
    print(label)
    for key, text in options:
        marker = " (Standard)" if key == default_key else ""
        print(f"  {key}) {text}{marker}")
    valid = {key: text for key, text in options}
    while True:
        choice = input("Auswahl: ").strip().lower()
        if not choice:
            return default_key
        if choice in valid:
            return choice
        print("Ungültige Auswahl.")


def prompt_player(label: str, player_index: dict[str, str]) -> str:
    while True:
        raw_name = prompt_text(label)
        normalized = normalize_name(raw_name)
        if normalized in player_index:
            return player_index[normalized]

        suggestions = difflib.get_close_matches(normalized, player_index.keys(), n=5, cutoff=0.5)
        if not suggestions:
            print("Spieler nicht gefunden. Versuch es erneut.")
            continue

        print("Meintest du einen von diesen Spielern?")
        suggestion_options = []
        for idx, suggestion in enumerate(suggestions, start=1):
            canonical = player_index[suggestion]
            suggestion_options.append((str(idx), canonical))
            print(f"  {idx}) {canonical}")
        print("  0) Neu eingeben")

        selection = input("Auswahl: ").strip()
        if selection == "0":
            continue
        selected = dict(suggestion_options).get(selection)
        if selected:
            return selected
        print("Ungültige Auswahl.")


def interactive_mode(args: argparse.Namespace) -> None:
    matches = load_matches(args.start_year, args.end_year)
    latest_history_date = matches["tourney_date"].max()

    print("Tennis Match Predictor")
    print("----------------------")
    print("Ich frage dich jetzt Schritt für Schritt nach dem Match.")
    print(f"Standardmäßig nutze ich die letzten {args.history_years} Jahre Historie.")
    print("")

    default_date = str((latest_history_date + pd.Timedelta(days=1)).date())
    date_text = prompt_text("Match-Datum (YYYY-MM-DD)", default=default_date)
    match_date = pd.Timestamp(date_text)

    history_matches = filter_history_window(matches, match_date, args.history_years)
    if history_matches.empty:
        raise ValueError("Für dieses Datum gibt es im 20-Jahres-Fenster keine Historie.")

    print("Historie wird geladen und Spielerprofile werden aufgebaut ...")
    samples, player_states, h2h_records = build_samples(history_matches)
    player_index = build_player_index(player_states)

    player_a_key = prompt_player("Spieler A", player_index)
    while True:
        player_b_key = prompt_player("Spieler B", player_index)
        if player_b_key != player_a_key:
            break
        print("Bitte zwei unterschiedliche Spieler wählen.")

    surface_map = {"1": "Hard", "2": "Clay", "3": "Grass", "4": "Carpet"}
    surface_choice = prompt_choice(
        "Belag auswählen",
        [("1", "Hard"), ("2", "Clay"), ("3", "Grass"), ("4", "Carpet")],
        default_key="1",
    )
    surface = surface_map[surface_choice]

    level_map = {"1": "G", "2": "M", "3": "A", "4": "D"}
    level_choice = prompt_choice(
        "Turnier-Level auswählen",
        [
            ("1", "G = Grand Slam"),
            ("2", "M = Masters"),
            ("3", "A = ATP Tour"),
            ("4", "D = Davis Cup / Sonstiges"),
        ],
        default_key="3",
    )
    tourney_level = level_map[level_choice]

    best_of_map = {"1": "3", "2": "5"}
    best_of_choice = prompt_choice(
        "Format auswählen",
        [("1", "Best of 3"), ("2", "Best of 5")],
        default_key="1",
    )
    best_of = best_of_map[best_of_choice]

    tournament = prompt_text("Turniername", default="", required=False)
    odds_a_text = prompt_text("Dezimalquote Spieler A (optional)", default="", required=False)
    odds_b_text = prompt_text("Dezimalquote Spieler B (optional)", default="", required=False)
    model_choice = prompt_choice(
        "Modell auswählen",
        [("1", "Hist Gradient Boosting"), ("2", "Random Forest"), ("3", "Logistic Regression")],
        default_key="1",
    )
    model_name = {
        "1": "hist_gradient_boosting",
        "2": "random_forest",
        "3": "logistic",
    }[model_choice]

    model, calibrator = fit_prediction_model(samples, model_name, match_date)
    odds_a = float(odds_a_text) if odds_a_text else None
    odds_b = float(odds_b_text) if odds_b_text else None
    market_prob_a, market_prob_b, market_overround = de_vig_from_odds(odds_a, odds_b)

    prediction_row = build_prediction_row(
        player_a=player_a_key,
        player_b=player_b_key,
        surface=surface,
        tourney_level=tourney_level,
        best_of=best_of,
        match_date=match_date,
        player_states=player_states,
        h2h_records=h2h_records,
        market_prob_a=market_prob_a,
        market_prob_b=market_prob_b,
        market_overround=market_overround,
    )

    probability_a = predict_with_calibration(model, prediction_row, calibrator)[0]
    probability_b = 1.0 - probability_a
    player_a_name = player_states[player_a_key].display_name or player_a_key
    player_b_name = player_states[player_b_key].display_name or player_b_key
    predicted_winner = player_a_name if probability_a >= 0.5 else player_b_name

    print("")
    print("Vorhersage")
    print("----------")
    if tournament:
        print(f"Turnier: {tournament}")
    print(
        f"Match: {player_a_name} vs {player_b_name} | {surface} | Level {tourney_level} | Best of {best_of}"
    )
    print(f"Match-Datum: {match_date.date()}")
    print(
        f"Verwendete Historie: {len(history_matches):,} Matches bis {history_matches['match_date'].max().date()} "
        f"aus den letzten {args.history_years} Jahren"
    )
    print(f"Siegchance {player_a_name}: {probability_a:.2%}")
    print(f"Siegchance {player_b_name}: {probability_b:.2%}")
    print("Decision threshold: 0.500")
    print(f"Prognose: {predicted_winner}")

    show_features = prompt_choice(
        "Willst du die benutzten Feature-Werte sehen?",
        [("1", "Ja"), ("2", "Nein")],
        default_key="2",
    )
    if show_features == "1":
        print("")
        print("Feature snapshot")
        print("----------------")
        for feature in NUMERIC_FEATURES:
            print(f"{feature}: {prediction_row.iloc[0][feature]}")


def main() -> None:
    args = parse_args()
    if args.command == "interactive":
        interactive_mode(args)
    elif args.command == "benchmark":
        benchmark_models(args)
    elif args.command in {"backtest", "evaluate"}:
        evaluate_model(args)
    elif args.command == "predict":
        predict_match(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
