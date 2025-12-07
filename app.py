import streamlit as st
import pandas as pd
import math
import os
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import altair as alt

# -------------------------------
# Hugging Face Router 설정
# -------------------------------
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# -------------------------------
# 기본 매핑 / 상수
# -------------------------------
LEVEL_MAP = {0: "novice", 1: "competence", 2: "leader"}
LEVEL_KO = {0: "저연차", 1: "중간연차", 2: "고연차"}

SHIFT_MAP = {
    0: "D",    # day
    1: "E",    # eve
    2: "N",    # night
    3: "9D",   # 9시 day
    4: "OFF",
}

# 날짜별 기준 인력
SHIFT_REQUIREMENTS = {
    0: 6,  # D
    1: 6,  # E
    2: 5,  # N
    3: 1,  # 9D
}

# 근무 시작/끝 시간 (시간 단위, 밤 근무는 24+ 로 표현)
SHIFT_START_HOUR = {
    0: 6.5,   # D: 06:30
    1: 14.5,  # E: 14:30
    2: 22.5,  # N: 22:30
    3: 9.0,   # 9D: 09:00
}

SHIFT_END_HOUR = {
    0: 15.0,  # D: 15:00
    1: 23.0,  # E: 23:00
    2: 31.0,  # N: 07:00 (다음날 7시 = 24+7)
    3: 17.5,  # 9D: 17:30
}

RISK_SCORE = {
    "no": 0,
    "low": 1,
    "moderate": 2,
    "critical": 3,
}

RISK_LABEL_KO = {
    "critical": "고위험",
    "moderate": "중등도 위험",
    "low": "저위험",
    "no": "정상",
    "no_preference": "선호 없음",
    "no_request": "요청 없음",
}


def risk_to_ko(v: str) -> str:
    if pd.isna(v):
        return ""
    return RISK_LABEL_KO.get(str(v), str(v))


# -------------------------------
# 공통 유틸 함수
# -------------------------------
def get_date_columns(df: pd.DataFrame):
    """nurse_id, nurse_name, level 을 제외한 날짜 컬럼 목록"""
    return [c for c in df.columns if c not in ["nurse_id", "nurse_name", "level"]]


def longest_streak(values, is_valid):
    """values 리스트에서 is_valid(x)==True 인 값들의 최장 연속 길이"""
    max_run = 0
    current = 0
    for v in values:
        if pd.isna(v):
            current = 0
            continue
        if is_valid(int(v)):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# -------------------------------
# 위험도 판정 함수들
# -------------------------------
def work_streak_risk(n: int) -> str:
    if n >= 6:
        return "critical"
    elif n == 5:
        return "moderate"
    elif n == 4:
        return "low"
    else:
        return "no"


def night_streak_risk(n: int) -> str:
    if n >= 5:
        return "critical"
    elif n == 4:
        return "moderate"
    elif n == 3:
        return "low"
    else:
        return "no"


def total_off_days_risk(off_days: int) -> str:
    if off_days <= 8:
        return "critical"
    elif off_days == 9:
        return "moderate"
    elif 10 <= off_days <= 12:
        return "low"
    else:
        return "no"


def total_night_days_risk(n_nights: int) -> str:
    if n_nights >= 7:
        return "critical"
    elif n_nights == 6:
        return "low"
    else:
        return "no"


def ed_quick_return_risk(crit_cnt: int, mod_cnt: int) -> str:
    if crit_cnt > 0:
        return "critical"
    elif mod_cnt > 0:
        return "moderate"
    else:
        return "no"


def n_quick_return_risk(crit_cnt: int, mod_cnt: int, low_cnt: int) -> str:
    if crit_cnt > 0:
        return "critical"
    elif mod_cnt > 0:
        return "moderate"
    elif low_cnt > 0:
        return "low"
    else:
        return "no"


def staffing_risk_label(shortage: int) -> str:
    if shortage <= 0:
        return "no"
    elif shortage == 1:
        return "moderate"
    else:
        return "critical"


def overall_staffing_risk_day(risks_for_day):
    if "critical" in risks_for_day:
        return "critical"
    elif "moderate" in risks_for_day:
        return "moderate"
    elif "low" in risks_for_day:
        return "low"
    else:
        return "no"


def off_interval_risk(min_rest_hours):
    if min_rest_hours is None or (isinstance(min_rest_hours, float) and math.isnan(min_rest_hours)):
        return "no"
    if min_rest_hours < 11:
        return "critical"
    elif 11 <= min_rest_hours < 16:
        return "low"
    else:
        return "no"


def ratio_risk(r):
    if r is None or pd.isna(r):
        return "no"
    if r > 1.4:
        return "critical"
    elif r > 1.2:
        return "moderate"
    elif r >= 1.0:
        return "low"
    else:
        return "no"


# -------------------------------
# Quick return 스캔
# -------------------------------
def scan_quick_returns(values):
    """
    ED quick return:
      critical: ED(1,0), E9D(1,3)
      moderate: EOD(1,4,0), EO9D(1,4,3)

    N quick return (공주님 정의 반영):
      critical: ND(2,0), NOD(2,4,0), N9D(2,3), NO9D(2,4,3), NE(2,1)
      moderate: NOE(2,4,1)
    """
    ed_crit = ed_mod = 0
    n_crit = n_mod = n_low = 0

    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        v = int(v)

        # --- ED quick return ---
        if v == 1:  # E
            # 한 칸 뒤: D(0) 또는 9D(3) → critical
            if i + 1 < len(values) and not pd.isna(values[i + 1]):
                v1 = int(values[i + 1])
                if v1 in (0, 3):
                    ed_crit += 1
            # 두 칸 뒤: E-O-D / E-O-9D → moderate
            if i + 2 < len(values) and not pd.isna(values[i + 1]) and not pd.isna(values[i + 2]):
                v1 = int(values[i + 1])
                v2 = int(values[i + 2])
                if v1 == 4 and v2 in (0, 3):
                    ed_mod += 1

        # --- N quick return ---
        if v == 2:  # N
            # 한 칸 뒤: D(0), E(1), 9D(3) → 모두 critical
            if i + 1 < len(values) and not pd.isna(values[i + 1]):
                v1 = int(values[i + 1])
                if v1 in (0, 1, 3):
                    n_crit += 1

            # 두 칸 뒤: N-O-D / N-O-9D / N-O-E
            if i + 2 < len(values) and not pd.isna(values[i + 1]) and not pd.isna(values[i + 2]):
                v1 = int(values[i + 1])
                v2 = int(values[i + 2])
                if v1 == 4:  # 중간에 OFF
                    if v2 in (0, 3):    # NOD, NO9D
                        n_crit += 1
                    elif v2 == 1:       # NOE
                        n_mod += 1

    return ed_crit, ed_mod, n_crit, n_mod, n_low


# -------------------------------
# Staffing 분석
# -------------------------------
def compute_staffing_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    date_cols = get_date_columns(schedule_df)
    numeric = schedule_df[date_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for col in date_cols:
        col_values = numeric[col]
        row = {"date": col}
        day_risks = []

        for shift_code, required in SHIFT_REQUIREMENTS.items():
            count = (col_values == shift_code).sum()
            shortage = required - count
            shortage_for_display = max(shortage, 0)
            risk = staffing_risk_label(shortage)

            label = SHIFT_MAP[shift_code]
            row[f"{label}_count"] = int(count)
            row[f"{label}_required"] = required
            row[f"{label}_shortage"] = int(shortage_for_display)
            row[f"{label}_risk"] = risk
            day_risks.append(risk)

        row["overall_staffing_risk"] = overall_staffing_risk_day(day_risks)
        rows.append(row)

    df = pd.DataFrame(rows)
    # 날짜를 datetime 으로 한 번 변환해두기 (그래프용)
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass
    return df


# -------------------------------
# 최소 휴식시간 계산
# -------------------------------
def compute_min_off_interval(values):
    """연속 근무들 사이의 최소 휴식시간(시간 단위) 계산"""
    work_indices = [
        idx for idx, v in enumerate(values)
        if not pd.isna(v) and int(v) in (0, 1, 2, 3)
    ]
    if len(work_indices) <= 1:
        return None

    rests = []
    for k in range(len(work_indices) - 1):
        i = work_indices[k]
        j = work_indices[k + 1]
        s1 = int(values[i])
        s2 = int(values[j])

        end_time = SHIFT_END_HOUR.get(s1)
        start_time = SHIFT_START_HOUR.get(s2)
        if end_time is None or start_time is None:
            continue

        rest = (j - i) * 24 + start_time - end_time
        rests.append(rest)

    if not rests:
        return None
    return min(rests)


# -------------------------------
# 월간 feature 계산 (위험도 + 공정성 기본)
# -------------------------------
def compute_monthly_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    date_cols = get_date_columns(schedule_df)
    numeric = schedule_df[date_cols].apply(pd.to_numeric, errors="coerce")

    # 기본 집계
    total_off_days = (numeric == 4).sum(axis=1)
    total_night_days = (numeric == 2).sum(axis=1)
    total_working_days = numeric.isin([0, 1, 2, 3]).sum(axis=1)

    working_streaks = []
    night_streaks = []
    ed_crit_list = []
    ed_mod_list = []
    n_crit_list = []
    n_mod_list = []
    n_low_list = []
    min_rest_list = []

    for _, row in numeric.iterrows():
        values = row.values.tolist()

        max_work_streak = longest_streak(values, lambda x: x in (0, 1, 2, 3))
        max_night_streak = longest_streak(values, lambda x: x == 2)
        working_streaks.append(max_work_streak)
        night_streaks.append(max_night_streak)

        ed_c, ed_m, n_c, n_m, n_l = scan_quick_returns(values)
        ed_crit_list.append(ed_c)
        ed_mod_list.append(ed_m)
        n_crit_list.append(n_c)
        n_mod_list.append(n_m)
        n_low_list.append(n_l)

        min_rest = compute_min_off_interval(values)
        min_rest_list.append(min_rest)

    # 결과 테이블 구성
    result = schedule_df[["nurse_id", "nurse_name", "level"]].copy()
    result["level_name"] = result["level"].map(LEVEL_MAP)

    result["total_off_days"] = total_off_days
    result["total_night_days"] = total_night_days
    result["total_working_days"] = total_working_days

    result["total_off_days_risk"] = result["total_off_days"].apply(total_off_days_risk)
    result["total_night_days_risk"] = result["total_night_days"].apply(total_night_days_risk)

    result["consecutive_working_days"] = working_streaks
    result["consecutive_working_days_risk"] = [
        work_streak_risk(n) for n in working_streaks
    ]

    result["consecutive_night_shifts"] = night_streaks
    result["consecutive_night_shifts_risk"] = [
        night_streak_risk(n) for n in night_streaks
    ]

    result["min_off_interval_hours"] = min_rest_list
    result["min_off_interval_risk"] = result["min_off_interval_hours"].apply(
        off_interval_risk
    )

    # ED / N quick return
    result["ED_quick_return_critical"] = ed_crit_list
    result["ED_quick_return_moderate"] = ed_mod_list
    result["ED_quick_return_total"] = (
        result["ED_quick_return_critical"] + result["ED_quick_return_moderate"]
    )
    result["ED_quick_return_risk"] = [
        ed_quick_return_risk(c, m) for c, m in zip(ed_crit_list, ed_mod_list)
    ]

    result["N_quick_return_critical"] = n_crit_list
    result["N_quick_return_moderate"] = n_mod_list
    result["N_quick_return_low"] = n_low_list
    result["N_quick_return_total"] = (
        result["N_quick_return_critical"]
        + result["N_quick_return_moderate"]
        + result["N_quick_return_low"]
    )
    result["N_quick_return_risk"] = [
        n_quick_return_risk(c, m, l)
        for c, m, l in zip(n_crit_list, n_mod_list, n_low_list)
    ]

    # 연차 대비 근무/야간 비율
    result["level_night_ratio"] = pd.NA
    result["level_workingdays_ratio"] = pd.NA

    for lvl in result["level"].unique():
        mask_self = result["level"] == lvl
        mask_others = result["level"] != lvl
        if mask_others.sum() == 0:
            continue

        other_n_mean = result.loc[mask_others, "total_night_days"].mean()
        other_w_mean = result.loc[mask_others, "total_working_days"].mean()

        if other_n_mean and other_n_mean > 0:
            result.loc[mask_self, "level_night_ratio"] = (
                result.loc[mask_self, "total_night_days"] / other_n_mean
            )
        if other_w_mean and other_w_mean > 0:
            result.loc[mask_self, "level_workingdays_ratio"] = (
                result.loc[mask_self, "total_working_days"] / other_w_mean
            )

    result["level_night_ratio_risk"] = result["level_night_ratio"].apply(ratio_risk)
    result["level_workingdays_ratio_risk"] = result["level_workingdays_ratio"].apply(
        ratio_risk
    )

    return result


# -------------------------------
# Swing 패턴 탐지
# -------------------------------
def has_swing_pattern(values):
    """
    연속 근무(OFF 없이 0~3) 구간에서
    근무 코드가 바뀌는 지점이 1번이라도 있으면 swing 패턴 있다고 봄.
    """
    prev = None
    for v in values:
        if pd.isna(v):
            prev = None
            continue
        v_int = int(v)
        if v_int in (0, 1, 2, 3):
            if prev is not None and prev in (0, 1, 2, 3) and prev != v_int:
                return True
            prev = v_int
        else:
            prev = None
    return False


# -------------------------------
# 선호근무 기반 공정성 feature (참고 코드 버전 반영)
# -------------------------------
def compute_preference_features(
    base_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    pref_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    base_df : compute_monthly_features() 결과
    schedule_df : 실제 근무표 (코드 0~4)
    pref_df : 선호근무표 (같은 nurse_id + 날짜 컬럼, 마지막에 preferred_swing_types / preferred_shift_types)
    """
    base = base_df.copy()
    base = base.set_index("nurse_id", drop=False)

    # 날짜 컬럼 정렬
    date_cols_sched = get_date_columns(schedule_df)
    date_cols = [c for c in date_cols_sched if c in pref_df.columns]

    # 수치형으로 변환
    sched_numeric = (
        schedule_df.set_index("nurse_id")[date_cols]
        .apply(pd.to_numeric, errors="coerce")
    )
    pref_numeric = (
        pref_df.set_index("nurse_id")[date_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    # 선호 shift 타입 (예: 0=day, 1=eve 등, nurse 단위 1개 값)
    if "preferred_shift_types" in pref_df.columns:
        pref_shift_series = pd.to_numeric(
            pref_df.set_index("nurse_id")["preferred_shift_types"],
            errors="coerce",
        )
    else:
        pref_shift_series = pd.Series(dtype="float64")

    # 변형근무(swing) 선호 여부 (* 등 텍스트가 있으면 True)
    if "preferred_swing_types" in pref_df.columns:
        swing_raw = pref_df.set_index("nurse_id")["preferred_swing_types"]
        swing_flag_series = swing_raw.notna() & (
            swing_raw.astype(str).str.strip() != ""
        )
    else:
        swing_flag_series = pd.Series(
            False,
            index=pref_df.set_index("nurse_id").index,
        )

    # 결과 저장용 dict
    shift_type_code_dict = {}
    shift_ratio_dict = {}
    shift_req_dict = {}
    shift_match_dict = {}

    duty_req_dict = {}
    duty_match_dict = {}
    duty_ratio_dict = {}

    swing_flag_dict = {}
    swing_pattern_dict = {}

    # 간호사별로 선호 반영 계산
    for nid in base.index:
        # 기본 값 초기화
        shift_type_code = math.nan
        shift_ratio = math.nan
        shift_req = 0
        shift_match = 0

        duty_req = 0
        duty_match = 0
        duty_ratio = math.nan

        swing_flag = False
        swing_has_pattern = False

        # nurse_id가 두 DF에 모두 존재하는 경우에만 계산
        if (nid in sched_numeric.index) and (nid in pref_numeric.index):
            sched_row = sched_numeric.loc[nid]
            pref_row = pref_numeric.loc[nid]

            total_working = (
                base.loc[nid, "total_working_days"]
                if "total_working_days" in base.columns
                else sched_row.isin([0, 1, 2, 3]).sum()
            )

            # --- (1) preferred_shift_types : 전체 선호 근무 타입 반영률 ---
            if nid in pref_shift_series.index:
                p_code = pref_shift_series.loc[nid]
                shift_type_code = p_code
                if not pd.isna(p_code) and total_working and total_working > 0:
                    match_days = (sched_row == p_code).sum()
                    shift_match = int(match_days)
                    shift_req = int(total_working)
                    shift_ratio = match_days / float(total_working)

            # --- (2) preferred_duty_choice_types : 날짜별 희망 듀티 반영률 ---
            for col in date_cols:
                p_val = pref_row[col]
                s_val = sched_row[col]
                if not pd.isna(p_val):
                    duty_req += 1
                    if not pd.isna(s_val) and int(p_val) == int(s_val):
                        duty_match += 1
            if duty_req > 0:
                duty_ratio = duty_match / float(duty_req)

            # --- (3) preferred_swing_types : swing 선호 + 실제 swing 패턴 ---
            if nid in swing_flag_series.index:
                swing_flag = bool(swing_flag_series.loc[nid])
            # 스케줄에서 swing 패턴 존재 여부
            swing_has_pattern = has_swing_pattern(sched_row.values.tolist())

        # dict에 저장
        shift_type_code_dict[nid] = shift_type_code
        shift_ratio_dict[nid] = shift_ratio
        shift_req_dict[nid] = shift_req
        shift_match_dict[nid] = shift_match

        duty_req_dict[nid] = duty_req
        duty_match_dict[nid] = duty_match
        duty_ratio_dict[nid] = duty_ratio

        swing_flag_dict[nid] = swing_flag
        swing_pattern_dict[nid] = swing_has_pattern

    # --- (1) 선호 근무타입 반영률의 분위수 기반 risk ---
    shift_ratio_series = pd.Series(shift_ratio_dict)
    valid_shift = shift_ratio_series.dropna()
    if len(valid_shift) > 0:
        q10 = valid_shift.quantile(0.10)
        q25 = valid_shift.quantile(0.25)
    else:
        q10 = q25 = None

    shift_risk_dict = {}
    for nid, r in shift_ratio_dict.items():
        code = shift_type_code_dict.get(nid, math.nan)
        if (
            code is None
            or (isinstance(code, float) and math.isnan(code))
            or q10 is None
        ):
            shift_risk = "no_preference"
        else:
            if r < q10:
                shift_risk = "critical"
            elif r < q25:
                shift_risk = "moderate"
            else:
                shift_risk = "low"
        shift_risk_dict[nid] = shift_risk

    # --- (2) 희망 듀티 반영률 기반 risk ---
    duty_risk_dict = {}
    for nid, r in duty_ratio_dict.items():
        req = duty_req_dict.get(nid, 0)
        if req == 0 or pd.isna(r):
            duty_risk = "no_request"
        else:
            if r <= 0.75:
                duty_risk = "critical"
            elif r <= 0.875:
                duty_risk = "moderate"
            else:
                duty_risk = "low"
        duty_risk_dict[nid] = duty_risk

    # --- (3) swing 선호 반영 여부 risk ---
    swing_risk_dict = {}
    for nid in base.index:
        flag = swing_flag_dict.get(nid, False)
        has_pattern = swing_pattern_dict.get(nid, False)
        if not flag:
            swing_risk = "no"
        else:
            if has_pattern:
                swing_risk = "no"   # 선호 반영 잘 됨 → risk 없음
            else:
                swing_risk = "low"  # 선호 반영 안 됨 → low risk 정도
        swing_risk_dict[nid] = swing_risk

    # base DF에 컬럼 추가
    base["preferred_shift_type_code"] = pd.Series(shift_type_code_dict)
    base["preferred_shift_ratio"] = pd.Series(shift_ratio_dict)
    base["preferred_shift_ratio_risk"] = pd.Series(shift_risk_dict)
    base["preferred_shift_total"] = pd.Series(shift_req_dict)
    base["preferred_shift_matched"] = pd.Series(shift_match_dict)

    base["preferred_duty_requests"] = pd.Series(duty_req_dict)
    base["preferred_duty_matched"] = pd.Series(duty_match_dict)
    base["preferred_duty_choice_ratio"] = pd.Series(duty_ratio_dict)
    base["preferred_duty_choice_risk"] = pd.Series(duty_risk_dict)

    base["preferred_swing_flag"] = pd.Series(swing_flag_dict)
    base["preferred_swing_has_pattern"] = pd.Series(swing_pattern_dict)
    base["preferred_swing_risk"] = pd.Series(swing_risk_dict)

    return base.reset_index(drop=True)


# -------------------------------
# 클러스터링 + 이상치 탐지
# -------------------------------
def run_clustering_and_outlier(fairness_df: pd.DataFrame, n_clusters: int = 3, contamination: float = 0.1):
    df = fairness_df.copy()

    # 사용할 feature(숫자형) 선정
    feature_cols = [
        "total_off_days",
        "total_night_days",
        "total_working_days",
        "consecutive_working_days",
        "consecutive_night_shifts",
        "min_off_interval_hours",
        "ED_quick_return_total",
        "N_quick_return_total",
        "level_night_ratio",
        "level_workingdays_ratio",
        "preferred_shift_ratio",
        "preferred_duty_choice_ratio",
    ]

    # 존재하는 컬럼만 사용
    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) == 0 or len(df) < 3:
        df["cluster"] = 0
        df["cluster_name"] = "Cluster A"
        df["is_outlier"] = "normal"
        return df

    X = df[feature_cols].astype(float).copy()

    # 결측치는 각 컬럼의 중앙값으로 채우기
    for c in feature_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans 클러스터링
    k = min(n_clusters, len(df))  # 간호사 수보다 큰 K 방지
    if k < 2:
        k = 2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters
    df["cluster_name"] = df["cluster"].apply(lambda x: f"Cluster {chr(ord('A') + int(x))}")

    # IsolationForest 이상치 탐지
    try:
        iso = IsolationForest(contamination=contamination, random_state=42)
        out_pred = iso.fit_predict(X_scaled)  # -1: outlier, 1: normal
        df["anomaly_score_raw"] = out_pred
        df["is_outlier"] = df["anomaly_score_raw"].apply(lambda v: "outlier" if v == -1 else "normal")
    except Exception:
        df["anomaly_score_raw"] = 1
        df["is_outlier"] = "normal"

    return df


# -------------------------------
# LLM 호출 (Hugging Face Router)
# -------------------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Hugging Face Router를 통해 Llama-3.1-8B-Instruct를 호출하는 함수.
    system_prompt, user_prompt는 모두 한국어 문자열로 넘깁니다.
    """
    if not HF_API_TOKEN:
        return (
            "❌ HF_API_TOKEN이 설정되지 않았습니다.\n"
            "CMD 창에서 먼저 다음 명령을 실행해 주세요:\n"
            "  set HF_API_TOKEN=hf_로_시작하는_토큰값"
        )

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    }

    try:
        res = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[오류] LLM 호출 중 문제가 발생했습니다: {e}"


# -------------------------------
# Streamlit 앱 기본 설정 & 세션 상태
# -------------------------------
st.set_page_config(
    page_title="Nurse Schedule AI (로컬+HF)",
    layout="wide",
)

if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = None
    st.session_state.monthly_features_df = None
    st.session_state.staffing_df = None
    st.session_state.pref_df = None
    st.session_state.fairness_pref_df = None
    st.session_state.clustered_df = None
    st.session_state.chat_history = []

st.title("🩺 간호사 스케줄 인텔리전스 (Nurse Schedule Intelligence)")

st.markdown(
    """
이 앱은 **간호사 근무표(스케줄)**와 **선호근무표(pref)**를 업로드하면,

1. AI를 통한 챗봇 Q&A  
2. 날짜별 인력 기준 충족 여부와 환자 안전 위험도  
3. 개별 간호사 공정성  
4. 스케줄 패턴 자동 분류와 LLM 기반 자동 해석 리포트     

를 보여주는 도구입니다.
"""
)

st.write("---")

tab_schedule, tab_chat, tab_risk, tab_fairness, tab_report = st.tabs(
    ["📂 Schedule", "💬 Chatbot", "📊 Risk Dashboard", "⚖️ Fairness Dashboard", "🧠 AI Report"]
)

# ============================================================
# 1. Schedule 탭 (업로드 + 미리보기)
# ============================================================
with tab_schedule:
    st.subheader("📂 근무표 / 선호근무표 업로드")

    col1, col2 = st.columns(2)

    with col1:
        schedule_file = st.file_uploader(
            "1️⃣ 근무 스케줄 엑셀 파일 업로드 (필수)",
            type=["xlsx", "xls"],
            key="schedule_file_uploader",
        )

    with col2:
        pref_file = st.file_uploader(
            "2️⃣ 선호근무표 엑셀 파일 업로드 (선택)",
            type=["xlsx", "xls"],
            key="pref_file_uploader",
        )

    # 업로드 후 즉시 분석
    if schedule_file is not None:
        try:
            schedule_df = pd.read_excel(schedule_file)
            st.session_state.schedule_df = schedule_df
            st.session_state.monthly_features_df = compute_monthly_features(schedule_df)
            st.session_state.staffing_df = compute_staffing_features(schedule_df)
        except Exception as e:
            st.error(f"근무표 엑셀을 읽는 중 오류가 발생했습니다: {e}")
            st.stop()

        st.subheader("📄 원본 근무표 미리보기")
        st.caption("※ nurse_id / nurse_name / level 이후에 날짜별 근무코드(0=D,1=E,2=N,3=9D,4=OFF)가 와야 합니다.")
        st.dataframe(schedule_df.head(15), use_container_width=True)

    if (pref_file is not None) and (st.session_state.schedule_df is not None):
        try:
            pref_df = pd.read_excel(pref_file)
            st.session_state.pref_df = pref_df

            fairness_pref_df = compute_preference_features(
                st.session_state.monthly_features_df,
                st.session_state.schedule_df,
                pref_df,
            )
            st.session_state.fairness_pref_df = fairness_pref_df

            # 클러스터링
            st.sidebar.markdown("### ⚙️ 클러스터링 / 이상치 탐지 설정")
            k_for_cluster = st.sidebar.slider("클러스터 개수 (KMeans)", 2, 5, 3, 1)
            contamination_rate = st.sidebar.slider("이상치 비율 (IsolationForest)", 0.05, 0.3, 0.1, 0.05)

            clustered_df = run_clustering_and_outlier(
                fairness_pref_df,
                n_clusters=k_for_cluster,
                contamination=contamination_rate,
            )
            st.session_state.clustered_df = clustered_df
        except Exception as e:
            st.error(f"선호근무표 엑셀을 읽는 중 오류가 발생했습니다: {e}")
            st.stop()

        st.subheader("📄 선호근무표 미리보기")
        st.caption(
            "※ 근무표와 동일한 nurse_id / nurse_name / level / 날짜 컬럼 구조를 가지며, "
            "`preferred_swing_types`, `preferred_shift_types` 컬럼이 뒤에 추가된 형태여야 합니다."
        )
        st.dataframe(pref_df.head(15), use_container_width=True)

    if st.session_state.schedule_df is None:
        st.info("먼저 **근무 스케줄 엑셀 파일**을 업로드해 주세요.")


# 공통으로 사용할 데이터 단축 변수
schedule_df = st.session_state.schedule_df
monthly_features_df = st.session_state.monthly_features_df
staffing_df = st.session_state.staffing_df
pref_df = st.session_state.pref_df
clustered_df = st.session_state.clustered_df

# ============================================================
# 2. Chatbot 탭
# ============================================================
with tab_chat:
    st.subheader("💬 AI 챗봇 (간호사별 + 부서 전체 Q&A)")

    if clustered_df is None:
        st.info("챗봇을 사용하려면 **Schedule 탭에서 근무표와 선호근무표를 모두 업로드**해 주세요.")
    else:
        # 부서 전체 요약 (비교 질문용)
        summary_lines = []
        for _, r in clustered_df.iterrows():
            summary_lines.append(
                f"- {r['nurse_id']} {r['nurse_name']} "
                f"(연차={LEVEL_KO.get(r['level'], r['level'])}, "
                f"cluster={r['cluster_name']}, outlier={r['is_outlier']}, "
                f"총근무일={r['total_working_days']}, N근무={r['total_night_days']}, "
                f"연속근무={r['consecutive_working_days']}, 연속N={r['consecutive_night_shifts']}, "
                f"ED QR={r['ED_quick_return_total']}, N QR={r['N_quick_return_total']}, "
                f"희망근무 반영률={r['preferred_shift_ratio']}, "
                f"선호근태 반영률={r['preferred_duty_choice_ratio']}, "
                f"swing 위험도={r['preferred_swing_risk']})"
            )
        all_nurses_summary = "\n".join(summary_lines)

        nurse_options = clustered_df["nurse_id"].astype(str) + " - " + clustered_df["nurse_name"].astype(str)
        chat_nurse_label = st.selectbox(
            "기준이 될 간호사를 하나 선택하세요 (비교 질문도 가능)",
            nurse_options,
            key="chat_nurse_select",
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if chat_nurse_label:
            chat_nurse_id = chat_nurse_label.split(" - ")[0]
            chat_nurse_row = clustered_df[clustered_df["nurse_id"].astype(str) == chat_nurse_id].iloc[0]

            # 기준 간호사 상세
            cols_for_report = [
                "nurse_id", "nurse_name", "level_name",
                "cluster_name", "is_outlier",
                "total_working_days", "total_night_days",
                "consecutive_working_days", "consecutive_night_shifts",
                "min_off_interval_hours",
                "ED_quick_return_total", "N_quick_return_total",
                "total_off_days", "total_night_days_risk",
                "total_off_days_risk",
                "consecutive_working_days_risk",
                "consecutive_night_shifts_risk",
                "min_off_interval_risk",
                "preferred_shift_ratio", "preferred_shift_ratio_risk",
                "preferred_duty_choice_ratio", "preferred_duty_choice_risk",
                "preferred_swing_risk",
                "level_night_ratio", "level_night_ratio_risk",
                "level_workingdays_ratio", "level_workingdays_ratio_risk",
            ]

            chat_info_lines = []
            for c in cols_for_report:
                if c in chat_nurse_row.index:
                    chat_info_lines.append(f"- {c}: {chat_nurse_row[c]}")
            chat_info_text = "\n".join(chat_info_lines)

            st.markdown(f"선택된 간호사: **{chat_nurse_row['nurse_name']}**")

            # 기존 대화 출력
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_question = st.chat_input(
                "부서 전체에 대해 물어봐도 되고, 두 사람 비교나 outlier/클러스터 관련 질문을 하셔도 됩니다."
            )

            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})

                system_prompt_chat = """
당신은 간호사 근무표 분석 도구의 AI 챗봇입니다.
입력에는 부서 전체 간호사들의 요약 정보와,
선택된 간호사의 상세 분석 결과가 함께 제공됩니다.

[위험도 라벨 규칙]
- risk 값: critical, moderate, low, no, no_preference, no_request.
- critical: 고위험 / 즉각적인 개선 필요.
- moderate: 중등도 위험 / 주의 필요.
- low: 경미한 위험 또는 약한 불균형.
- no 계열: 정상 범위, 위험 요인 아님.

[해석 규칙]
1. 위험도 판단은 항상 *_risk 컬럼을 기준으로 하세요.
2. risk가 'no'인 항목을 위험하다고 표현하지 마세요.
   필요하면 '정상 범위입니다' 정도로만 설명하세요.
3. 사용자가 '누가 더 힘든 스케줄인지', '누가 가장 위험한지',
   '누가 나와 패턴이 비슷한지'를 물으면,
   각 간호사의 *_risk 값과 outlier/cluster 정보를 비교해 답하세요.
4. 제공된 데이터에 없는 사실은 추측하지 말고
   '데이터에 없는 정보입니다'라고 답하세요.
5. 사용자에게는 'critical' 등의 영어 대신
   '고위험', '중등도 위험', '저위험', '정상'과 같은 한글 표현을 사용하세요.
6. level_name(novice/competence/leader)은 각각
   '저연차', '중간연차', '고연차'를 의미하므로,
   설명 시 한국어로 표현하세요.

항상 환자안전과 공정성 관점에서 친절하게, 한국어 존댓말로 설명하세요.
"""

                base_context = f"""
[부서 전체 간호사 스케줄 요약]

{all_nurses_summary}

[선택된 간호사의 상세 분석 값]

{chat_info_text}
"""

                user_prompt_chat = f"""
위의 정보를 참고하여, 다음 질문에 답변해 주세요.

질문: {user_question}
"""

                with st.spinner("AI가 답변을 생성하는 중입니다..."):
                    chat_answer = call_llm(system_prompt_chat, base_context + "\n\n" + user_prompt_chat)

                st.session_state.chat_history.append({"role": "assistant", "content": chat_answer})

                with st.chat_message("assistant"):
                    st.markdown(chat_answer)


# ============================================================
# 3. Risk Dashboard 탭
# ============================================================
with tab_risk:
    st.subheader("📊 Risk Dashboard")

    if (staffing_df is None) or (monthly_features_df is None):
        st.info("Risk Dashboard를 보려면 **Schedule 탭에서 근무표를 업로드**해 주세요.")
    else:
        # -------------------------
        # 3-1. 날짜별 인력 기준 충족 여부 (기존 그대로)
        # -------------------------
        st.markdown("### 📈 날짜별 인력 기준 충족 여부")

        plot_df = staffing_df.copy()
        plot_df["overall_risk_score"] = plot_df["overall_staffing_risk"].map(RISK_SCORE)

        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="날짜"),
                y=alt.Y(
                    "overall_risk_score:Q",
                    title="위험도 수준",
                    scale=alt.Scale(domain=[0, 3]),
                    axis=alt.Axis(
                        values=[0, 1, 2, 3],
                        labelExpr="datum.value == 0 ? 'no' : datum.value == 1 ? 'low' : datum.value == 2 ? 'moderate' : 'critical'"
                    ),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="날짜"),
                    alt.Tooltip("overall_staffing_risk:N", title="최고 위험도"),
                ],
            )
        )

        st.altair_chart(chart, use_container_width=True)

        # 날짜 선택해서 상세 텍스트 보여주기
        try:
            date_options = plot_df["date"].dt.date.astype(str).tolist()
        except Exception:
            date_options = plot_df["date"].astype(str).tolist()

        selected_date_str = st.selectbox("상세 인력 정보를 볼 날짜를 선택하세요", date_options)
        day_row = plot_df[plot_df["date"].astype(str).str.contains(selected_date_str)].iloc[0]

        st.markdown(f"**[{selected_date_str}] 인력 기준 상세**")

        for code, label in SHIFT_MAP.items():
            risk = day_row.get(f"{label}_risk", "no")
            shortage = int(day_row.get(f"{label}_shortage", 0))
            if risk == "no":
                text = "기준 충족"
            else:
                text = f"{risk_to_ko(risk)} / {shortage}명 부족"
            st.write(f"- {label}: {text}")

        st.write("---")

        # -------------------------
        # 3-2. 환자 안전 위험도 (개인 기준) – 참고 코드 버전
        # -------------------------
        st.subheader("🛡️ 환자 안전 위험도 (개인 기준)")

        cols = [
            "nurse_id",
            "nurse_name",
            "level",
            "ED_quick_return_risk",
            "N_quick_return_risk",
            "consecutive_working_days",
            "consecutive_working_days_risk",
            "consecutive_night_shifts",
            "consecutive_night_shifts_risk",
        ]
        risk_df = monthly_features_df[cols].copy()

        risk_df["연차"] = risk_df["level"].map(LEVEL_KO)
        risk_df["E→D 파행근무 위험도"] = risk_df["ED_quick_return_risk"].apply(risk_to_ko)
        risk_df["N 파행근무 위험도"] = risk_df["N_quick_return_risk"].apply(risk_to_ko)
        risk_df["연속 근무 일 수"] = risk_df["consecutive_working_days"]
        risk_df["연속 근무 위험도"] = risk_df["consecutive_working_days_risk"].apply(
            risk_to_ko
        )
        risk_df["연속 N근무 일 수"] = risk_df["consecutive_night_shifts"]
        risk_df["연속 N근무 위험도"] = risk_df["consecutive_night_shifts_risk"].apply(
            risk_to_ko
        )

        display_cols = [
            "nurse_id",
            "nurse_name",
            "연차",
            "E→D 파행근무 위험도",
            "N 파행근무 위험도",
            "연속 근무 일 수",
            "연속 근무 위험도",
            "연속 N근무 일 수",
            "연속 N근무 위험도",
        ]
        st.dataframe(risk_df[display_cols], use_container_width=True)

        st.markdown("#### 📊 연속 근무/연속 N근무 분포 (막대그래프)")

        bar_df = risk_df[["nurse_name", "연속 근무 일 수", "연속 N근무 일 수"]].set_index(
            "nurse_name"
        )
        st.bar_chart(bar_df)

        # -------------------------
        # 3-3. 위험도 Heatmap (가로축 한글로)
        # -------------------------
        st.markdown("#### 🔥 위험도 Heatmap")

        heat_df = monthly_features_df[
            ["nurse_id", "nurse_name",
             "ED_quick_return_risk", "N_quick_return_risk",
             "consecutive_working_days_risk", "consecutive_night_shifts_risk"]
        ].copy()

        heat_df = heat_df.rename(columns={"nurse_id": "ID", "nurse_name": "간호사"})
        heat_long = heat_df.melt(
            id_vars=["ID", "간호사"],
            var_name="지표",
            value_name="risk",
        )

        # 지표 이름을 한글로 매핑
        label_map = {
            "ED_quick_return_risk": "E→D 파행근무 위험도",
            "N_quick_return_risk": "N 파행근무 위험도",
            "consecutive_working_days_risk": "연속 근무 위험도",
            "consecutive_night_shifts_risk": "연속 N근무 위험도",
        }
        heat_long["지표"] = heat_long["지표"].map(label_map).fillna(heat_long["지표"])

        heat_long["risk_score"] = heat_long["risk"].map(RISK_SCORE)

        heat_chart = (
            alt.Chart(heat_long)
            .mark_rect()
            .encode(
                x=alt.X("지표:N", title="지표"),
                y=alt.Y("간호사:N", title="간호사"),
                color=alt.Color(
                    "risk_score:Q",
                    scale=alt.Scale(domain=[0, 3], range=["#e0f7fa", "#80deea", "#ffb74d", "#e53935"]),
                    legend=alt.Legend(title="위험도(0~3)"),
                ),
                tooltip=["ID", "간호사", "지표", "risk"],
            )
        )

        st.altair_chart(heat_chart, use_container_width=True)
        # (요청에 따라 E→D/N 파행근무 개별 그래프는 삭제)


# ============================================================
# 4. Fairness Dashboard 탭 (참고 코드 버전 그대로)
# ============================================================
with tab_fairness:
    st.subheader("⚖️ 공정성 대시보드")

    if clustered_df is None:
        st.info("공정성 분석을 보려면 **Schedule 탭에서 선호근무표까지 업로드**해 주세요.")
    else:
        # 1) 공정성 비교표
        st.markdown("### 1) 공정성 비교표")

        fair_cols = [
            "nurse_id",
            "nurse_name",
            "level",
            "preferred_shift_ratio",
            "preferred_shift_ratio_risk",
            "preferred_duty_choice_ratio",
            "preferred_duty_choice_risk",
            "preferred_swing_risk",
            "level_workingdays_ratio",
            "level_workingdays_ratio_risk",
            "level_night_ratio",
            "level_night_ratio_risk",
            "total_working_days",
            "total_off_days",
            "total_off_days_risk",
            "total_night_days",
            "total_night_days_risk",
            "min_off_interval_risk",
            "preferred_shift_total",
            "preferred_shift_matched",
            "preferred_duty_requests",
            "preferred_duty_matched",
        ]

        fair_df = clustered_df[fair_cols].copy()

        # ratio를 "분수" 형태로 표현
        def shift_ratio_str(row):
            total = row["preferred_shift_total"]
            matched = row["preferred_shift_matched"]
            if pd.isna(total) or total == 0:
                return "-"
            return f"{int(matched)}/{int(total)}"

        def duty_ratio_str(row):
            total = row["preferred_duty_requests"]
            matched = row["preferred_duty_matched"]
            if total == 0:
                return "-"
            return f"{int(matched)}/{int(total)}"

        fair_df["preferred_shift_ratio"] = fair_df.apply(shift_ratio_str, axis=1)
        fair_df["preferred_duty_choice_ratio"] = fair_df.apply(
            duty_ratio_str, axis=1
        )

        fair_df["연차"] = fair_df["level"].map(LEVEL_KO)
        fair_df["희망근무 반영율"] = fair_df["preferred_shift_ratio"]
        fair_df["희망근무 반영 위험도"] = fair_df["preferred_shift_ratio_risk"].apply(
            risk_to_ko
        )
        fair_df["선호근태 반영율"] = fair_df["preferred_duty_choice_ratio"]
        fair_df["선호근태 반영 위험도"] = fair_df[
            "preferred_duty_choice_risk"
        ].apply(risk_to_ko)
        fair_df["혼합교대 선호 반영 위험도"] = fair_df["preferred_swing_risk"].apply(
            risk_to_ko
        )
        fair_df["연차 대비 근무일수 비율"] = fair_df["level_workingdays_ratio"]
        fair_df["연차 대비 근무일수 위험도"] = fair_df[
            "level_workingdays_ratio_risk"
        ].apply(risk_to_ko)
        fair_df["연차 대비 N근무 비율"] = fair_df["level_night_ratio"]
        fair_df["연차 대비 N근무 위험도"] = fair_df["level_night_ratio_risk"].apply(
            risk_to_ko
        )
        fair_df["총 근무일수"] = fair_df["total_working_days"]
        fair_df["총 OFF 일수"] = fair_df["total_off_days"]
        fair_df["총 OFF 위험도"] = fair_df["total_off_days_risk"].apply(risk_to_ko)
        fair_df["총 N근무 일수"] = fair_df["total_night_days"]
        fair_df["총 N근무 위험도"] = fair_df["total_night_days_risk"].apply(risk_to_ko)
        fair_df["최소 휴식시간 위험도"] = fair_df["min_off_interval_risk"].apply(
            risk_to_ko
        )

        display_cols = [
            "nurse_id",
            "nurse_name",
            "연차",
            "희망근무 반영율",
            "희망근무 반영 위험도",
            "선호근태 반영율",
            "선호근태 반영 위험도",
            "혼합교대 선호 반영 위험도",
            "연차 대비 근무일수 비율",
            "연차 대비 근무일수 위험도",
            "연차 대비 N근무 비율",
            "연차 대비 N근무 위험도",
            "총 근무일수",
            "총 OFF 일수",
            "총 OFF 위험도",
            "총 N근무 일수",
            "총 N근무 위험도",
            "최소 휴식시간 위험도",
        ]
        st.dataframe(fair_df[display_cols], use_container_width=True)

        st.write("---")
        st.markdown("### 2) 개별 간호사 공정성 분석")

        nurse_options = fair_df["nurse_id"].astype(str) + " - " + fair_df["nurse_name"]
        selected_label = st.selectbox(
            "분석할 간호사를 선택해주세요", nurse_options
        )
        selected_id = selected_label.split(" - ")[0]
        row = fair_df[fair_df["nurse_id"].astype(str) == selected_id].iloc[0]
        raw_row = clustered_df[clustered_df["nurse_id"].astype(str) == selected_id].iloc[0]

        st.markdown(f"#### 👩‍⚕️ {row['nurse_name']} 간호사 공정성 분석")

        # ◆ 선호 반영율
        st.markdown("**◆ 선호 반영율**")

        # 희망근무
        pref_shift_type = raw_row.get("preferred_shift_type_code", math.nan)
        shift_matched = int(raw_row.get("preferred_shift_matched", 0))
        shift_total = int(raw_row.get("preferred_shift_total", 0))
        if pd.isna(pref_shift_type) or shift_total == 0:
            txt_shift = "· 희망근무를 별도로 신청하지 않았습니다."
        else:
            shift_code = int(pref_shift_type)
            shift_name = SHIFT_MAP.get(shift_code, str(shift_code))
            txt_shift = (
                f"· 희망근무 반영율: {shift_total}일 중 {shift_matched}일 "
                f"(주로 {shift_name} 근무를 선호)"
            )

        # 선호근태
        duty_req = int(raw_row.get("preferred_duty_requests", 0))
        duty_match = int(raw_row.get("preferred_duty_matched", 0))
        if duty_req == 0:
            txt_duty = "· 선호근태를 별도로 신청하지 않았습니다."
        else:
            txt_duty = f"· 선호근태 반영율: {duty_req}일 중 {duty_match}일"

        # 혼합교대
        swing_flag = bool(raw_row.get("preferred_swing_flag", False))
        swing_risk = raw_row.get("preferred_swing_risk", "no")
        if not swing_flag:
            txt_swing = "· 혼합교대를 특별히 선호하지 않습니다."
        else:
            if swing_risk == "no":
                txt_swing = "· 혼합교대 선호 반영: 선호가 대부분 반영되어 있습니다."
            else:
                txt_swing = "· 혼합교대 선호 반영: 선호가 충분히 반영되지 않았습니다."

        st.markdown("\n".join([txt_shift, txt_duty, txt_swing]))

        # ◆ 연차 기반 공정성
        st.markdown("**◆ 연차 기반 공정성**")

        def mag_from_risk(r):
            if r == "critical":
                return "매우 많음"
            elif r == "moderate":
                return "많음"
            elif r == "low":
                return "약간 많음"
            else:
                return "비슷하거나 적음"

        night_ratio_risk = raw_row.get("level_night_ratio_risk", "no")
        work_ratio_risk = raw_row.get("level_workingdays_ratio_risk", "no")

        txt_night = (
            "· 다른 연차군과 비교했을 때 N 근무 일 수가 "
            f"{mag_from_risk(night_ratio_risk)} 수준입니다."
        )
        txt_work = (
            "· 다른 연차군과 비교했을 때 전체 근무 일 수가 "
            f"{mag_from_risk(work_ratio_risk)} 수준입니다."
        )
        st.markdown("\n".join([txt_night, txt_work]))

        # ◆ OFF / N / Interval
        st.markdown("**◆ OFF / N / Interval**")

        off_days = int(raw_row.get("total_off_days", 0))
        off_risk = raw_row.get("total_off_days_risk", "no")
        night_days = int(raw_row.get("total_night_days", 0))
        night_days_risk = raw_row.get("total_night_days_risk", "no")
        min_interval = raw_row.get("min_off_interval_hours", None)
        min_interval_risk = raw_row.get("min_off_interval_risk", "no")

        txt_off = f"· 총 OFF 일 수: {off_days}일 – {risk_to_ko(off_risk)}"
        txt_night_days = f"· 총 N 근무 일 수: {night_days}일 – {risk_to_ko(night_days_risk)}"
        if min_interval is None or (isinstance(min_interval, float) and math.isnan(min_interval)):
            txt_int = "· 근무 간 최소 휴식 시간: 계산 불가(근무 패턴이 충분하지 않음)"
        else:
            txt_int = (
                f"· 근무 간 최소 휴식 시간: {round(float(min_interval), 1)}시간 – "
                f"{risk_to_ko(min_interval_risk)}"
            )

        st.markdown("\n".join([txt_off, txt_night_days, txt_int]))


# ============================================================
# 5. AI Report 탭 (클러스터링+outlier 설명 추가)
# ============================================================
with tab_report:
    if clustered_df is None:
        st.subheader("🧠 AI 기반 스케줄 해석 리포트")
        st.info("AI 리포트를 보려면 **Schedule 탭에서 선호근무표까지 업로드**해 주세요.")
    else:
        # -------------------------
        # 5-1. 클러스터링과 outlier 다이어그램 + 설명
        # -------------------------
        st.markdown("### 클러스터링과 outlier")

        # 다이어그램: Cluster별 간호사 이름 나열
        cluster_names = sorted(clustered_df["cluster_name"].unique())
        diag_lines = ["```"]
        for cname in cluster_names:
            names = clustered_df[clustered_df["cluster_name"] == cname]["nurse_name"].tolist()
            diag_lines.append(f"{cname}:")
            if names:
                for n in names:
                    diag_lines.append(f"  - {n}")
            else:
                diag_lines.append("  (간호사 없음)")
            diag_lines.append("")
        diag_lines.append("```")
        st.markdown("\n".join(diag_lines))

        # 간단한 특성 설명 (총 근무일수 / N 근무 기준으로)
        metric_cols = [c for c in ["total_working_days", "total_night_days"] if c in clustered_df.columns]
        explanation_lines = []
        if metric_cols:
            overall_mean = clustered_df[metric_cols].mean()

            def compare_word(val, overall):
                if pd.isna(val) or pd.isna(overall):
                    return "비슷한"
                diff = val - overall
                if diff > 1:
                    return "더 많은"
                elif diff < -1:
                    return "더 적은"
                else:
                    return "비슷한"

            for cname in cluster_names:
                sub = clustered_df[clustered_df["cluster_name"] == cname]
                means = sub[metric_cols].mean()
                if "total_working_days" in metric_cols:
                    w_phrase = compare_word(means["total_working_days"], overall_mean["total_working_days"])
                else:
                    w_phrase = "비슷한"
                if "total_night_days" in metric_cols:
                    n_phrase = compare_word(means["total_night_days"], overall_mean["total_night_days"])
                else:
                    n_phrase = "비슷한"

                explanation_lines.append(
                    f"- **{cname}**: 전체 평균과 비교했을 때 총 근무일 수가 {w_phrase} 편이고, "
                    f"N 근무 횟수가 {n_phrase} 편인 간호사들이 모여 있습니다."
                )

        out_df = clustered_df[clustered_df["is_outlier"] == "outlier"]
        if len(out_df) > 0:
            out_names = ", ".join(out_df["nurse_name"].astype(str).tolist())
            out_text = (
                f"\n\n이상치(outlier)로 분류된 간호사는 **{out_names}** 입니다. "
                "이 간호사들은 총 근무일 수, 야간 근무, quick return, OFF 일수 등에서 "
                "다른 간호사 그룹과 비교했을 때 상대적으로 크게 벗어난 패턴을 보여 "
                "스케줄 조정이나 추가적인 지원이 필요한지 검토해 볼 수 있습니다."
            )
        else:
            out_text = (
                "\n\n현재 설정된 기준(IsolationForest)에 따라 이상치(outlier)로 분류된 간호사는 없습니다."
            )

        st.markdown(
            "클러스터는 총 근무일 수, N 근무 횟수, quick return 발생, "
            "선호근무 반영률 등의 패턴이 비슷한 간호사들을 하나의 그룹으로 묶은 것입니다.\n\n"
            + ("\n".join(explanation_lines) if explanation_lines else "")
            + out_text
        )

        st.write("---")

        # -------------------------
        # 5-2. 기존 AI 기반 스케줄 해석 리포트
        # -------------------------
        st.subheader("🧠 AI 기반 스케줄 해석 리포트")

        nurse_options = clustered_df["nurse_id"].astype(str) + " - " + clustered_df["nurse_name"].astype(str)
        selected_label = st.selectbox(
            "리포트를 보고 싶은 간호사를 선택하세요", nurse_options
        )

        cols_for_report = [
            "nurse_id", "nurse_name", "level_name",
            "cluster_name", "is_outlier",
            "total_working_days", "total_night_days",
            "consecutive_working_days", "consecutive_night_shifts",
            "min_off_interval_hours",
            "ED_quick_return_total", "N_quick_return_total",
            "total_off_days", "total_night_days_risk",
            "total_off_days_risk",
            "consecutive_working_days_risk",
            "consecutive_night_shifts_risk",
            "min_off_interval_risk",
            "preferred_shift_ratio", "preferred_shift_ratio_risk",
            "preferred_duty_choice_ratio", "preferred_duty_choice_risk",
            "preferred_swing_risk",
            "level_night_ratio", "level_night_ratio_risk",
            "level_workingdays_ratio", "level_workingdays_ratio_risk",
        ]

        if selected_label:
            selected_id = selected_label.split(" - ")[0]
            nurse_row = clustered_df[clustered_df["nurse_id"].astype(str) == selected_id].iloc[0]

            if st.button("이 간호사의 AI 리포트 생성하기"):
                info_lines = []
                for c in cols_for_report:
                    if c in nurse_row.index:
                        info_lines.append(f"- {c}: {nurse_row[c]}")
                info_text = "\n".join(info_lines)

                system_prompt = """
당신은 간호사 근무 스케줄 분석을 돕는 AI 어시스턴트입니다.
입력으로는 각 간호사별 다양한 수치 지표와 함께,
각 지표에 대한 위험도 라벨(*_risk)이 함께 제공됩니다.

[위험도 라벨 규칙]
- risk 값: critical, moderate, low, no, no_preference, no_request.
- critical: 고위험, 즉각적인 개선이 필요함.
- moderate: 중등도 위험, 주의 깊은 모니터링과 조정이 필요함.
- low: 경미한 위험 또는 약한 불균형(참고 수준).
- no / no_preference / no_request: 정상 범위이며 위험 요인으로 취급하지 않음.

[해석 규칙 – 매우 중요]
1. '환자안전 관점 주요 위험 요인'에는 critical / moderate 수준만 포함하십시오.
   low나 no인 지표는 여기에 넣지 마세요.
2. 'no'인 지표는 '정상 범위'라고만 간단히 언급하거나, 필요 없으면 생략해도 됩니다.
3. 연속근무일수, 야간근무일수 등 수치는 반드시 대응하는 *_risk 값과 함께 해석해야 합니다.
4. 공정성 관련 지표(level_*_ratio_risk, preferred_*_risk 등)는
   '공정성/형평성 관점 주요 이슈'에서 다루고, critical / moderate 위주로 설명하세요.
5. 사용자에게 설명할 때 'critical' 같은 영어를 그대로 쓰지 말고
   '고위험', '중등도 위험', '저위험', '정상'과 같이 한글로 표현하십시오.
6. level_name(novice/competence/leader)은 각각
   '저연차', '중간연차', '고연차'를 의미합니다. 설명 시 한국어 표현을 사용하세요.

[지표 의미(간략)]
- total_off_days_risk: 한 달 OFF 일수가 적절한지(너무 적으면 위험).
- total_night_days_risk: 한 달 야간 근무일수의 부담 정도.
- consecutive_working_days_risk: 연속 근무 일수의 과도 여부.
- consecutive_night_shifts_risk: 연속 야간 근무 과도 여부.
- min_off_interval_risk: 근무 사이 최소 휴식 시간 위반 여부.
- ED_quick_return_risk / N_quick_return_risk:
  교대 간 간격이 너무 짧은 quick return 패턴의 위험.

[답변 형식]
1) 환자안전 관점 주요 위험 요인 (critical / moderate만)
2) 공정성/형평성 관점 주요 이슈
3) 스케줄 조정 시사점 (야간/연속근무 조정, 선호근무 반영 등)

한국어 존댓말을 사용하고, 5~10문장 정도로 간결하게 작성하세요.
"""

                user_prompt = f"""
다음은 한 간호사의 근무 스케줄 분석 결과입니다.

{info_text}

이 정보를 바탕으로,
1) 환자안전 관점 주요 위험 요인
2) 공정성/형평성 관점 주요 이슈
3) 스케줄 조정 시사점

을 항목형 요약 + 짧은 설명으로 정리해 주세요.
"""

                with st.spinner("AI가 리포트를 작성하는 중입니다..."):
                    answer = call_llm(system_prompt, user_prompt)

                st.markdown(answer)
