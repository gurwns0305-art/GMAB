# # GARCH–Duan(1,1): 정리된 실행 순서
# 
# **원본 `GARCH.ipynb`의 중복 셀을 정리하여 재배치한 버전입니다.** 아래에서 위로 순차 실행하세요. 기존 셀 코드는 변경하지 않았으며 출력은 지웠습니다. 원본 파일은 그대로 보존했습니다.
# 
# 1. 데이터 불러오기 / NAV 전처리 / 로그수익률 Train·Test 분리
# 2. GARCH 필터 정의 / 다중 시작점 확인 / MLE 추정
# 3. 잔차 진단 / Q측도 동학 구성
# 4. Radon–Nikodym 확률가중치 → 조건부분산 보존 → 복수 변동성 수준의 국소 검증
# 5. 1·3·10년 할인 펀드 가치 마팅게일 검증
# 
# **유의:** 본 노트북에는 아직 GMAB 지급액의 본격적인 가격계산 셀이 없습니다. `valuation_date_T_minus_1` 날짜 매핑도 공시 규칙의 독립 확인이 필요한 가정입니다.
# 
# ---
# 
# ## 데이터 교체본
# 기준 원본: 사용자가 지정한 GARCH_reordered.ipynb. 모델·검증·가격계산·시뮬레이션 설정은 유지했습니다.
# 변경 코드 셀은 데이터 로딩(1), 학습/테스트 날짜(3), 기존 가격 변수 연결 1줄(22)뿐입니다.
# 학습 2012~2014, 테스트 2015~2025. 테스트 전체는 양끝 포함 11개 연도이며, 기존 MC 계약기간 T=10은 그대로입니다.
# 원본의 공시 기준일−1달력일 변환과 평일 필터·252일 연율화를 그대로 보존했습니다.
# 이 날짜 매핑은 원본 가정의 유지이며 실제 거래일/공표시각을 새로 검증한 것이 아닙니다.
# 앞서 새로 작성했던 다른 GMAB 노트북의 기능/모형은 합치지 않았습니다.
# 사망률 CSV를 함께 동봉했으며 해당 파일의 10년 생존확률은 약 0.9092031803입니다.
# 이전 데이터에서 나온 출력은 지웠습니다. 모두 실행하면 새 데이터 결과를 생성합니다.

# ==========================================
# Cell 1. Import Libraries & Data Loading
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

# VS Code: open this package folder, then run this notebook.
# Colab: upload both bundled CSV files when prompted.
CSV_PATH = "KLVL5225111_NAV_date_aligned.csv"
if not Path(CSV_PATH).is_file():
    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(
            "Open the extracted package folder in VS Code, or set CSV_PATH to the bundled aligned CSV."
        ) from exc
    uploaded = files.upload()
    candidates = []
    for name in uploaded:
        if name.lower().endswith(".csv"):
            header = pd.read_csv(name, nrows=0, encoding="utf-8-sig").columns
            if {"disclosure_date", "valuation_date_T_minus_1", "NAV_KRW", "date_type"}.issubset(header):
                candidates.append(name)
    if len(candidates) != 1:
        raise ValueError("Upload one NAV aligned CSV and the bundled male_60_69_mortality.csv.")
    CSV_PATH = candidates[0]
raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("데이터 크기:", raw.shape)
print("컬럼명:", raw.columns.tolist())
display(raw.head())


# ==========================================
# Cell 2. NAV Data Preprocessing
# ==========================================

# 1. 필요한 컬럼 추출
nav = raw[["disclosure_date", "valuation_date_T_minus_1", "NAV_KRW", "date_type"]].copy()

# 2. 컬럼명 정리
nav = nav.rename(columns={
    "valuation_date_T_minus_1": "date",
    "NAV_KRW": "NAV"
})

# 3. 날짜 변환
nav["date"] = pd.to_datetime(nav["date"])
nav["disclosure_date"] = pd.to_datetime(nav["disclosure_date"])

# 4. NAV 숫자 변환
nav["NAV"] = pd.to_numeric(nav["NAV"], errors="coerce")

# 5. 결측치 및 중복 확인
print("===== 데이터 품질 점검 =====")
print("전체 관측치:", len(nav))
print("날짜 결측치:", nav["date"].isna().sum())
print("NAV 결측치:", nav["NAV"].isna().sum())
print("중복 날짜:", nav["date"].duplicated().sum())

# 6. 결측 및 비정상 값 제거
nav = nav.dropna(subset=["date", "NAV"])
nav = nav[nav["NAV"] > 0]

nav = (
    nav.sort_values("date")
       .drop_duplicates(subset="date")
       .reset_index(drop=True)
)

# 7. 평일 데이터 추출
nav_weekday = nav[nav["date"].dt.dayofweek < 5].copy().reset_index(drop=True)

# 8. 결과 확인
print("\n===== 전처리 결과 =====")
print("전체 데이터:", len(nav))
print("평일 데이터:", len(nav_weekday))
print("주말 제외:", len(nav) - len(nav_weekday))
print("분석 시작일:", nav_weekday["date"].min().date())
print("분석 종료일:", nav_weekday["date"].max().date())

display(nav_weekday.head())

# ==========================================
# Cell 3. Log Returns & Train/Test Split
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# 1. 날짜순 정렬
data = nav_weekday.sort_values("date").copy().reset_index(drop=True)

# 2. 일별 로그수익률 계산
data["log_return"] = np.log(data["NAV"] / data["NAV"].shift(1))
data["log_return_pct"] = data["log_return"] * 100

# 첫 번째 결측 수익률 제거
data = data.dropna(subset=["log_return"]).reset_index(drop=True)

# 3. Train/Test 분할
# Period selection only: returns are calculated before filtering, retaining the actual prior NAV.
TRAIN_START = pd.Timestamp("2012-01-01")
TRAIN_END = pd.Timestamp("2014-12-31")
TEST_START = pd.Timestamp("2015-01-01")
TEST_END = pd.Timestamp("2025-12-31")
split_date = TEST_START

data = data[data["date"].between(TRAIN_START, TEST_END)].copy().reset_index(drop=True)
train = data[data["date"].between(TRAIN_START, TRAIN_END)].copy()
test = data[data["date"].between(TEST_START, TEST_END)].copy()
assert not train.empty and not test.empty
assert train["date"].max() < test["date"].min()

# 4. GARCH 입력 데이터 생성
y_train = train["log_return_pct"].to_numpy()
y_test = test["log_return_pct"].to_numpy()

# 5. 기초 통계량 확인
summary = pd.DataFrame({
    "구분": ["Train", "Test"],
    "관측치": [len(train), len(test)],
    "시작일": [train["date"].min().date(), test["date"].min().date()],
    "종료일": [train["date"].max().date(), test["date"].max().date()],
    "평균 로그수익률(%)": [y_train.mean(), y_test.mean()],
    "일별 표준편차(%)": [y_train.std(ddof=1), y_test.std(ddof=1)]
})
print("===== Train/Test 분할 결과 =====")
display(summary)

# 6. 시계열 그래프
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(data["date"], data["NAV"], linewidth=1)
axes[0].axvline(split_date, color="red", linestyle="--", label="Train/Test Split")
axes[0].set_title("AIA Equity Fund NAV")
axes[0].set_ylabel("NAV (KRW)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(data["date"], data["log_return_pct"], linewidth=0.7)
axes[1].axvline(split_date, color="red", linestyle="--", label="Train/Test Split")
axes[1].set_title("Daily Log Returns")
axes[1].set_ylabel("Log Return (%)")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==============================================
# Cell 4. GARCH-Duan(1,1) Conditional Variance
# ==============================================

import numpy as np

# ----------------------------------------------
# 1. 기본 설정
# ----------------------------------------------

DAYS = 252

# 연속복리 기준 연 3% 무위험금리 (임시 가정)
rf_annual = 0.03

# 일별 무위험금리, 퍼센트 단위
rf_daily_pct = 100 * rf_annual / DAYS


# ----------------------------------------------
# 2. GARCH-Duan Filter
# ----------------------------------------------

def garch_duan_filter(
    params,
    returns,
    rf_pct,
    h0=None
):
    """
    GARCH-Duan(1,1) conditional variance filter

    Parameters
    ----------
    params : (omega, alpha, beta, lambda)
        omega : %^2/day
        alpha : dimensionless
        beta  : dimensionless
        lambda: dimensionless

    returns : array
        Daily log returns in percent units

    rf_pct : float
        Daily risk-free log return in percent units

    h0 : float or None
        Initial conditional variance in %^2

    Returns
    -------
    h : array
        Conditional variance

    z : array
        Standardized innovations

    mean : array
        Conditional mean log return
    """

    omega, alpha, beta, lam = params

    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    # 파라미터 유효성 확인
    if (
        omega <= 0
        or alpha < 0
        or beta < 0
        or alpha + beta >= 1
    ):
        raise ValueError("Invalid GARCH parameters")

    if n == 0:
        raise ValueError("Returns array is empty")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be finite")

    h = np.zeros(n)
    z = np.zeros(n)
    mean = np.zeros(n)

    # ------------------------------------------
    # 3. 초기 조건부분산
    # ------------------------------------------

    if h0 is None:
        h[0] = omega / (1 - alpha - beta)
    else:
        if not np.isfinite(h0) or h0 <= 0:
            raise ValueError("h0 must be positive")
        h[0] = h0

    # ------------------------------------------
    # 4. 재귀적 조건부분산 계산
    # ------------------------------------------

    for t in range(n):

        # 조건부 평균: 퍼센트 단위
        # h/200 = 100 * (0.5 * h / 10000)
        mean[t] = (
            rf_pct
            - h[t] / 200
            + lam * np.sqrt(h[t])
        )

        # 표준화 잔차
        z[t] = (
            returns[t] - mean[t]
        ) / np.sqrt(h[t])

        # 다음 날 조건부분산
        if t < n - 1:
            h[t + 1] = (
                omega
                + alpha * h[t] * z[t]**2
                + beta * h[t]
            )

    return h, z, mean


# ----------------------------------------------
# 5. 임시 파라미터로 작동 확인
# ----------------------------------------------

# 아직 MLE 추정값이 아닙니다.
# 함수가 정상 작동하는지 확인하는 테스트용 값
test_params = (
    0.02,   # omega
    0.05,   # alpha
    0.90,   # beta
    0.03    # lambda
)

h_test, z_test, mean_test = garch_duan_filter(
    params=test_params,
    returns=y_train,
    rf_pct=rf_daily_pct
)

# ----------------------------------------------
# 6. 결과 확인
# ----------------------------------------------

print("===== GARCH-Duan Filter Test =====")

print(f"관측치 수: {len(y_train):,}")
print(f"초기 조건부분산: {h_test[0]:.6f} (%²)")
print(f"최종 조건부분산: {h_test[-1]:.6f} (%²)")

print(
    f"최종 연환산 조건부 변동성: "
    f"{np.sqrt(h_test[-1] * DAYS):.4f}%"
)

print(f"표준화 잔차 평균: {np.mean(z_test):.6f}")
print(f"표준화 잔차 표준편차: {np.std(z_test, ddof=1):.6f}")

assert np.all(np.isfinite(h_test))
assert np.all(h_test > 0)
assert np.all(np.isfinite(z_test))

print("\nGARCH-Duan 필터 정상 작동")

# ==============================================
# Cell 5-A. Multi-start Initial Values
# ==============================================

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

sample_var = np.var(y_train, ddof=1)

initial_guesses = []

# ----------------------------------------------
# 1. 다양한 지속성 구간 설정
# ----------------------------------------------

persistence_ranges = [
    (0.50, 0.70),   # 낮은 지속성
    (0.70, 0.85),   # 중간 지속성
    (0.85, 0.95),   # 높은 지속성
    (0.95, 0.995)   # 매우 높은 지속성
]

# ----------------------------------------------
# 2. 각 지속성 구간에서 초기값 생성
# ----------------------------------------------

N_PER_GROUP = 15

for lower, upper in persistence_ranges:

    for _ in range(N_PER_GROUP):

        # alpha + beta
        persistence = rng.uniform(lower, upper)

        # alpha
        alpha = rng.uniform(
            0.001,
            min(0.30, persistence - 0.001)
        )

        # beta
        beta = persistence - alpha

        # lambda
        lam = rng.uniform(-1.0, 1.0)

        # 장기분산 = 표본분산이 되도록 omega 설정
        omega = sample_var * (1 - persistence)

        initial_guesses.append([
            omega,
            alpha,
            beta,
            lam
        ])

# ----------------------------------------------
# 3. 생성 결과 확인
# ----------------------------------------------

initial_df = pd.DataFrame(
    initial_guesses,
    columns=["omega", "alpha", "beta", "lambda"]
)

initial_df["persistence"] = (
    initial_df["alpha"] + initial_df["beta"]
)

initial_df["long_run_var"] = (
    initial_df["omega"]
    / (1 - initial_df["persistence"])
)

print("생성된 초기값 개수:", len(initial_guesses))

display(initial_df.head(10))

print("\n파라미터 초기값 분포")
display(initial_df.describe().round(4))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sc = axes[0].scatter(
    initial_df["alpha"],
    initial_df["beta"],
    c=initial_df["persistence"],
    cmap="viridis",
    alpha=0.8
)

axes[0].plot(
    [0, 1],
    [1, 0],
    "r--",
    label="alpha + beta = 1"
)

axes[0].set_xlim(0, 0.35)
axes[0].set_ylim(0.15, 1.0)

axes[0].set_xlabel("alpha")
axes[0].set_ylabel("beta")
axes[0].set_title("Initial Alpha-Beta Values")
axes[0].legend()

plt.colorbar(sc, ax=axes[0], label="Persistence")

axes[1].hist(
    initial_df["lambda"],
    bins=15,
    edgecolor="black"
)

axes[1].set_title("Initial Lambda Values")
axes[1].set_xlabel("lambda")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# =====================================================
# Cell 5. GARCH-Duan(1,1) Multi-start MLE
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from IPython.display import display


# -----------------------------------------------------
# 1. 데이터 및 기본 설정
# -----------------------------------------------------

y = np.asarray(y_train, dtype=float)

assert np.all(np.isfinite(y))
assert len(y) > 100

sample_var = np.var(y, ddof=1)

N_STARTS = 60
MAX_ITER = 3000
SEED = 42

rng = np.random.default_rng(SEED)


# -----------------------------------------------------
# 2. Negative Log-Likelihood
# -----------------------------------------------------

def negative_log_likelihood(params, returns, rf_pct):

    omega, alpha, beta, lam = params

    if (
        omega <= 0
        or alpha < 0
        or beta < 0
        or alpha + beta >= 0.999
        or not np.all(np.isfinite(params))
    ):
        return 1e12

    try:
        with np.errstate(over="raise", invalid="raise"):

            h, z, mean = garch_duan_filter(
                params=params,
                returns=returns,
                rf_pct=rf_pct
            )

            if (
                np.any(h <= 0)
                or not np.all(np.isfinite(h))
                or not np.all(np.isfinite(z))
            ):
                return 1e12

            nll = 0.5 * np.sum(
                np.log(2 * np.pi)
                + np.log(h)
                + z**2
            )

            if not np.isfinite(nll):
                return 1e12

            return float(nll)

    except (ValueError, FloatingPointError, OverflowError):
        return 1e12


# -----------------------------------------------------
# 3. 다양한 초기값 생성
# -----------------------------------------------------

initial_guesses = []

# 지속성 수준을 다양하게 설정
persistence_ranges = [
    (0.50, 0.70),
    (0.70, 0.85),
    (0.85, 0.95),
    (0.95, 0.995)
]

for lower, upper in persistence_ranges:

    for _ in range(N_STARTS // len(persistence_ranges)):

        # GARCH 지속성
        rho0 = rng.uniform(lower, upper)

        # ARCH 계수
        alpha0 = rng.uniform(
            0.001,
            min(0.30, rho0 - 0.001)
        )

        # GARCH 계수
        beta0 = rho0 - alpha0

        # 위험프리미엄 계수
        lambda0 = rng.uniform(-1.0, 1.0)

        # 초기 장기분산도 다양화
        variance_multiplier = rng.uniform(0.5, 2.0)

        omega0 = (
            sample_var
            * variance_multiplier
            * (1 - rho0)
        )

        initial_guesses.append([
            omega0,
            alpha0,
            beta0,
            lambda0
        ])

print(f"생성된 초기값: {len(initial_guesses)}개")


# -----------------------------------------------------
# 4. 최적화 제약조건
# -----------------------------------------------------

bounds = [
    (1e-8, max(10 * sample_var, 1e-4)),  # omega
    (0.0, 0.998),                       # alpha
    (0.0, 0.998),                       # beta
    (-5.0, 5.0)                         # lambda
]

constraints = [
    {
        "type": "ineq",
        "fun": lambda p: 0.999 - p[1] - p[2]
    }
]


# -----------------------------------------------------
# 5. Multi-start MLE 실행
# -----------------------------------------------------

all_results = []
all_histories = {}

for i, x0 in enumerate(initial_guesses):

    history = []

    def callback(xk):

        nll = negative_log_likelihood(
            xk, y, rf_daily_pct
        )

        history.append({
            "iteration": len(history) + 1,
            "omega": xk[0],
            "alpha": xk[1],
            "beta": xk[2],
            "lambda": xk[3],
            "nll": nll
        })

    result = minimize(
        fun=negative_log_likelihood,
        x0=x0,
        args=(y, rf_daily_pct),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={
            "maxiter": MAX_ITER,
            "ftol": 1e-9,
            "disp": False
        }
    )

    feasible = (
        np.all(np.isfinite(result.x))
        and result.x[0] > 0
        and result.x[1] >= 0
        and result.x[2] >= 0
        and result.x[1] + result.x[2] < 0.999
    )

    converged = (
        result.success
        and feasible
        and np.isfinite(result.fun)
        and result.fun < 1e11
    )

    all_results.append({
        "start": i + 1,
        "success": converged,
        "nll": result.fun,
        "omega": result.x[0],
        "alpha": result.x[1],
        "beta": result.x[2],
        "lambda": result.x[3],
        "persistence_P": result.x[1] + result.x[2],
        "iterations": result.nit,
        "message": result.message,
        "optimizer_result": result
    })

    all_histories[i + 1] = pd.DataFrame(history)

    if (i + 1) % 10 == 0:
        print(f"진행: {i + 1}/{len(initial_guesses)}")


# -----------------------------------------------------
# 6. 수렴 결과 정리
# -----------------------------------------------------

results_df = pd.DataFrame(all_results)

successful = (
    results_df[results_df["success"]]
    .sort_values("nll")
    .reset_index(drop=True)
)

print("\n===== Optimization Summary =====")

print(f"총 초기값: {len(results_df)}")
print(f"성공한 최적화: {len(successful)}")
print(f"실패한 최적화: {len(results_df) - len(successful)}")

if successful.empty:
    raise RuntimeError(
        "정상적으로 수렴한 결과가 없습니다. "
        "초기값과 목적함수를 점검하세요."
    )


# -----------------------------------------------------
# 7. 최종 최적해 선택
# -----------------------------------------------------

best_row = successful.iloc[0]

best_start = int(best_row["start"])

best_result = best_row["optimizer_result"]

params_hat = best_result.x.copy()

omega_hat, alpha_hat, beta_hat, lambda_hat = params_hat

print("\n===== Final MLE Estimates =====")

parameter_table = pd.DataFrame({
    "Parameter": [
        "omega",
        "alpha",
        "beta",
        "lambda",
        "alpha + beta"
    ],
    "Estimate": [
        omega_hat,
        alpha_hat,
        beta_hat,
        lambda_hat,
        alpha_hat + beta_hat
    ]
})

display(parameter_table.round(6))

print(f"Best starting point: {best_start}")
print(f"Maximum Log-Likelihood: {-best_result.fun:.6f}")
print(f"Optimization success: {best_result.success}")


# -----------------------------------------------------
# 8. 여러 초기값의 최종 결과 비교
# -----------------------------------------------------

comparison_cols = [
    "start",
    "nll",
    "omega",
    "alpha",
    "beta",
    "lambda",
    "persistence_P",
    "iterations"
]

print("\n===== Top 10 Converged Results =====")

display(
    successful[comparison_cols]
    .head(10)
    .round(6)
)

# 최적 로그우도와 거의 일치하는 결과 수
NLL_TOL = 1e-3

near_best = successful[
    np.abs(successful["nll"] - best_result.fun) < NLL_TOL
].copy()

print(
    f"\n최고 NLL과 {NLL_TOL} 미만 차이의 결과:",
    len(near_best),
    "/",
    len(successful)
)

# 우도뿐 아니라 파라미터도 유사한지 검사
PARAM_TOL = 1e-2

param_columns = ["omega", "alpha", "beta", "lambda"]

if len(near_best) > 0:

    param_differences = np.abs(
        near_best[param_columns].to_numpy()
        - params_hat
    )

    same_parameters = np.all(
        param_differences < PARAM_TOL,
        axis=1
    )

    print(
        "우도와 파라미터가 모두 유사한 결과:",
        int(same_parameters.sum()),
        "/",
        len(successful)
    )


# -----------------------------------------------------
# 9. 수렴 과정 시각화
# -----------------------------------------------------

history_df = all_histories[best_start].copy()

history_df["param_change"] = (
    history_df[param_columns]
    .diff()
    .abs()
    .max(axis=1)
)

history_df["nll_change"] = (
    history_df["nll"]
    .diff()
    .abs()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    history_df["iteration"],
    history_df["nll"],
    marker="o",
    markersize=3
)

axes[0].set_title("Negative Log-Likelihood Convergence")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Negative Log-Likelihood")
axes[0].grid(alpha=0.3)

# 로그축을 위해 0보다 큰 변화량만 사용
valid_changes = history_df[
    history_df["param_change"] > 0
]

axes[1].semilogy(
    valid_changes["iteration"],
    valid_changes["param_change"],
    marker="o",
    markersize=3
)

axes[1].set_title("Parameter Change")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Maximum Absolute Change")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 10. 추정된 조건부분산 및 잔차
# -----------------------------------------------------

h_hat, z_hat, mean_hat = garch_duan_filter(
    params=params_hat,
    returns=y,
    rf_pct=rf_daily_pct
)

# 마지막 관측 충격까지 반영한 다음 시점 분산
h_next = (
    omega_hat
    + alpha_hat * h_hat[-1] * z_hat[-1]**2
    + beta_hat * h_hat[-1]
)

# P 측도 장기분산
long_run_var_P = (
    omega_hat / (1 - alpha_hat - beta_hat)
)

# Q 측도 분산 지속성
rho_Q = (
    alpha_hat * (1 + lambda_hat**2)
    + beta_hat
)

print("\n===== Model Properties =====")

print(f"P Persistence: {alpha_hat + beta_hat:.6f}")
print(f"Q Persistence: {rho_Q:.6f}")

print(
    f"P Long-run Volatility: "
    f"{np.sqrt(long_run_var_P * 252):.4f}%"
)

print(
    f"Next Annualized Conditional Volatility: "
    f"{np.sqrt(h_next * 252):.4f}%"
)

print(f"Standardized Residual Mean: {z_hat.mean():.6f}")
print(f"Standardized Residual Std: {z_hat.std(ddof=1):.6f}")

if rho_Q >= 1:
    print(
        "\nWARNING: Q-measure variance persistence >= 1. "
        "Long-horizon simulation requires additional stability checks."
    )
else:
    print("\nQ-measure variance persistence < 1")

print("\n===== 완료 =====")
print("최종 추정 파라미터는 params_hat에 저장되었습니다.")
print("다음 시점 조건부분산은 h_next에 저장되었습니다.")

# ==============================================
# Cell 6. GARCH-Duan Model Diagnostics
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from scipy.stats import jarque_bera
from IPython.display import display

# 셀 5에서 추정된 잔차
z = np.asarray(z_hat, dtype=float)

# 1. Ljung-Box 검정
lb_z = acorr_ljungbox(
    z,
    lags=[5, 10, 20],
    return_df=True
)

lb_z2 = acorr_ljungbox(
    z**2,
    lags=[5, 10, 20],
    return_df=True
)

# 2. ARCH-LM 검정
arch_lm, arch_p, arch_f, arch_fp = het_arch(
    z,
    nlags=10,
    ddof=0
)

# 3. Jarque-Bera 정규성 검정
jb = jarque_bera(z)

print("===== Ljung-Box: Standardized Residuals =====")
display(lb_z)

print("\n===== Ljung-Box: Squared Residuals =====")
display(lb_z2)

print("\n===== ARCH-LM Test =====")
print(f"LM Statistic: {arch_lm:.4f}")
print(f"p-value: {arch_p:.6g}")

print("\n===== Jarque-Bera Test =====")
print(f"JB Statistic: {jb.statistic:.4f}")
print(f"p-value: {jb.pvalue:.6g}")

# 4. 조건부 변동성
# h_hat은 퍼센트 단위 수익률의 일별 분산
conditional_vol = np.sqrt(h_hat * 252)

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

axes[0].plot(
    train["date"].to_numpy(),
    conditional_vol,
    label="GARCH Conditional Volatility"
)

axes[0].set_title("Annualized GARCH Conditional Volatility")
axes[0].set_ylabel("Volatility (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    train["date"].to_numpy(),
    z,
    linewidth=0.7
)

axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_title("Standardized Residuals")
axes[1].set_ylabel("Standardized Innovation")
axes[1].set_xlabel("Date")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ## Ljung-Box test H0: 잔차에 자기상관이 존재하지 않는다
# 
# ## ARCH-LM의 H0: 표준화 잔차에 지정한 시차까지 ARCH 효과가 없다
# 
# ## 자크-베라: 표준화 잔차는 정규분포를 따른다

from scipy.stats import skew, kurtosis

print(f"Skewness: {skew(z_hat, bias=False):.4f}")
print(f"Excess Kurtosis: {kurtosis(z_hat, fisher=True, bias=False):.4f}")

# # 잔차가 정규분포를 따르지는 않는 듯

# ==============================================
# Cell 7. Duan LRNVR - Risk-Neutral Dynamics
# ==============================================

import numpy as np
import pandas as pd
from IPython.display import display

# ----------------------------------------------
# 1. 추정 파라미터 불러오기
# ----------------------------------------------

omega_Q, alpha_Q, beta_Q, lambda_Q = params_hat

# 학습기간 종료 후 최초 조건부분산
# 단위: (%^2)
h0_Q = float(h_next)

# 무위험금리
R_ANNUAL = 0.03
DAYS = 252

# 일별 연속복리 무위험수익률 (%)
rf_daily_Q = 100 * R_ANNUAL / DAYS

# ----------------------------------------------
# 2. 위험중립 변동성 지속성
# ----------------------------------------------

rho_P = alpha_Q + beta_Q

rho_Q = (
    alpha_Q * (1 + lambda_Q**2)
    + beta_Q
)

print("===== GARCH-Duan Risk-Neutral Parameters =====")
print(f"omega   : {omega_Q:.8f}")
print(f"alpha   : {alpha_Q:.6f}")
print(f"beta    : {beta_Q:.6f}")
print(f"lambda  : {lambda_Q:.6f}")
print(f"P persistence : {rho_P:.6f}")
print(f"Q persistence : {rho_Q:.6f}")

if rho_Q >= 1:
    print(
        "WARNING: Q 측도에서 장기 평균분산의 "
        "안정성 조건을 만족하지 않습니다."
    )

# ----------------------------------------------
# 3. 위험중립 GARCH 한 단계 계산
# ----------------------------------------------

def duan_q_step(h_t, z_q, params, rf_pct):

    omega, alpha, beta, lam = params

    # Q측도 일별 로그수익률 (%)
    return_pct = (
        rf_pct
        - h_t / 200
        + np.sqrt(h_t) * z_q
    )

    # Q측도 다음 시점 조건부분산 (%^2)
    h_next_q = (
        omega
        + alpha * h_t * (z_q - lam)**2
        + beta * h_t
    )

    return return_pct, h_next_q


# ----------------------------------------------
# 4. 단일 경로 테스트: 10개 관측 구간
# ----------------------------------------------

rng = np.random.default_rng(42)

N_STEPS_TEST = 10
F0 = 10_000_000

F_path = np.empty(N_STEPS_TEST + 1)
h_path = np.empty(N_STEPS_TEST + 1)

F_path[0] = F0
h_path[0] = h0_Q

for t in range(N_STEPS_TEST):

    z_q = rng.standard_normal()

    return_pct, next_h = duan_q_step(
        h_t=h_path[t],
        z_q=z_q,
        params=params_hat,
        rf_pct=rf_daily_Q
    )

    # 퍼센트 로그수익률 → 소수 로그수익률
    F_path[t + 1] = (
        F_path[t] * np.exp(return_pct / 100)
    )

    h_path[t + 1] = next_h

# ----------------------------------------------
# 5. 결과 확인
# ----------------------------------------------

q_test_df = pd.DataFrame({
    "step": np.arange(N_STEPS_TEST + 1),
    "fund_value": F_path,
    "variance_pct2": h_path,
    "annual_vol_pct": np.sqrt(h_path * DAYS)
})

display(q_test_df.round(4))

assert np.all(np.isfinite(F_path))
assert np.all(np.isfinite(h_path))
assert np.all(F_path > 0)
assert np.all(h_path > 0)

print("\nQ 측도 단일 경로 테스트 완료")

# ===============================================
# Cell 7-B. Radon-Nikodym Density Validation
# ===============================================

import numpy as np
import pandas as pd
from IPython.display import display

rng = np.random.default_rng(123)
N = 300_000

omega, alpha, beta, lam = params_hat

# 1. P측도에서 표준정규 충격 생성
z_p = rng.standard_normal(N)

# 2. P -> Q 확률가중치
L = np.exp(
    -lam * z_p - 0.5 * lam**2
)

# 3. 같은 확률변수를 Q측도에서 해석
z_q = z_p + lam

# 4. Q측도 기대값 = P측도 가중기대값
E_L = np.mean(L)
E_Q_z = np.mean(L * z_q)
E_Q_z2 = np.mean(L * z_q**2)

# Q측도에서의 분산
Var_Q_z = E_Q_z2 - E_Q_z**2

result = pd.DataFrame({
    "검증 항목": [
        "E_P[L]",
        "E_Q[Z_Q]",
        "Var_Q(Z_Q)",
        "L > 0"
    ],
    "이론값": [
        1.0,
        0.0,
        1.0,
        True
    ],
    "MC 결과": [
        E_L,
        E_Q_z,
        Var_Q_z,
        bool(np.all(L > 0))
    ]
})

display(result)

print(f"가중치 평균 오차: {E_L - 1:.6f}")
print(f"Q 충격 평균 오차: {E_Q_z:.6f}")
print(f"Q 충격 분산 오차: {Var_Q_z - 1:.6f}")


# 설명을 추가하자면
# L은 가중치이기 때문에 기댓값이 1에 가깝게 나와야 함
# 그리고 Zq는 정규분포(0,1) 이기에 평균은 0에 가까워야 함
# 마찬가지로 Zq 분산도

# ==================================================
# Cell 7-C. LRNVR Conditional Variance Validation
# ==================================================

import numpy as np
import pandas as pd
from IPython.display import display

# 1. 기본 설정
rng = np.random.default_rng(123)
N = 300_000

DAYS = 252
r_daily = 0.03 / DAYS

omega, alpha, beta, lam = params_hat

# GARCH 조건부분산: %² -> 소수 단위
h = float(h_next) / 10000

# 2. P측도 충격 생성
z_p = rng.standard_normal(N)

# P측도 로그수익률
R_p = (
    r_daily
    - 0.5 * h
    + lam * np.sqrt(h)
    + np.sqrt(h) * z_p
)

# 3. Radon-Nikodym 밀도
L = np.exp(
    -lam * z_p - 0.5 * lam**2
)

# 4. 같은 수익률을 Q측도에서 재가중해 평가
# 유한 표본에서는 가중치를 정규화하여 계산
w = L / L.sum()

mean_q_weighted = np.sum(w * R_p)

var_q_weighted = np.sum(
    w * (R_p - mean_q_weighted)**2
)

# 5. Q측도에서 직접 시뮬레이션
z_q = rng.standard_normal(N)

R_q = (
    r_daily
    - 0.5 * h
    + np.sqrt(h) * z_q
)

# 6. 각 측도 조건부분산
var_p = np.var(R_p, ddof=1)
var_q_direct = np.var(R_q, ddof=1)

# 이론적 조건부 평균
mean_p_theory = r_daily - 0.5*h + lam*np.sqrt(h)
mean_q_theory = r_daily - 0.5*h

# 7. 결과표
results = pd.DataFrame({
    "Measure": [
        "P direct",
        "Q weighted",
        "Q direct",
        "Theoretical"
    ],
    "Mean": [
        R_p.mean(),
        mean_q_weighted,
        R_q.mean(),
        mean_q_theory
    ],
    "Variance": [
        var_p,
        var_q_weighted,
        var_q_direct,
        h
    ]
})

results["Volatility (%)"] = (
    np.sqrt(results["Variance"]) * 100
)

print("===== LRNVR Conditional Variance =====")
display(results.round(8))

print("\n===== Conditional Mean =====")
print(f"P theoretical mean: {mean_p_theory:.8f}")
print(f"Q theoretical mean: {mean_q_theory:.8f}")

print("\n===== Relative Variance Error =====")
print(f"P direct  : {(var_p/h - 1)*100:.4f}%")
print(f"Q weighted: {(var_q_weighted/h - 1)*100:.4f}%")
print(f"Q direct  : {(var_q_direct/h - 1)*100:.4f}%")

print("\n===== Weight Diagnostics =====")
print(f"E_P[L]: {L.mean():.6f}")
print(f"Effective sample size: {1 / np.sum(w**2):,.0f}")

# Q측도에서 직접 난수 생성 -> Q direct
# P측도에서 먼저 구하고 Q측도로 변환 -> Q weighted

# ===============================================
# Cell 7-A. Duan LRNVR Conditional Validation
# ===============================================

import numpy as np
import pandas as pd
from IPython.display import display

rng = np.random.default_rng(123)

N = 200_000
r_daily = 0.03 / 252

omega, alpha, beta, lam = params_hat

# 학습 종료 시점 조건부분산 (%^2)
h0 = float(h_next)

# 서로 다른 초기 변동성 수준에서 검증
h_levels = [
    0.5 * h0,
    h0,
    2.0 * h0
]

results = []

for H in h_levels:

    # P, Q 충격
    z_p = rng.standard_normal(N)
    z_q = rng.standard_normal(N)

    # H는 %^2이므로 소수 단위로 변환
    h = H / 10000

    # P측도 로그수익률
    R_p = (
        r_daily
        - 0.5 * h
        + lam * np.sqrt(h)
        + np.sqrt(h) * z_p
    )

    # Q측도 로그수익률
    R_q = (
        r_daily
        - 0.5 * h
        + np.sqrt(h) * z_q
    )

    # 한 기간 조건부분산 비교
    var_p = np.var(R_p, ddof=1)
    var_q = np.var(R_q, ddof=1)

    # Q측도 펀드 성장률
    gross_return_q = np.exp(R_q)

    # 할인된 성장률
    discounted_growth = (
        np.exp(-r_daily) * gross_return_q
    )

    mc_mean = discounted_growth.mean()
    mc_se = discounted_growth.std(ddof=1) / np.sqrt(N)

    results.append({
        "H (%^2)": H,
        "Theoretical Var": h,
        "P Variance": var_p,
        "Q Variance": var_q,
        "Q Discounted Mean": mc_mean,
        "MC SE": mc_se,
        "Z-score": (mc_mean - 1.0) / mc_se
    })

validation = pd.DataFrame(results)

display(validation.round(8))

# ==================================================
# Cell 7-1. Risk-Neutral Martingale Validation
# ==================================================

import numpy as np
import pandas as pd
from IPython.display import display

# ------------------------------------------
# 1. 기본 설정
# ------------------------------------------

F0 = 10_000_000 # 현재 펀드 가치
R_ANNUAL = 0.03 # 연속복리 무위험 금리 3%
DAYS = 252 # 연간 시뮬레이션 기간 수

# 우선 1년, 3년, 10년 검증
CHECK_YEARS = [1, 3, 10]

N_PATHS = 20_000 # 몬테카를로 경로 수
rng = np.random.default_rng(42) # 난수 생성기

omega, alpha, beta, lam = params_hat

rf_daily_pct = 100 * R_ANNUAL / DAYS

# ------------------------------------------
# 2. 이론적 1단계 마팅게일 확인
# ------------------------------------------

# 퍼센트 제곱 단위의 조건부분산
h0 = float(h_next)

# F_t+1 = F_t * e^R
# mgf 성질
# E[exp(root(h)*z)] = exp(h/2)


# 로그정규 충격의 기댓값
# E[exp(sqrt(h)/100 * Z)]
normal_adjustment = np.exp(h0 / 20000)

# 위험중립 1기간 기대 성장률
expected_growth = (
    np.exp(rf_daily_pct / 100 - h0 / 20000)
    * normal_adjustment
)

target_growth = np.exp(R_ANNUAL / DAYS)

print("===== One-Step Martingale Check =====")
print(f"Expected growth : {expected_growth:.12f}")
print(f"Risk-free growth: {target_growth:.12f}")

# 두 값이 수치적으로 거의 같은지 확인
assert np.isclose(
    expected_growth,
    target_growth,
    rtol=1e-10
)

print("1단계 조건부 기대값 검증 통과")

# ------------------------------------------
# 3. 다기간 위험중립 경로 생성
# ------------------------------------------

max_steps = max(CHECK_YEARS) * DAYS

F = np.full(N_PATHS, float(F0)) # 길이가 20,000인 배열을 만들고 모든 원소를 1,000만원으로 초기화
h = np.full(N_PATHS, h0) # 모든 경로의 최초 조건부분산을 동일한 값으로

results = []

for t in range(1, max_steps + 1):

    z_q = rng.standard_normal(N_PATHS) # z값 정규분포에서 랜덤하게 생성

    return_pct, h_new = duan_q_step( # 이전에 작성한 함수로 두 값을 동시에 계산
        h_t=h,
        z_q=z_q,
        params=params_hat,
        rf_pct=rf_daily_pct
    )

    F = F * np.exp(return_pct / 100)
    h = h_new

    if (
        not np.all(np.isfinite(F)) # 모든 펀드 가치와 조건부 분산이 유한한지 확인

        or not np.all(np.isfinite(h))
    ):
        raise FloatingPointError(
            f"수치적 발산 발생: step={t}"
        )

    if t in [year * DAYS for year in CHECK_YEARS]: # [252,756,2520]

        T = t / DAYS

        discounted_F = np.exp(-R_ANNUAL * T) * F # 현가로 할인

        mc_mean = discounted_F.mean()

        mc_se = discounted_F.std(ddof=1) / np.sqrt(N_PATHS) # 표준오차

        difference = mc_mean - F0

        z_score = difference / mc_se

        results.append({
            "Years": T,
            "Discounted Mean": mc_mean,
            "Target F0": F0,
            "Difference": difference,
            "Relative Error (%)": difference / F0 * 100,
            "MC SE": mc_se,
            "Z-score": z_score,
            "Within 95% MC CI": abs(z_score) <= 1.96
        }) # 신뢰구간도 계산

# ------------------------------------------
# 4. 결과 정리
# ------------------------------------------

validation_df = pd.DataFrame(results)

print("\n===== Multi-Period Martingale Validation =====")

display(validation_df.round(4))

print("\n===== 해석 =====")

for _, row in validation_df.iterrows():

    if row["Within 95% MC CI"]:
        print(
            f"{int(row['Years'])}년: "
            "할인된 평균과 F0의 차이가 "
            "95% Monte Carlo 오차 범위 내에 있습니다."
        )
    else:
        print(
            f"{int(row['Years'])}년: "
            "할인된 평균과 F0의 차이가 "
            "95% Monte Carlo 오차 범위를 벗어났습니다."
        )

# ------------------------------------------
# 5. 위험중립 분산 지속성
# ------------------------------------------

rho_P = alpha + beta
rho_Q = alpha * (1 + lam**2) + beta

print("\n===== Variance Persistence =====")
print(f"P persistence: {rho_P:.6f}")
print(f"Q persistence: {rho_Q:.6f}")

if rho_Q >= 1:
    print(
        "주의: Q 측도에서 장기 평균 조건부분산의 "
        "안정성 조건을 충족하지 않습니다."
    )

# ====================================================
# Cell 8. Q-GARCH Simulation & Martingale Validation
# ====================================================

import numpy as np
import pandas as pd
from IPython.display import display

# 기본 계약 설정
F0 = 10_000_000
G = 10_000_000
R_ANNUAL = 0.03
DAYS = 252
T = 10

N_PATHS = 30_000
rng = np.random.default_rng(42)

rf_daily_pct = 100 * R_ANNUAL / DAYS
h0_q = float(h_next)   # %² 단위

# Q측도 시뮬레이션 시작
F = np.full(N_PATHS, float(F0))
h = np.full(N_PATHS, h0_q)

check_steps = {252: 1, 756: 3, 2520: 10}
martingale_results = []

for step in range(1, T * DAYS + 1):

    z_q = rng.standard_normal(N_PATHS)

    return_pct, h_new = duan_q_step(
        h, z_q, params_hat, rf_daily_pct
    )

    F *= np.exp(return_pct / 100)
    h = h_new

    if not (
        np.all(np.isfinite(F))
        and np.all(np.isfinite(h))
        and np.all(h > 0)
    ):
        raise ValueError(
            f"Simulation became invalid at step {step}"
        )

    if step in check_steps:

        year = check_steps[step]
        discounted_F = np.exp(-R_ANNUAL * year) * F

        mc_mean = discounted_F.mean()
        mc_se = discounted_F.std(ddof=1) / np.sqrt(N_PATHS)

        martingale_results.append({
            "Year": year,
            "Discounted Mean": mc_mean,
            "Target": F0,
            "MC SE": mc_se,
            "Z-score": (mc_mean - F0) / mc_se,
            "Within 95% CI": (
                abs(mc_mean - F0) <= 1.96 * mc_se
            )
        })

# 10년 만기 최종 경로 저장
F_T = F.copy()

martingale_df = pd.DataFrame(martingale_results)

print("===== Multi-period Martingale Validation =====")
display(martingale_df.round(4))

# ==============================================
# Cell 9. GMAB Pricing & Put-Call Parity
# ==============================================

discount = np.exp(-R_ANNUAL * T)

# 동일한 만기 펀드 경로로 콜과 풋 지급액 계산
put_payoff = np.maximum(G - F_T, 0.0)
call_payoff = np.maximum(F_T - G, 0.0)

# 각각의 현재가치
put_pv = discount * put_payoff
call_pv = discount * call_payoff

# Monte Carlo 옵션가격
put_price = put_pv.mean()
call_price = call_pv.mean()

put_se = put_pv.std(ddof=1) / np.sqrt(N_PATHS)

# 풋-콜 패리티
parity_lhs = call_price - put_price
parity_rhs = F0 - G * discount

# 같은 경로에서 계산한 패리티 잔차
parity_error = parity_lhs - parity_rhs

# 경로별 지급액 항등식 검증
payoff_identity_error = np.max(
    np.abs(call_payoff - put_payoff - (F_T - G))
)

print("===== GARCH-Duan GMAB Pricing =====")
print(f"GMAB (Put) Price: {put_price:,.2f} KRW")
print(f"Call Price      : {call_price:,.2f} KRW")
print(f"GMAB MC SE      : {put_se:,.2f} KRW")
print(
    f"GMAB 95% MC CI  : "
    f"[{put_price - 1.96*put_se:,.2f}, "
    f"{put_price + 1.96*put_se:,.2f}] KRW"
)

print("\n===== Put-Call Parity =====")
print(f"Call - Put      : {parity_lhs:,.2f}")
print(f"F0 - PV(G)      : {parity_rhs:,.2f}")
print(f"Parity Error    : {parity_error:,.2f}")
print(f"Payoff Identity Max Error: {payoff_identity_error:.8f}")

# 동일 경로의 할인 펀드 가치 평균
discounted_mean = discount * F_T.mean()
discounted_se = (
    discount * F_T.std(ddof=1) / np.sqrt(N_PATHS)
)

print("\n===== Parity Error Diagnostic =====")
print(f"Discounted Fund Mean: {discounted_mean:,.2f}")
print(f"Discounted Fund SE  : {discounted_se:,.2f}")
print(
    f"Parity Z-score      : "
    f"{parity_error / discounted_se:.4f}"
)

# ==============================================
# Cell 10. GMAB Monte Carlo Convergence
# ==============================================

path_counts = [1_000, 2_000, 5_000, 10_000, 20_000, 30_000]

convergence_results = []

for n in path_counts:

    sample = put_pv[:n]

    price = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)

    convergence_results.append({
        "N Paths": n,
        "GMAB Price": price,
        "MC SE": se,
        "95% CI Lower": price - 1.96 * se,
        "95% CI Upper": price + 1.96 * se
    })

convergence_df = pd.DataFrame(convergence_results)

print("===== Monte Carlo Convergence =====")
display(convergence_df.round(2))

# ==================================================
# Cell 11. GBM MC vs Black-Scholes Benchmark
# ==================================================

from scipy.stats import norm

# 학습기간 일별 로그수익률에서 연율 변동성 계산
sigma_annual = np.std(y_train, ddof=1) / 100 * np.sqrt(DAYS)

# Black-Scholes Put
d1 = (
    np.log(F0 / G)
    + (R_ANNUAL + 0.5 * sigma_annual**2) * T
) / (sigma_annual * np.sqrt(T))

d2 = d1 - sigma_annual * np.sqrt(T)

bs_put = (
    G * np.exp(-R_ANNUAL * T) * norm.cdf(-d2)
    - F0 * norm.cdf(-d1)
)

# GBM terminal Monte Carlo
rng_gbm = np.random.default_rng(2026)
N_GBM = 200_000

z_gbm = rng_gbm.standard_normal(N_GBM)

F_T_gbm = F0 * np.exp(
    (R_ANNUAL - 0.5 * sigma_annual**2) * T
    + sigma_annual * np.sqrt(T) * z_gbm
)

gbm_put_pv = (
    np.exp(-R_ANNUAL * T)
    * np.maximum(G - F_T_gbm, 0)
)

gbm_mc = gbm_put_pv.mean()
gbm_se = gbm_put_pv.std(ddof=1) / np.sqrt(N_GBM)

print("===== GBM Benchmark =====")
print(f"Annualized Sigma : {sigma_annual:.4%}")
print(f"Black-Scholes Put: {bs_put:,.2f} KRW")
print(f"GBM MC Put       : {gbm_mc:,.2f} KRW")
print(f"MC SE            : {gbm_se:,.2f} KRW")
print(f"Difference       : {gbm_mc - bs_put:,.2f} KRW")
print(
    f"Within 95% MC CI: "
    f"{abs(gbm_mc - bs_put) <= 1.96 * gbm_se}"
)

# ==============================================
# Cell 12. Guarantee Sensitivity Validation
# ==============================================

guarantees = np.array([
    8_000_000,
    9_000_000,
    10_000_000,
    11_000_000,
    12_000_000
])

prices = []

for guarantee in guarantees:
    payoff = np.maximum(guarantee - F_T, 0.0)
    prices.append(discount * payoff.mean())

sensitivity_df = pd.DataFrame({
    "Guarantee": guarantees,
    "GMAB Price": prices
})

display(sensitivity_df.round(2))

print(
    "Monotonicity Check:",
    np.all(np.diff(prices) >= -1e-8)
)

# 추가로 보증금액 변화에 대한 이론적 상한 확인
increments = np.diff(prices)
upper_bounds = discount * np.diff(guarantees)

print(
    "Price Increment Bound Check:",
    np.all(increments <= upper_bounds + 1e-8)
)


import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# GMAB 사망위험 반영
# 가입자: 60세 남성 / 만기: 10년 / 해약 없음
# ==========================================

AGE = 60
T = 10

# 1. KOSIS에서 추출한 사망률 데이터 불러오기
file = Path("male_60_69_mortality.csv")

if not file.exists():
    file = Path("/content/male_60_69_mortality.csv")

mortality = pd.read_csv(file, encoding="utf-8-sig")

# 2. 연령별 사망률 정리
mortality = (
    mortality.loc[
        mortality["age"].between(AGE, AGE + T - 1),
        ["age", "qx"]
    ]
    .sort_values("age")
    .reset_index(drop=True)
)

assert mortality["age"].tolist() == list(range(AGE, AGE + T))
assert mortality["qx"].between(0, 1).all()

# 3. 생존확률 계산
mortality["px"] = 1 - mortality["qx"]

survival_10y = float(mortality["px"].prod())

# 4. 기존 GARCH-Duan 프라이싱 결과에 반영
# gmab_price는 기존 프라이싱 셀에서 산출한 값
gmab_price = put_price  # Existing Cell 9 output; alias only, no pricing formula change.
gmab_price_survival = gmab_price * survival_10y

print(f"10년 생존확률: {survival_10y:.6%}")
print(f"기존 GMAB 가격: {gmab_price:,.2f}원")
print(f"생존확률 반영 가격: {gmab_price_survival:,.2f}원")

display(mortality)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# ==========================================================
# 1. 계약 및 시뮬레이션 설정
# ==========================================================
F0 = 10_000_000
G0 = 10_000_000
T = 10
r = 0.03
rollup_rate = 0.02

RATCHET_YEARS = [3, 6]
TRADING_DAYS = 252
N_PATHS = 30_000
SEED = 42

# 기존 생명표에서 계산한 60세 남성 10년 생존확률
survival_10y = 0.90920318

# 기존 GARCH-Duan 추정 결과를 사용
omega, alpha, beta, lam = np.asarray(
    params_hat, dtype=float
).reshape(-1)

initial_h = float(np.asarray(h_next).reshape(-1)[0])

assert 0 < initial_h
assert omega > 0
assert alpha >= 0 and beta >= 0
assert alpha + beta < 1
assert all(0 < yr < T for yr in RATCHET_YEARS)

# ==========================================================
# 2. 위험중립 GARCH-Duan 경로 생성
# ==========================================================
rng = np.random.default_rng(SEED)

F = np.full(N_PATHS, F0, dtype=float)
h = np.full(N_PATHS, initial_h, dtype=float)

G_ratchet = np.full(N_PATHS, G0, dtype=float)

# 갱신 전후 보증금액을 기록
ratchet_records = []

# 그래프용 일부 경로
N_SHOW = 30
fund_plot = np.zeros((T+1, N_SHOW))
ratchet_plot = np.zeros((T+1, N_SHOW))
rollup_plot = np.zeros(T+1)

fund_plot[0] = F[:N_SHOW]
ratchet_plot[0] = G_ratchet[:N_SHOW]
rollup_plot[0] = G0

r_daily_pct = 100 * r / TRADING_DAYS

# 현재 시점의 경로 상태에서 10년을 시뮬레이션
for step in range(1, T * TRADING_DAYS + 1):

    z = rng.standard_normal(N_PATHS)

    sqrt_h = np.sqrt(h)

    # Duan Q측도 로그수익률, % 단위
    log_return_pct = (
        r_daily_pct
        - h / 200
        + sqrt_h * z
    )

    F *= np.exp(log_return_pct / 100)

    # Q측도 조건부분산 업데이트
    h = (
        omega
        + alpha * h * (z - lam)**2
        + beta * h
    )
    h = np.maximum(h, 1e-12)

    # 매년 말
    if step % TRADING_DAYS == 0:

        year = step // TRADING_DAYS

        # Ratchet 갱신
        if year in RATCHET_YEARS:
            old_G = G_ratchet.copy()

            G_ratchet = np.maximum(
                G_ratchet, F
            )

            ratchet_records.append({
                "갱신연도": year,
                "상향조정 비율":
                    np.mean(G_ratchet > old_G),
                "갱신 후 평균 보증금액":
                    np.mean(G_ratchet),
                "갱신 후 중앙 보증금액":
                    np.median(G_ratchet)
            })

        fund_plot[year] = F[:N_SHOW]
        ratchet_plot[year] = G_ratchet[:N_SHOW]

        rollup_plot[year] = (
            G0 * (1 + rollup_rate)**year
        )

F_T = F.copy()
G_ratchet_T = G_ratchet.copy()

# ==========================================================
# 3. 세 가지 보증구조의 만기 지급액
# ==========================================================

# 원금보증형
G_fixed_T = G0
payoff_fixed = np.maximum(G_fixed_T - F_T, 0)

# Roll-up형
G_rollup_T = G0 * (1 + rollup_rate)**T
payoff_rollup = np.maximum(G_rollup_T - F_T, 0)

# Ratchet형
payoff_ratchet = np.maximum(G_ratchet_T - F_T, 0)

# ==========================================================
# 4. 보증발동 확률과 가격 계산
# ==========================================================
discount = np.exp(-r * T)

def evaluate_gmab(label, payoff, guarantee):
    # 사망위험을 제외한 Q측도 보증발동확률
    prob = np.mean(payoff > 0)

    # 위험중립 평가
    discounted = discount * payoff

    price = discounted.mean()

    # 만기 생존확률 반영
    survival_price = survival_10y * price

    # MC 표준오차 (생존확률은 결정론적 가중치)
    se = (
        survival_10y
        * discounted.std(ddof=1)
        / np.sqrt(N_PATHS)
    )

    return {
        "보증구조": label,
        "평균 만기 보증금액": np.mean(guarantee),
        "만기 발동확률(Q)": prob,
        "생존 및 발동확률": survival_10y * prob,
        "생존조건 제외 가격": price,
        "생존조건 반영 가격": survival_price,
        "MC 표준오차": se,
    }

results = pd.DataFrame([
    evaluate_gmab(
        "원금보증",
        payoff_fixed,
        np.full(N_PATHS, G_fixed_T)
    ),
    evaluate_gmab(
        "Roll-up 2%",
        payoff_rollup,
        np.full(N_PATHS, G_rollup_T)
    ),
    evaluate_gmab(
        "Ratchet (3년, 6년)",
        payoff_ratchet,
        G_ratchet_T
    )
])

print("\n[GMAB 보증구조별 비교]")
display(results)

print("\n[Ratchet 갱신 결과]")
display(pd.DataFrame(ratchet_records))

# ==========================================================
# 5. Q측도 마팅게일 검증
# ==========================================================
discounted_fund = discount * F_T

martingale_ratio = discounted_fund.mean() / F0
martingale_se = (
    discounted_fund.std(ddof=1) / np.sqrt(N_PATHS)
)

martingale_z = (
    discounted_fund.mean() - F0
) / martingale_se

print(f"\n할인 펀드 평균 / F0: {martingale_ratio:.6f}")
print(f"마팅게일 z-score: {martingale_z:.3f}")

# ==========================================================
# 6. Ratchet 보증금액 분포
# ==========================================================
print("\n[만기 Ratchet 보증금액 분포]")
display(pd.Series({
    "5% 분위수": np.quantile(G_ratchet_T, 0.05),
    "중앙값": np.median(G_ratchet_T),
    "평균": G_ratchet_T.mean(),
    "95% 분위수": np.quantile(G_ratchet_T, 0.95),
    "최댓값": G_ratchet_T.max(),
}).to_frame("원"))

# ==========================================================
# 7. 시각화: 펀드 경로 vs Ratchet 보증금액
# ==========================================================
years = np.arange(T+1)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 첫 번째: 시뮬레이션된 펀드 가치
for j in range(N_SHOW):
    axes[0].plot(
        years,
        fund_plot[:, j] / 1e6,
        alpha=0.3,
        linewidth=0.9
    )

axes[0].plot(
    years,
    rollup_plot / 1e6,
    color="black",
    linewidth=2.5,
    label="Roll-up 2%"
)

axes[0].set_title("Simulated fund paths vs Roll-up")
axes[0].set_xlabel("Years")
axes[0].set_ylabel("KRW million")
axes[0].legend()

# 두 번째: 펀드 + 경로별 Ratchet 보증금액
for j in range(min(5, N_SHOW)):
    axes[1].plot(
        years,
        fund_plot[:, j] / 1e6,
        alpha=0.45
    )

    axes[1].step(
        years,
        ratchet_plot[:, j] / 1e6,
        where="post",
        linewidth=2
    )

for year in RATCHET_YEARS:
    axes[1].axvline(
        year, linestyle="--", alpha=0.5
    )

axes[1].set_title("Ratchet: fund and guarantee paths")
axes[1].set_xlabel("Years")
axes[1].set_ylabel("KRW million")

plt.tight_layout()
plt.show()
