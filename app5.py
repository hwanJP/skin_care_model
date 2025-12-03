import streamlit as st
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize
import os
import jwt
import hashlib
import time
from datetime import datetime, timedelta
from streamlit_cookies_manager import EncryptedCookieManager

DEBUG_MODE = False
EPSILON = 0.01  # 선택된 원료가 반드시 0이 아니게 하기 위한 최소값
PH_MIN = 4.0
PH_MAX = 9.0    # 범위를 조금 넉넉하게 잡음
WATER_MIN = 50  # D.I.WATER 최소함량
WATER_MAX = 100  # D.I.WATER 최대함량
DATASET_BOUNDS = '스킨케어1팀_원료 간략화 목록_251127.csv'
bounds_from_csv = pd.read_csv(DATASET_BOUNDS)

# ====== 1. 공통: 모델 & 데이터 로더 ======
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    model = obj["model"]
    X = obj["X"]
    y = obj["y"]
    df = obj.get("df", None)

    n_features = df[X].shape[1]

    # 독립변수 컬럼명 추론
    if isinstance(df, pd.DataFrame):
        if df.shape[1] >= n_features:
            feature_columns = list(df.columns[-n_features:])
        else:
            feature_columns = [f"feature_{i+1}" for i in range(n_features)]
    else:
        feature_columns = [f"feature_{i+1}" for i in range(n_features)]

    return model, X, y, df, feature_columns

# ====== pkl 경로 설정 (사용자 환경에 맞게 파일명 확인 필요) ======
MODEL_PATH_PH_NO_VISC = "linear_regression_간략화85all_205x54_pH측정법.pkl" # pH예측, all
MODEL_PATH_VISC_WITH_VISC = "linear_regression_간략화85점도있음_68x54_점도값1000.pkl" # 점도예측, 점도있음
MODEL_PATH_PH_WITH_VISC = "linear_regression_간략화85점도있음_68x54_pH측정법.pkl"# pH예측, 점도있음

MODEL_PATH_VISC_ALL = "linear_regression_간략화85all_205x54_점도값1000.pkl" # 점도예측, all
MODEL_PATH_PH_ALL = "linear_regression_간략화85점도있음_68x54_점도값1000.pkl" # pH예측, 점도있음

# ====== 모델 로드 (파일이 없으면 에러가 나므로 try-except 처리는 생략함) ======
try:
    model_ph0, X_ph0, y_ph0, df_ph0, feat_ph0 = load_model(MODEL_PATH_PH_NO_VISC)
    model_visc, X_visc, y_visc, df_visc, feat_visc = load_model(MODEL_PATH_VISC_WITH_VISC)
    model_phv, X_phv, y_phv, df_phv, feat_phv = load_model(MODEL_PATH_PH_WITH_VISC)
    model_all_visc, X_all_visc, y_all_visc, df_all_visc, feat_all_visc = load_model(MODEL_PATH_VISC_ALL)
    model_all_ph, X_all_ph, y_all_ph, df_all_ph, feat_all_ph = load_model(MODEL_PATH_PH_ALL)
except FileNotFoundError:
    st.error("모델 파일(.pkl)을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()


if DEBUG_MODE: 

    # st.write(f"df_ph0: {X_ph0}")
    # st.write(f"model_visc: {X_visc}")
    # st.write(f"model_phv: {X_phv}")
    # st.write(f"model_all_visc: {X_all_visc}")
    # st.write(f"model_all_ph: {X_all_ph}")

    st.write(f"bounds ")
    bounds_from_csv = pd.read_csv(DATASET_BOUNDS)
    st.dataframe(bounds_from_csv)

    bounds_by_dataset = df_all_visc[X_all_visc].describe().loc[['min', 'max'], :].T
    bounds_by_dataset['idx'] = bounds_by_dataset.index
    bounds_by_dataset2 = pd.merge(bounds_by_dataset, bounds_from_csv, left_on='idx', right_on='원료명')
    # bounds_by_dataset.loc['D.I-WATER','min'] = 50

    st.write("bounds_by_dataset")
    st.write(bounds_by_dataset2)
    # st.write(bounds_by_dataset.loc['D.I-WATER','min'])


def load_bounds(df, X=None):
    bounds_from_csv = pd.read_csv(DATASET_BOUNDS)
    
    if X is None:
        bounds_by_dataset = df.describe().loc[['min', 'max'], :].T
    else:
        bounds_by_dataset = df[X].describe().loc[['min', 'max'], :].T

    bounds_by_dataset['idx'] = bounds_by_dataset.index
    bounds_by_dataset_temp = pd.merge(bounds_by_dataset, bounds_from_csv, left_on='idx', right_on='원료명')
    bounds_by_dataset_final = bounds_by_dataset_temp[['min','최대적용함량(%)']]
    bounds_by_dataset_final.columns = ['min','max']
    bounds_by_dataset_final.index = bounds_by_dataset['idx'].values

    if DEBUG_MODE:
        st.write('bounds_by_dataset_final')
        st.dataframe(bounds_by_dataset_final)

    return bounds_by_dataset_final

# ====== 2. 핵심 예측 함수 (수정됨) ======

def optimize_recipe(
    model,
    X_train,
    target_value: float,
    total: float = 100.0,
    must_use_idx=None,
    x0=None
):
    """
    [수정됨] 목적함수를 상대오차((pred-target)/target)^2 로 변경하여
    점도처럼 값이 큰 타겟에서도 예측가 잘 되도록 개선함.
    """
    n_features = X_train.shape[1]

    # --- Helper Functions ---
    def build_bounds_from_data():
        # bounds_by_dataset = X_train.describe().loc[['min', 'max'], :].T.values
        # bounds_by_dataset = df_all_visc[X_all_visc].describe().loc[['min', 'max'], :].T
        bounds_by_dataset = load_bounds(df_all_visc, X_all_visc)

        # 제약조건 custom
        bounds_by_dataset.loc['D.I-WATER','min'] = WATER_MIN
        bounds_by_dataset.loc['D.I-WATER','max'] = WATER_MAX
        
        b = bounds_by_dataset.values.astype(float)
        b = np.nan_to_num(b, nan=0.0)
        return b

    def apply_must_use(bounds, must_use_idx):
        if must_use_idx is None or len(must_use_idx) == 0:
            return bounds
        b = bounds.copy()
        for idx in must_use_idx:
            lb, ub = b[idx]
            if ub <= 0:
                continue
            lb = max(lb, EPSILON)
            if lb >= ub:
                lb = max(EPSILON, ub * 0.5)
            b[idx] = [lb, ub]
        return b

    def make_feasible_x0(lb, ub, total, x0_init=None):
        lb = lb.astype(float)
        ub = ub.astype(float)
        if x0_init is not None:
            x = np.clip(x0_init, lb, ub)
        else:
            x = lb.copy()
        
        current_sum = x.sum()
        remaining = total - current_sum
        capacity = ub - lb
        cap_sum = capacity.sum()

        if cap_sum <= 0:
            if abs(remaining) < 1e-8: return x
            else: return None

        x = x + capacity * (remaining / cap_sum)
        x = np.clip(x, lb, ub)
        s = x.sum()
        if abs(s - total) > 1e-6 and s > 0:
            x = x * (total / s)
        return x

    def run_slsqp(bounds, x0_feasible):
        lb_arr = bounds[:, 0]
        ub_arr = bounds[:, 1]
        sum_lb = lb_arr.sum()
        sum_ub = ub_arr.sum()

        if not (sum_lb <= total + 1e-5 and sum_ub >= total - 1e-5):
            return None, None, None

        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - total})

        # [핵심 수정] Objective Function Normalization # ========================> 점도, pH 둘다 들어올때 다시 확인하기
        def objective(x):
            pred = model.predict([x])[0]
            # 타겟값이 0에 가까우면 일반 제곱오차, 아니면 상대오차 제곱 사용
            if abs(target_value) > 1e-6:
                return ((pred - target_value) / target_value) ** 2
            else:
                return (pred - target_value) ** 2

        try:
            result = minimize(
                objective,
                x0_feasible,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-9}
            )
        except Exception as e:
            return None, None, None

        if not result.success:
            # 점도의 경우 스케일 때문에 정밀도 문제로 False가 뜰 수 있지만
            # message가 'Positive directional derivative...' 등인 경우 결과는 유효할 수 있음
            # 하지만 안전하게 None 반환 혹은 디버그 모드 확인
            return None, None, result

        optimized_x = result.x
        predicted = model.predict([optimized_x])[0]
        return optimized_x, predicted, result

    # --- 1차 시도: 데이터 bounds ---
    bounds1 = build_bounds_from_data()
    bounds1 = apply_must_use(bounds1, must_use_idx)
    lb1, ub1 = bounds1[:, 0], bounds1[:, 1]

    if x0 is None:
        x0_feasible1 = make_feasible_x0(lb1, ub1, total)
    else:
        x0_feasible1 = make_feasible_x0(lb1, ub1, total, x0_init=x0)

    if x0_feasible1 is not None:
        opt_x, pred, res = run_slsqp(bounds1, x0_feasible1)
        if opt_x is not None:
            return opt_x, pred, res

    # --- 2차 시도: 완화된 bounds (0 ~ total) ---
    bounds2 = np.array([[0.0, total]] * n_features, dtype=float)
    bounds2 = apply_must_use(bounds2, must_use_idx)
    lb2, ub2 = bounds2[:, 0], bounds2[:, 1]

    if x0 is None:
        x0_init2 = np.full(n_features, total / n_features)
    else:
        x0_init2 = x0.copy()

    x0_feasible2 = make_feasible_x0(lb2, ub2, total, x0_init=x0_init2)
    if x0_feasible2 is None:
        return None, None, None

    opt_x2, pred2, res2 = run_slsqp(bounds2, x0_feasible2)
    return opt_x2, pred2, res2


def optimize_recipe_3(models, X_train, target_value, total: float = 100.0, must_use_idx=None):
    """
    [수정됨] Case 3: pH와 점도 동시 예측 함수
    각 타겟에 대해 정규화된 오차 합을 최소화함.
    """
    n_features = X_train.shape[1]
    tgt_arr = np.atleast_1d(target_value).astype(float)
    
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]

    # 초기값: 평균 함량
    x0 = np.clip(np.median(X_train, axis=0), 0.0, None)
    s = x0.sum()
    if s > 0: x0 = x0 * (total/s)

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - total})

    # bounds_by_dataset = X_train.describe().loc[['min', 'max'], :].T.values
    bounds_by_dataset = load_bounds(X_train) # 251201수정
    # st.write(bounds_by_dataset)
    bounds = bounds_by_dataset.astype(float)
    # st.write(must_use_idx)
    # st.write(bounds.iloc[1,:].values)
    
    # Must Use 적용 - 251201수정
    if must_use_idx is not None:
        for idx in must_use_idx:
            # lb, ub = bounds[idx]
            lb, ub = bounds.iloc[idx].values
            if ub <= 0: 
                ub = total
            lb = max(lb, EPSILON)
            if lb >= ub: 
                lb = max(EPSILON, ub * 0.5)
            # bounds[idx] = [lb, ub]
            # bounds.loc[idx, 'min'] = lb # 수정
            # bounds.loc[idx, 'max'] = ub # 수정
            bounds.iloc[idx, bounds.columns.get_loc("min")] = lb
            bounds.iloc[idx, bounds.columns.get_loc("max")] = ub

    # DataFrame -> list of (lb, ub) tuples
    bounds_for_scipy = [tuple(row) for row in bounds[["min", "max"]].to_numpy()]


    # [핵심 수정] 다중 타겟 정규화 Objective
    def objective(x):
        loss = 0.0
        for m, tgt in zip(model_list, tgt_arr):
            pred = m.predict([x])[0]
            # 스케일 정규화 (pH는 5, 점도는 5000일 때 동등하게 기여하도록)
            if abs(tgt) > 1e-6:
                loss += ((pred - tgt) / tgt) ** 2
            else:
                loss += (pred - tgt) ** 2
        return loss

    # 1차 시도
    try:
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds_for_scipy, constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9}
        )
    except Exception as e:
        st.write(e)
        return None, None, None

    # 1차 실패 시 Bounds 완화 후 재시도 로직 (간소화)
    if not result.success:
        bounds_relaxed = np.array([[0.0, total]] * n_features)
        if must_use_idx:
            for idx in must_use_idx:
                bounds_relaxed[idx][0] = EPSILON
        try:
            result = minimize(
                objective, x0, method="SLSQP", bounds=bounds_relaxed, constraints=cons
            )
        except Exception as e:
            st.write(e)
            pass

    if not result.success:
        return None, None, result

    optimized_x = result.x
    preds_final = np.array([m.predict([optimized_x])[0] for m in model_list])
    return optimized_x, preds_final, result


# ====== Helper: 랜덤 초기값 생성 ======
def random_x0_from_bounds(X_train, total: float = 100.0):
    desc = X_train.describe().loc[['min', 'max'], :].T.values.astype(float)
    lb, ub = desc[:, 0], desc[:, 1]
    n_features = X_train.shape[1]
    u = np.random.rand(n_features)
    x = lb + u * (ub - lb)
    s = x.sum()
    if s > 0: x = x * (total / s)
    else: x[:] = total / n_features
    x = np.clip(x, lb, ub)
    return x

# ====== Wrapper: 범위/반복 실행 ======
def optimize_recipe_range_v2(model, X_train, target_min: float, target_max=None, total=100.0, must_use_idx=None):
    n_runs = 5  # 실행 횟수
    results = []
    failures = []
    
    mid_target = target_min if target_max is None else round(0.5 * (target_min + target_max), 3)

    for attempt in range(n_runs):
        x0_rand = random_x0_from_bounds(X_train, total)
        recipe, pred, opt_result = optimize_recipe(
            model, X_train, target_value=mid_target, total=total,
            must_use_idx=must_use_idx, x0=x0_rand
        )

        if recipe is None:
            failures.append({"target": mid_target, "result": opt_result})
            continue

        # Loss 계산 시에도 정규화된 값으로 비교하는 것이 좋으나, 리포팅은 직관적으로
        loss = (pred - mid_target) ** 2
        results.append({
            "target": mid_target, "predicted": pred,
            "recipe": recipe, "result": opt_result, "loss": loss
        })

    results = sorted(results, key=lambda x: x["loss"])
    return results, failures

# ====== Wrapper: Case 3 (다중모델) ======
def optimize_recipe_range_v3(models, X_train, target_ph_min, target_ph_max, target_visc, total=100.0, must_use_idx=None):
    n_runs = 5
    results = []
    failures = []
    
    mid_ph = target_ph_min if target_ph_max is None else round(0.5 * (target_ph_min + target_ph_max), 2)
    targets = [mid_ph, target_visc]

    for attempt in range(n_runs):
        recipe, preds, opt_result = optimize_recipe_3(
            models, X_train, target_value=targets, total=total, must_use_idx=must_use_idx
        )

        if recipe is None:
            failures.append({"result": opt_result})
            continue
        
        # Loss: 정규화된 오차 합
        loss_val = ((preds[0] - mid_ph)/mid_ph)**2 + ((preds[1] - target_visc)/target_visc)**2
        
        results.append({
            "target_ph": mid_ph, "target_visc": target_visc,
            "pred_ph": preds[0], "pred_visc": preds[1],
            "recipe": recipe, "loss": loss_val
        })
    
    results = sorted(results, key=lambda x: x["loss"])
    return results, failures


def download_df_to_csv(df, target, name):
    # e.g. name target => pH7.25

    # 수정-251201: all 원료=0인 row 제외
    df_cols = list(df.columns)
    df['_zero'] = df[df_cols[1:]].sum(axis=1)
    cond_non_zero = (df['_zero'] != 0)
    df_over0 = df[cond_non_zero].iloc[:,:-1]

    # 수정-251201: 세로길이 계산해서 반영
    num_rows = len(df_over0)
    row_height = 35 # Approximate height of a row in pixels
    border_height = 3 # Approximate height for borders
    calculated_height = (num_rows + 1) * row_height + border_height

    st.dataframe(df_over0, height=calculated_height)

    csv = df_over0.to_csv(index=True).encode("utf-8-sig")
    fn_target_val = f"{name}{target}"
    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name=f"predicted_candidates_{fn_target_val}.csv",
        mime='text/csv'
    )



def update_checklist(index, key_check):
    """체크박스 상태 변경 시, st.session_state.all_checklist_state 업데이트하는 콜백 함수"""
    # 체크박스의 새로운 상태는 st.session_state[f"{key_check}_{index}"]에 저장되어 있음
    st.session_state.all_checklist_state[index] = st.session_state[f"{key_check}_{index}"]


def picked_from_df(df, key):
    '''df에서 사용자가 선택한 인덱스 리스트 반환 => [1, 36, 42]
    expander_search_checklist 내부에서만 쓰임.
    '''
    selected = st.data_editor(
        df,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "선택",
                default=False,
            )
        },
        disabled=['원료명','기능'],
        hide_index=True, # index번호 안보이게 처리
        key=key
    )

    seleced_res = list(selected['선택'].values)

    temp = []
    for x in zip(df.index, seleced_res):
        temp.append(x)
    temp_picked = [x[0] for x in temp if x[1] == True]

    return temp_picked


def expander_search_checklist(ingredients, expander_name, key_check, key_search):
    # 1. 모든 항목의 체크 상태를 저장할 전역 상태 초기화
    if "all_checklist_state" not in st.session_state:
        # ingredients 길이만큼 False로 초기화
        st.session_state.all_checklist_state = [False] * len(ingredients)

    # 기존 순서 기준 테이블 
    df_picked_final = pd.DataFrame(ingredients)

    # 전달받은 리스트 파일 테이블 - 구분 참조용
    df_ingredients_pick = pd.read_csv(DATASET_BOUNDS).iloc[:,:-1]
    df_picked_final_all = pd.merge(df_picked_final, df_ingredients_pick, left_on=0, right_on='원료명', how='left')
    df_picked_final_all['선택'] = False

    ingredients_tabname = ['점증제', '방부제', '보습제', '용제', 'pH조절제', '킬레이팅제']

    tab_i1, tab_i2, tab_i3, tab_i4, tab_i5, tab_i6= st.tabs(ingredients_tabname)

    with tab_i1:
        df_for_pick1 = df_picked_final_all[df_picked_final_all['기능']=='점증제'].loc[:, ['원료명','기능','선택']]
        temp_picked1 = picked_from_df(df_for_pick1, key=f'{key_check}_tab_i1111adsdadaw')

    with tab_i2:
        df_for_pick2 = df_picked_final_all[df_picked_final_all['기능']=='방부제'].loc[:, ['원료명','기능','선택']]
        temp_picked2 = picked_from_df(df_for_pick2, key=f'{key_check}_tab_i2222wqewadsqw')
    
    with tab_i3:
        df_for_pick3 = df_picked_final_all[df_picked_final_all['기능']=='보습제'].loc[:, ['원료명','기능','선택']]
        temp_picked3 = picked_from_df(df_for_pick3, key=f'{key_check}_tab_i3333ewqeaff')

    with tab_i4:
        df_for_pick4 = df_picked_final_all[df_picked_final_all['기능']=='용제'].loc[:, ['원료명','기능','선택']]
        temp_picked4 = picked_from_df(df_for_pick4, key=f'{key_check}_tab_i4444weadvfbfe')

    with tab_i5:
        df_for_pick5 = df_picked_final_all[df_picked_final_all['기능']=='pH조절제'].loc[:, ['원료명','기능','선택']]
        temp_picked5 = picked_from_df(df_for_pick5, key=f'{key_check}_tab_i5555qweadsa')
    
    with tab_i6:
        df_for_pick6 = df_picked_final_all[df_picked_final_all['기능']=='킬레이팅제'].loc[:, ['원료명','기능','선택']]
        temp_picked6 = picked_from_df(df_for_pick6, key=f'{key_check}_tab_i6666dwqewqea')

    must_use_idx = temp_picked1 + temp_picked2 + temp_picked3 + temp_picked4 + temp_picked5 + temp_picked6

    if DEBUG_MODE:
        st.write(must_use_idx)

    # 기존- 리스트의 체크박스 형식으로 선택
    # with st.expander(expander_name):
    #     # 검색창 입력 시 Streamlit 앱이 재실행됨
    #     query = st.text_input("검색", value="", key=key_search).upper()

    #     must_use_idx = []
    #     for i, name in enumerate(ingredients):
    #         # 2. 검색 조건에 맞지 않으면 체크박스를 표시하지 않고 다음 항목으로 이동
    #         if query and query not in name.upper():
    #             continue

    #         # 3. 체크박스 생성 및 상태 처리
    #         # - key: 각 항목 고유의 키 (i를 포함)
    #         # - value: 세션 상태에 저장된 현재 선택 상태를 불러옴
    #         # - on_change: 체크 상태 변경 시 콜백 함수(update_checklist) 호출
    #         is_checked = st.checkbox(
    #             label=name, 
    #             value=st.session_state.all_checklist_state[i], # 저장된 값 사용
    #             key=f"{key_check}_{i}", 
    #             on_change=update_checklist,
    #             args=(i, key_check) # 콜백 함수에 전달할 인자
    #         )

    #         # 4. 현재 표시된/필터링된 항목 중 체크된 항목의 인덱스 수집
    #         if is_checked:
    #             must_use_idx.append(i)

    #     # 5. 결과를 st.session_state에 저장 (선택 사항, 반환값으로도 사용 가능)
    #     # st.session_state.checklist = must_use_idx

        
    return must_use_idx




# ================================
# 3. JWT + 쿠키 기반 세션 관리 로그인 기능
# ================================
# 하드코딩된 로그인 정보
VALID_USERNAME = "kolmar123@kolmar.co.kr"
VALID_PASSWORD = "kolmar123"

# 쿠키 관리자 초기화
cookies = EncryptedCookieManager(
    prefix="streamlit_login_",
    password=os.getenv("COOKIE_PASSWORD", "default_password_change_in_production")
)

# JWT 설정
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
SESSION_DURATION = int(os.getenv("SESSION_DURATION", "86400"))  # 24시간 (초 단위)

def hash_password(password):
    """비밀번호를 SHA256으로 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_jwt_token(email):
    """JWT 토큰 생성"""
    payload = {
        'email': email,
        'exp': datetime.utcnow() + timedelta(seconds=SESSION_DURATION),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token):
    """JWT 토큰 검증"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        # 기존 토큰과의 호환성을 위해 'email' 또는 'username' 키 확인
        email = payload.get('email') or payload.get('username')
        return email
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def check_login(email, password):
    """로그인 확인 (하드코딩된 사용자 정보 사용)"""
    if email == VALID_USERNAME and password == VALID_PASSWORD:
        return True, "로그인 성공"
    else:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다."

def login_user(email, password):
    """사용자 로그인 처리"""
    success, message = check_login(email, password)
    if success:
        token = create_jwt_token(email)
        cookies['auth_token'] = token
        cookies.save()
        return True
    return False

def logout_user():
    """사용자 로그아웃 처리"""
    try:
        # 1. 쿠키에서 auth_token 완전 삭제
        if 'auth_token' in cookies:
            cookies['auth_token'] = ""  # 빈 문자열로 설정
            del cookies['auth_token']    # 키 자체를 삭제
            cookies.save()
        
        # 2. 세션 상태 초기화
        if 'logged_in' in st.session_state:
            del st.session_state['logged_in']
        if 'username' in st.session_state:
            del st.session_state['username']
            
    except Exception as e:
        print(f"로그아웃 중 오류 발생: {e}")
        # 오류가 발생해도 강제로 세션 상태 초기화
        if 'logged_in' in st.session_state:
            del st.session_state['logged_in']
        if 'username' in st.session_state:
            del st.session_state['username']

def check_authentication():
    """인증 상태 확인"""
    # 쿠키가 준비되지 않았으면 False 반환
    if not cookies.ready():
        return False
    
    # 세션 상태에서 인증 확인
    if st.session_state.get('logged_in', False):
        return True
    
    # 쿠키에서 토큰 확인
    if 'auth_token' in cookies and cookies['auth_token']:
        try:
            email = verify_jwt_token(cookies['auth_token'])
            if email:
                st.session_state['logged_in'] = True
                st.session_state['username'] = email
                return True
            else:
                # 토큰이 만료되었거나 유효하지 않음
                logout_user()
        except Exception as e:
            # 토큰 검증 중 오류 발생시 로그아웃
            print(f"토큰 검증 오류: {e}")
            logout_user()
    
    return False

def show_login_page():
    """로그인 페이지"""
    st.set_page_config(layout="centered", page_title="로그인")
    
    st.title("로그인")
    st.markdown("---")
    
    with st.form("login_form"):
        st.markdown("### 로그인 정보를 입력해주세요")
        email = st.text_input("이메일", placeholder="이메일을 입력하세요")
        password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
        
        submit_button = st.form_submit_button("로그인", use_container_width=True)
        
        if submit_button:
            if email and password:
                success = login_user(email, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = email
                    st.success("로그인 성공!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 이메일 또는 비밀번호가 올바르지 않습니다.")
            else:
                st.warning("⚠️ 이메일과 비밀번호를 모두 입력해주세요.")

# ================================
# 4. Streamlit UI (메인 앱)
# ================================
# 쿠키 로드
if not cookies.ready():
    st.stop()

# 로그인 상태 확인
if not check_authentication():
    show_login_page()
    st.stop()

# ================================
# 3. Streamlit UI
# ================================
# 페이지 설정
st.set_page_config(layout="wide", page_title="pH·점도 레시피 예측")

# 로그아웃 버튼 (사이드바 또는 상단)
col_logout, col_title = st.columns([1, 10])
with col_logout:
    if st.button("로그아웃"):
        logout_user()
        st.success("로그아웃되었습니다.")
        time.sleep(1)
        st.rerun()

with col_title:
    st.title("pH · 점도 기반 레시피 예측")
    st.caption("입력한 물성(pH, 점도)을 만족하는 최적의 원료 배합비(Total 100%)를 예측합니다.")

tab_zero, tab_c2, tab_c3 = st.tabs(["① pH 예측 (점도=0)", "② 점도 예측 (점도>0)", "③ pH + 점도 동시 예측"])


# --- Tab 1: pH Only ---
with tab_zero:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 목표 물성 입력")
        ph_min = st.number_input("pH Min (ph 4.0~7.2)", 4.0, 9.0, 6.0, 0.1)
        ph_max = st.number_input("pH Max (ph 4.0~7.2)", 4.0, 9.0, 6.5, 0.1)
        
        must_use_idx_ph0 = expander_search_checklist(feat_ph0, "필수 포함 원료 선택", "searh0", "t1")
        if DEBUG_MODE: st.write(f"must use idx:{must_use_idx_ph0}")

        if st.button("실행 (Case 1)", type="primary"):
            with col2:
                with st.spinner("Calculating..."):
                    results, fails = optimize_recipe_range_v2(
                        model_ph0, df_ph0[X_ph0], ph_min, ph_max, must_use_idx=must_use_idx_ph0
                    )
                    if not results:
                        st.error("예측 실패. 조건을 완화해보세요.")
                        if fails: st.caption(fails[0]['result'].message)
                    else:
                        best = results[0]
                        st.success(f"예측 결과 (예측 pH: {best['predicted']:.2f})")
                        
                        # 결과 표시
                        res_df = pd.DataFrame({"원료": feat_ph0})
                        for i, res in enumerate(results[:3]): # Top 3만 표시
                            res_df[f"Rank{i+1} ({res['predicted']:.2f})"] = res['recipe']
                        # res_df = res_df.style.format(subset=res_df.columns[1:], formatter="{:.2f}")
                        
                        res_df = res_df.round(2)
                        tn = f"{ph_min}~{ph_max}"                    
                        download_df_to_csv(res_df, tn, "pH")

# --- Tab 2: Viscosity Only ---
with tab_c2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 목표 물성 입력")
        visc_target = st.number_input("목표 점도 (cP, 250~25000)", 100.0, 50000.0, 5000.0, 100.0)

        # 예측엔 목표점도를 1000으로 나눈 값으로 예측
        visc_target_for_opt = visc_target / 1000
        
        # must_use_idx_visc = []
        # with st.expander("필수 포함 원료 선택 (점도 데이터)"):
        #     for i, name in enumerate(feat_visc):
        #         if st.checkbox(name, key=f"t2_{i}"): must_use_idx_visc.append(i)
        must_use_idx_visc = expander_search_checklist(feat_visc, "필수 포함 원료 선택", "searh1", "t2")

        if st.button("실행 (Case 2)", type="primary"):
            with col2:
                with st.spinner("Calculating..."):
                    # 모델: model_visc 사용
                    results, fails = optimize_recipe_range_v2(
                        model_visc, df_visc[X_visc], visc_target_for_opt, None, must_use_idx=must_use_idx_visc
                    )
                    if not results:
                        st.error("점도 예측 실패. 목표값이 데이터 범위를 벗어났을 수 있습니다.")
                        if fails: st.caption(fails[0]['result'].message)
                    else:
                        best = results[0]
                        st.success(f"예측 결과 (예측 점도: {round(best['predicted']*1000,0)})")
                        
                        res_df = pd.DataFrame({"원료": feat_visc})
                        for i, res in enumerate(results[:3]):
                            res_df[f"Rank{i+1} ({res['predicted']:.0f}k)"] = res['recipe']

                        res_df = res_df.round(2)
                        tn = f"{visc_target}"
                        
                        download_df_to_csv(res_df, tn, "visc")

# --- Tab 3: pH + Viscosity ---
with tab_c3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 목표 물성 입력")
        t3_ph_min = st.number_input("pH Min (ph 4.0~7.2)", 4.0, 9.0, 6.0, 0.1, key="t3_ph_min")
        t3_ph_max = st.number_input("pH Max (ph 4.0~7.2)", 4.0, 9.0, 6.5, 0.1, key="t3_ph_max")
        t3_visc = st.number_input("목표 점도 (cP, 250~25000)", 100.0, 50000.0, 3000.0, 100.0, key="t3_visc")
        # 예측엔 목표점도를 1000으로 나눈 값으로 예측
        t3_visc_for_opt = visc_target / 1000

        # must_use_idx_mix = []
        # # 여기서는 점도 데이터셋 기준으로 원료 리스트를 보여줌 (pH데이터셋과 원료 순서가 같다면)
        # with st.expander("필수 포함 원료 선택"):
        #     for i, name in enumerate(feat_phv): # model_phv의 feature 사용
        #         if st.checkbox(name, key=f"t3_{i}"): must_use_idx_mix.append(i)
        must_use_idx_mix = expander_search_checklist(feat_all_ph, "필수 포함 원료 선택", "searh2", "t3")
        

        if st.button("실행 (Case 3)", type="primary"):
            with col2:
                with st.spinner("Calculating..."):
                    # 모델 2개를 리스트로 전달: [pH모델, 점도모델]
                    # 데이터셋은 점도가 포함된 데이터셋(df_phv 혹은 df_visc)을 써야 함 (X 구조가 동일해야 함)
                    
                    # [주의] model_phv와 model_visc가 동일한 Feature 순서를 가진다고 가정합니다.
                    # 만약 다르다면 X 데이터를 맞춰주는 전처리가 추가로 필요합니다.
                    results, fails = optimize_recipe_range_v3(
                        [model_all_ph, model_all_visc], 
                        df_all_ph[X_all_ph], 
                        t3_ph_min, t3_ph_max, t3_visc_for_opt,
                        must_use_idx=must_use_idx_mix
                    )

                    if not results:
                        st.error("동시 예측 실패.")
                        if fails: st.caption(str(fails[0]))
                    else:
                        best = results[0]
                        st.success(f"예측 결과 (예측 pH: {best['pred_ph']:.2f}, 예측점도: {round(best['pred_visc']*1000,0)})")
                        
                        res_df = pd.DataFrame({"원료": feat_phv})
                        for i, res in enumerate(results[:3]):
                            header = f"R{i+1}(pH{res['pred_ph']:.1f}/Visc{res['pred_visc']:.0f}k)"
                            res_df[header] = res['recipe']

                        res_df = res_df.round(2)
                        tn = f"{t3_ph_min}~{t3_ph_min}&{visc_target}"
                        
                        download_df_to_csv(res_df, tn, "pH&visc")



                        