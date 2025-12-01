import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import streamlit.components.v1 as components
import time

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# chatbot.py에서 필요한 함수들 import
from chatbot import define_workflow, process_input, get_drug_info
from langchain.schema import HumanMessage, SystemMessage

# JWT + 쿠키 기반 세션 관리 + Azure Blob 사용자 인증
import jwt
import time
import hashlib
from datetime import datetime, timedelta
from streamlit_cookies_manager import EncryptedCookieManager

# 쿠키 관리자 초기화
cookies = EncryptedCookieManager(
    prefix="streamlit_login_",
    password=os.getenv("COOKIE_PASSWORD", "default_password_change_in_production")
)

# JWT 설정
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
SESSION_DURATION = int(os.getenv("SESSION_DURATION", "86400"))  # 1시간

def hash_password(password):
    """비밀번호를 SHA256으로 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users_from_azure_blob():
    """Azure Blob에서 사용자 정보 로드"""
    try:
        from azure_blob_utils import create_azure_blob_vectorstore
        from chatbot import AZURE_BLOB_ENABLED
        
        if not AZURE_BLOB_ENABLED:
            st.error("❌ Azure Blob이 비활성화되어 있습니다.")
            return {}
        
        azure_vectorstore = create_azure_blob_vectorstore()
        if not azure_vectorstore:
            st.error("❌ Azure Blob 연결에 실패했습니다.")
            return {}
        
        # Azure Blob에서 users.xlsx 파일 다운로드
        df = azure_vectorstore.download_excel_file("user/users.xlsx")
        if df is None:
            st.error("❌ 사용자 파일을 찾을 수 없습니다.")
            return {}
        
        # 이메일을 키로, 사용자 정보를 값으로 하는 딕셔너리 생성
        users = {}
        for _, row in df.iterrows():
            email = str(row['email']).strip()
            password_hash = str(row['password_hash']).strip()
            manager = int(row['manager']) if 'manager' in row else 0
            users[email] = {
                'password_hash': password_hash,
                'manager': manager
            }
        
        return users
        
    except Exception as e:
        st.error(f"❌ 사용자 파일 로드 중 오류: {str(e)}")
        return {}

def create_jwt_token(email, manager_status=0):
    """JWT 토큰 생성"""
    payload = {
        'email': email,
        'manager': manager_status,
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
        return email, payload.get('manager', 0)
    except jwt.ExpiredSignatureError:
        return None, 0
    except jwt.InvalidTokenError:
        return None, 0

def check_login(email, password):
    """Azure Blob 기반 로그인 확인"""
    users = load_users_from_azure_blob()
    
    if not users:
        return False, "사용자 파일을 로드할 수 없습니다.", 0
    
    if email not in users:
        return False, "존재하지 않는 이메일입니다.", 0
    
    # 입력된 비밀번호를 해시화하여 비교
    input_password_hash = hash_password(password)
    stored_password_hash = users[email]['password_hash']
    manager_status = users[email]['manager']
    
    if input_password_hash == stored_password_hash:
        return True, "로그인 성공", manager_status
    else:
        return False, "비밀번호가 일치하지 않습니다.", 0

def login_user(email, password):
    """사용자 로그인 처리"""
    success, message, manager_status = check_login(email, password)
    if success:
        token = create_jwt_token(email, manager_status)
        cookies['auth_token'] = token
        cookies.save()
        return True, manager_status
    return False, 0

def logout_user():
    """사용자 로그아웃 처리"""
    try:
        # 1. 쿠키에서 auth_token 완전 삭제
        if 'auth_token' in cookies:
            cookies['auth_token'] = ""  # 빈 문자열로 설정
            del cookies['auth_token']    # 키 자체를 삭제
            cookies.save()
        
        # 2. 모든 세션 상태 초기화
        session_keys_to_clear = [
            'logged_in', 'username', 'is_manager', 
            'messages', 'conversation_messages', 'data_source',
            'feedback_given', 'app'
        ]
        
        for key in session_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # 3. 피드백 관련 세션 상태도 초기화
        keys_to_remove = [key for key in st.session_state.keys() 
                         if key.startswith(('show_feedback_', 'feedback_type_', 'feedback_text_'))]
        for key in keys_to_remove:
            del st.session_state[key]
            
        print("로그아웃 완료: 모든 세션 상태가 초기화되었습니다.")
        
    except Exception as e:
        print(f"로그아웃 중 오류 발생: {e}")
        # 오류가 발생해도 강제로 세션 상태 초기화
        st.session_state.clear()

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
            email, manager_status = verify_jwt_token(cookies['auth_token'])
            if email:
                st.session_state['logged_in'] = True
                st.session_state['username'] = email
                st.session_state['is_manager'] = manager_status
                return True
            else:
                # 토큰이 만료되었거나 유효하지 않음
                logout_user()
        except Exception as e:
            # 토큰 검증 중 오류 발생시 로그아웃
            print(f"토큰 검증 오류: {e}")
            logout_user()
    
    return False

def login_page():
    """로그인 페이지"""
    st.markdown("""
<span style='font-size:18px;'>HK이노엔 ETC Product AI</span><br>
<span style='font-size:32px; font-weight:bold;'>R:EDI (Ready+Detail Info) 로그인</span>
""", unsafe_allow_html=True)
    st.markdown("---")
    
    # 로그인 폼
    with st.form("login_form"):
        st.markdown("### 로그인 정보를 입력해주세요")
        email = st.text_input("이메일", placeholder="이메일을 입력하세요")
        password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("로그인", use_container_width=True)
        with col2:
            if st.form_submit_button("초기화", use_container_width=True):
                st.rerun()
        
        if submit_button:
            if email and password:
                success, manager_status = login_user(email, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = email
                    st.session_state.is_manager = manager_status
                    st.success("로그인 성공!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 이메일 또는 비밀번호가 올바르지 않습니다.")
            else:
                st.warning("⚠️ 이메일과 비밀번호를 모두 입력해주세요.")
    
    # 사용자 정보 안내
    # st.markdown("---")
    # st.markdown("### 📋 사용 가능한 계정")
    # st.markdown("""
    # **관리자 권한 (담당자 정보 업로드 가능):**
    # - **innon1@inno-n.com** / innon123! 🔧
    # - **woonha.jung@inno-n.com** / woonha123! 🔧
    # - **kmhadmin@kolmar.co.kr** / kmhadmin123!@# 🔧
    
    # **일반 사용자:**
    # - **innon2@inno-n.com** / innon456!
    # - **eunhee.lee6@inno-n.com** / pharm123!
    # - **sojeong.kim3000@inno-n.com** / sojeong123!
    # - **jin.kim@inno-n.com** / jinkim1234!
    # - **yj.so@inno-n.com** / yjso1234!
    # - **jihyeok.lim@inno-n.com** / jihyeoklim1234!
    # """)

# 사용자 정보는 Azure Blob의 users/users.xlsx 파일에서 로드됩니다

# 피드백 저장을 위한 디렉토리 생성
FEEDBACK_DIR = "feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# 페이지 설정
st.set_page_config(
    page_title="R:EDI (Ready+Detail Info)",
    page_icon="",
    layout="wide"
)

# 스타일 설정
st.markdown("""
<style>
.main {
    padding: 2rem;
}

/* 출처 및 피드백 */
.source-citation {
    margin-top: 1rem;
    padding-top: 0.5rem;
    border-top: 1px solid #ddd;
    font-size: 0.9em;
    color: #666;
}
.feedback-section {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #ddd;
    font-size: 0.9em;
}
.feedback-buttons {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.feedback-button {
    padding: 0.5rem 1rem;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    background-color: white;
    cursor: pointer;
    transition: background-color 0.2s;
}
.feedback-button:hover {
    background-color: #f0f0f0;
}
.feedback-input {
    margin-top: 0.5rem;
    display: none;
}
.feedback-input textarea {
    width: 100%;
    margin: 0.5rem 0;
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    resize: vertical;
}
.feedback-input button {
    padding: 0.5rem 1rem;
    background-color: #0066cc;
    color: white;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
}
.feedback-input button:hover {
    background-color: #0052a3;
}
    
    /* 답변 내용 스타일 */
    .answer-content {
        width: 100%;
        max-width: 100%;
        margin: 20px auto;
        line-height: 1.6;
        word-wrap: break-word;
        box-sizing: border-box;
        padding: 0 20px;
    }
    
    /* 청크 컨테이너 스타일 */
    .chunk-container {
        width: 100%;
        margin: 20px 0;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e1e4e8;
        box-sizing: border-box;
    }
    
    .chunk-title {
        font-weight: 600;
        color: #24292e;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    .chunk-source {
        color: #586069;
        font-size: 0.9em;
        margin-bottom: 15px;
    }
    
    .chunk-content {
        width: 100%;
        background-color: #ffffff;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #e1e4e8;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.5;
        box-sizing: border-box;
    }
    
    /* details 태그 스타일 */
    details {
        width: 100%;
        margin: 10px 0;
        box-sizing: border-box;
    }
    
    details summary {
        cursor: pointer;
        padding: 8px 12px;
        background-color: #f1f3f4;
        border-radius: 4px;
        color: #24292e;
        font-weight: 500;
        width: 100%;
        box-sizing: border-box;
    }
    
    details summary:hover {
        background-color: #e8eaed;
    }
    
    details[open] summary {
        margin-bottom: 10px;
    }
    
    details details {
        margin-left: 20px;
    }
    
    details details summary {
        background-color: #e8eaed;
    }
    
    details details summary:hover {
        background-color: #dde0e3;
    }
    
    /* 테이블 스타일 */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        overflow-wrap: break-word;
        box-sizing: border-box;
    }
    
    th, td {
        padding: 8px;
        border: 1px solid #e1e4e8;
        word-wrap: break-word;
    }
    
    th {
        background-color: #f6f8fa;
        font-weight: 600;
    }
    
    /* 반응형 디자인 */
    @media screen and (max-width: 768px) {
        .answer-content {
            padding: 0 10px;
        }
        .chunk-container {
            padding: 10px;
        }
        .chunk-content {
            padding: 10px;
        }
        table {
            display: block;
            overflow-x: auto;
        }
    }
    
    .main .block-container {
        padding-bottom: 8rem;
    }
    
    /* 스트리밍 애니메이션 */
    .streaming-text {
        animation: typing 0.1s steps(1, end);
    }
    
    @keyframes typing {
        from { opacity: 0.7; }
        to { opacity: 1; }
    }
</style>

<script>
// 피드백 입력창 표시
function showFeedbackInput(idx, type) {
    document.getElementById('feedback_input_' + idx).style.display = 'block';
    window.feedback_type = type;
}

// 피드백 제출
function submitFeedback(idx) {
    const reason = document.getElementById('feedback_reason_' + idx).value;
    const is_satisfied = window.feedback_type === 'satisfied';

    fetch('/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            answer_id: idx,
            is_satisfied: is_satisfied,
            reason: reason
        })
    }).then(() => {
        document.getElementById('feedback_input_' + idx).style.display = 'none';
        window.location.reload();
    });
}

// 스크롤을 가장 아래로 이동
window.addEventListener("load", function() {
    const anchor = document.getElementById("bottom-anchor");
    if (anchor) {
        anchor.scrollIntoView({ behavior: "smooth" });
    }
});

// 새 메시지가 추가될 때 자동 스크롤
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

// DOM 변경 감지
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            setTimeout(scrollToBottom, 100);
        }
    });
});

// 페이지 로드 시 관찰 시작
document.addEventListener('DOMContentLoaded', function() {
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // details 태그 이벤트 리스너 추가
    document.addEventListener('click', function(e) {
        if (e.target.tagName === 'SUMMARY') {
            e.preventDefault();
            const details = e.target.parentElement;
            if (details.hasAttribute('open')) {
                details.removeAttribute('open');
            } else {
                details.setAttribute('open', '');
            }
        }
    });
});
</script>
""", unsafe_allow_html=True)



def initialize_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_messages' not in st.session_state:
        st.session_state.conversation_messages = []
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "내부자료"

    if 'app' not in st.session_state:
        try:
            st.session_state.app = define_workflow()
        except Exception as e:
            st.error("❌ 챗봇 워크플로우 초기화 중 오류가 발생했습니다.")
            st.error(str(e))
            st.stop()
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()  # 피드백을 이미 주었는지 추적


def reset_conversation():
    """대화 기록 초기화"""
    st.session_state.messages = []
    st.session_state.conversation_messages = []
    st.session_state.feedback_given = set()  # 피드백 초기화


def save_feedback(answer_id, is_satisfied, reason):
    """피드백을 Azure Blob 또는 로컬 파일로 저장"""
    # 해당 answer_id에 해당하는 대화 찾기
    question = ""
    answer = ""
    
    # 대화 기록에서 질문과 답변 찾기
    for i, msg in enumerate(st.session_state.messages):
        if i == answer_id:
            # 답변 저장
            answer = msg["content"]
        elif i == answer_id - 1:
            # 질문 저장
            question = msg["content"]

    feedback = {
        "timestamp": datetime.now().isoformat(),
        "answer_id": answer_id,
        "question": question,
        "answer": answer,
        "is_satisfied": is_satisfied,
        "reason": reason
    }
    
    # Azure Blob이 활성화되어 있는지 확인
    try:
        from azure_blob_utils import create_azure_blob_vectorstore, get_azure_blob_config
        from chatbot import AZURE_BLOB_ENABLED, azure_vectorstore
        
        if AZURE_BLOB_ENABLED and azure_vectorstore:
            # Azure Blob에 저장
            blob_path = f"feedback/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if azure_vectorstore.upload_json_data(feedback, blob_path):
                st.success("피드백이 Azure Blob에 저장되었습니다.")
                return True
            else:
                st.warning("⚠️ Azure Blob 저장 실패, 로컬 파일에 저장합니다.")
        
    except Exception as e:
        st.warning(f"⚠️ Azure Blob 저장 중 오류: {e}")
    
    # 로컬 파일에 저장 (폴백)
    filename = os.path.join(FEEDBACK_DIR, f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    st.success("피드백이 로컬 파일에 저장되었습니다.")
    return True


def display_chat_history():
    """채팅 히스토리 표시"""
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                # HTML 내용을 안전하게 렌더링
                st.markdown(message["content"], unsafe_allow_html=True)
                
                # 봇 메시지인 경우에만 피드백 컴포넌트 추가
                if idx not in st.session_state.feedback_given:
                    # 피드백 상태 초기화
                    if f"show_feedback_{idx}" not in st.session_state:
                        st.session_state[f"show_feedback_{idx}"] = False
                    if f"feedback_type_{idx}" not in st.session_state:
                        st.session_state[f"feedback_type_{idx}"] = None
                        
                    # 피드백 버튼 컨테이너
                    col1, col2 = st.columns([1, 1])
                    
                    # 만족/불만족 버튼
                    with col1:
                        if st.button("👍 만족", key=f"satisfied_{idx}"):
                            st.session_state[f"show_feedback_{idx}"] = True
                            st.session_state[f"feedback_type_{idx}"] = "satisfied"
                            st.rerun()
                            
                    with col2:
                        if st.button("👎 불만족", key=f"unsatisfied_{idx}"):
                            st.session_state[f"show_feedback_{idx}"] = True
                            st.session_state[f"feedback_type_{idx}"] = "unsatisfied"
                            st.rerun()
                    
                    # 피드백 입력창 표시
                    if st.session_state[f"show_feedback_{idx}"]:
                        feedback_text = st.text_area("의견을 자유롭게 작성해주세요", key=f"feedback_text_{idx}")
                        if st.button("제출", key=f"submit_{idx}"):
                            save_feedback(idx, 
                                        st.session_state[f"feedback_type_{idx}"] == "satisfied",
                                        feedback_text)
                            st.session_state.feedback_given.add(idx)
                            st.rerun()
    
    # 자동 스크롤을 위한 앵커
    st.markdown('<div id="bottom-anchor"></div>', unsafe_allow_html=True)
    
    components.html("""
    <script>
    window.scrollTo(0, document.body.scrollHeight);
    </script>
    """, height=0)


def process_user_input_streaming(user_input):
    """사용자 입력을 스트리밍 방식으로 처리"""
    try:
        # "답변 생성 중" 메시지 표시
        yield ""
        
        # 챗봇 처리 - 선택된 자료 유형 전달 (RAG 디버그 메시지는 터미널에 출력)
        response = process_input(st.session_state.app, user_input, st.session_state.conversation_messages, st.session_state.data_source)
        
        # 참고 출처 섹션 찾기 (더 간단하고 확실한 방법)
        import re
        
        # 참고 출처 관련 키워드들
        reference_keywords = [
            '참고 출처', '참고문헌', '참고한 문서', '참고 문서', 
            '출처 정보', '관련 문서', '참고자료', '출처'
        ]
        
        main_content = response
        reference_content = ""
        
        # 각 키워드로 시도해서 참고 출처 섹션 찾기
        for keyword in reference_keywords:
            # 키워드 앞에 숫자나 마크다운이 있을 수 있음
            patterns = [
                f'\n4\.\s*{keyword}',
                f'\n\*\*4\.\s*{keyword}\*\*',
                f'\n###\s*4\.\s*{keyword}',
                f'\n##\s*4\.\s*{keyword}',
                f'\n#\s*4\.\s*{keyword}',
                f'\n{keyword}',
                f'\n\*\*{keyword}\*\*',
                f'\n###\s*{keyword}',
                f'\n##\s*{keyword}',
                f'\n#\s*{keyword}',
                f'<details>.*{keyword}',
                f'<summary>.*{keyword}'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    split_point = match.start()
                    main_content = response[:split_point].strip()
                    reference_content = response[split_point:]
                    break
            
            if reference_content:  # 매칭되면 중단
                break
        
        # 문자 단위로 스트리밍 (더 명확한 효과)
        current_response = ""
        chunk_size = 10  # 10글자씩 스트리밍
        
        for i in range(0, len(main_content), chunk_size):
            chunk = main_content[i:i+chunk_size]
            current_response += chunk
            yield current_response
            # 스트리밍 효과를 위한 지연
            time.sleep(0.05)
        
        # 참고 출처 부분은 스트리밍하지 않고 한 번에 표시
        if reference_content:
            yield current_response + reference_content
        
        # 최종 응답을 세션에 저장
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        yield error_msg


def process_user_input(user_input):
    """사용자 입력 처리"""
    try:
        # 사용자 메시지를 세션에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 챗봇 처리 - 선택된 자료 유형 전달
        response = process_input(st.session_state.app, user_input, st.session_state.conversation_messages, st.session_state.data_source)
        
        # 챗봇 응답을 세션에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        return response
    except Exception as e:
        st.error("❌ 챗봇 처리 중 오류가 발생했습니다.")
        st.error(str(e))
        error_msg = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return error_msg

def main():
    # 쿠키 로드
    if not cookies.ready():
        st.stop()
    
    # 로그인 상태 확인
    if not check_authentication():
        login_page()
        return
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 제목
    st.markdown("""
<span style='font-size:18px;'>HK이노엔 ETC Product AI</span><br>
<span style='font-size:32px; font-weight:bold;'>R:EDI (Ready+Detail Info)</span>
""", unsafe_allow_html=True)

    
    # Azure Blob 상태 표시
    try:
        from chatbot import AZURE_BLOB_ENABLED, azure_vectorstore
        from azure_blob_utils import get_azure_blob_config
        
        # 환경 변수 상태 확인
        connection_string, container_name = get_azure_blob_config()
        
        # if AZURE_BLOB_ENABLED and azure_vectorstore:
        #     st.success("✅ Azure Blob Storage 연동 활성화 - 최신 벡터스토어 사용 중")
            
        #     # Azure Blob에서 벡터스토어 존재 여부 확인
        #     if azure_vectorstore.check_blob_exists("insurance_docs"):
        #         st.success("✅ Azure Blob에 벡터스토어 파일이 존재합니다")
        #     else:
        #         st.warning("⚠️ Azure Blob에 벡터스토어 파일이 없습니다")
                
        # else:
        #     st.info("ℹ️ 로컬 벡터스토어 사용 중")
            
        #     # 환경 변수 상태 표시
        #     if not connection_string:
        #         st.error("❌ AZURE_STORAGE_CONNECTION_STRING 환경 변수가 설정되지 않았습니다")
        #     if not container_name:
        #         st.error("❌ AZURE_STORAGE_CONTAINER_NAME 환경 변수가 설정되지 않았습니다")
                
        #     if connection_string and container_name:
        #         st.warning("⚠️ Azure Blob 설정은 있지만 연결에 실패했습니다. 연결 문자열을 확인해주세요.")
                
    except Exception as e:
        st.warning("⚠️ Azure Blob 상태 확인 중 오류가 발생했습니다.")
        st.error(str(e))
        
        # 환경 변수 디버깅 정보
        import os
        st.markdown("### 🔍 Azure Blob 환경 변수 디버깅")
        st.code(f"""
AZURE_STORAGE_CONNECTION_STRING: {'설정됨' if os.getenv("AZURE_STORAGE_CONNECTION_STRING") else '설정되지 않음'}
AZURE_STORAGE_CONTAINER_NAME: {'설정됨' if os.getenv("AZURE_STORAGE_CONTAINER_NAME") else '설정되지 않음'}
        """)
    
    # 사이드바 - 로그인 정보 및 자료 선택
    with st.sidebar:
        # 로그인 정보 표시
        st.markdown("### 사용자 정보")
        st.success(f"안녕하세요, **{st.session_state.username}**님!")
        
        # 로그아웃 버튼
        if st.button("로그아웃", use_container_width=True):
            logout_user()
            st.success("로그아웃되었습니다.")
            time.sleep(1)  # 사용자가 메시지를 볼 수 있도록 잠시 대기
            st.rerun()
        
        st.markdown("---")
        st.header("자료 선택")
        

        
        # 자료 유형 선택 드롭다운
        data_source = st.selectbox(
            "검색할 자료 유형을 선택하세요",
            options=["내부자료", "외부자료포함"],
            index=0 if st.session_state.data_source == "내부자료" else 1,  # 현재 선택된 값에 따라 인덱스 설정
            help="내부자료: 회사 내부 문서만 검색\n외부자료포함: 내부자료 + 외부 공개 자료 검색"
        )
        
        # 선택된 자료 유형을 세션 상태에 저장
        if data_source != st.session_state.data_source:
            st.session_state.data_source = data_source
            st.rerun()  # 변경 시 페이지 새로고침
        
        # 선택된 자료 유형 표시
        # st.info(f"현재 선택: **{data_source}**")
        
        # 자료 유형별 설명
        if data_source == "내부자료":
            st.success("내부자료 모드: 회사 내부 문서만 검색합니다.")
            st.caption("• RAG 검색만 사용")
            st.caption("• 식약처 API 비활성화")
            st.caption("• 심평원 검색 비활성화")
            st.caption("• 뉴스 검색 비활성화")
        else:
            st.success("외부자료포함 모드: 모든 기능을 사용합니다.")
            st.caption("• RAG 검색 사용")
            st.caption("• 식약처 API 사용")
            st.caption("• 심평원 검색 사용")
            st.caption("• 뉴스 검색 사용")
        
        # 구분선
        st.markdown("---")
        
        # 담당자 정보 관리 섹션 (manager 권한이 있는 경우만 표시)
        if st.session_state.get('is_manager', 0) == 1:
            st.header("담당자 정보 관리")
            st.success("관리자 권한으로 담당자 정보를 관리할 수 있습니다.")
            
            # 파일 업로드 위젯
            uploaded_file = st.file_uploader(
                "담당자 정보 엑셀 파일 업로드",
                type=['xlsx', 'xls'],
                help="제품별 담당자 매핑 테이블을 업로드하세요. 기존 파일을 덮어씁니다.",
                key="manager_upload"
            )
            
            if uploaded_file is not None:
                # 파일 정보 표시
                st.info(f"선택된 파일: {uploaded_file.name}")
                st.info(f"파일 크기: {uploaded_file.size:,} bytes")
                
                # 업로드 버튼
                if st.button("🚀 Azure Blob에 업로드", key="upload_to_azure"):
                    with st.spinner("파일을 Azure Blob에 업로드하는 중..."):
                        # 임시 파일 경로를 미리 정의
                        temp_path = f"temp_{uploaded_file.name}"
                        
                        try:
                            # 임시 파일로 저장
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Azure Blob 연결 확인
                            from azure_blob_utils import create_azure_blob_vectorstore
                            from chatbot import AZURE_BLOB_ENABLED
                            
                            if not AZURE_BLOB_ENABLED:
                                st.error("❌ Azure Blob이 비활성화되어 있습니다.")
                                st.info("💡 Azure Blob을 활성화한 후 다시 시도해주세요.")
                            else:
                                azure_vectorstore = create_azure_blob_vectorstore()
                                
                                if azure_vectorstore:
                                    # 기존 manager 폴더의 모든 Excel 파일 삭제
                                    st.info("기존 담당자 정보 파일들을 삭제하는 중...")
                                    delete_success = azure_vectorstore.delete_excel_files_in_folder("manager/")
                                    
                                    if delete_success:
                                        st.success("기존 파일 삭제 완료")
                                    else:
                                        st.warning("⚠️ 기존 파일 삭제 중 일부 오류 발생")
                                    
                                    # Azure Blob에 업로드 (manager 폴더에 저장)
                                    blob_path = "manager/제품별_담당자_매핑_테이블.xlsx"
                                    
                                    if azure_vectorstore.upload_excel_file(temp_path, blob_path):
                                        st.success("담당자 정보가 Azure Blob에 성공적으로 업로드되었습니다!")
                                        st.info("새로운 담당자 정보가 다음 질문부터 반영됩니다.")
                                        st.info("챗봇을 재시작하면 즉시 반영됩니다.")
                                        
                                        # 업로드된 파일 내용 미리보기 (선택사항)
                                        try:
                                            import pandas as pd
                                            df = pd.read_excel(temp_path)
                                            st.success(f"📋 업로드된 데이터: {len(df)}행")
                                            if len(df) > 0:
                                                st.caption("컬럼: " + ", ".join(df.columns.tolist()))
                                        except Exception as e:
                                            st.warning(f"⚠️ 파일 미리보기 실패: {str(e)}")
                                            
                                    else:
                                        st.error("❌ Azure Blob 업로드에 실패했습니다.")
                                        st.info("💡 Azure Blob 연결 상태를 확인해주세요.")
                                else:
                                    st.error("❌ Azure Blob 연결에 실패했습니다.")
                                    st.info("💡 Azure Blob 설정을 확인해주세요.")
                            
                        except Exception as e:
                            st.error(f"❌ 업로드 중 오류가 발생했습니다: {str(e)}")
                            st.info("💡 파일 형식과 Azure Blob 설정을 확인해주세요.")
                        
                        finally:
                            # 임시 파일 삭제
                            try:
                                temp_file = Path(temp_path)
                                if temp_file.exists():
                                    temp_file.unlink()
                                    st.info("🗑️ 임시 파일이 정리되었습니다.")
                            except Exception as e:
                                st.warning(f"⚠️ 임시 파일 삭제 실패: {str(e)}")
            
            # 현재 담당자 정보 상태 표시
            try:
                from chatbot import AZURE_BLOB_ENABLED, azure_vectorstore
                
                if AZURE_BLOB_ENABLED and azure_vectorstore:
                    # Azure Blob에서 담당자 정보 파일 존재 여부 확인
                    if azure_vectorstore.check_blob_exists("manager/제품별_담당자_매핑_테이블.xlsx"):
                        # st.success("Azure Blob에 담당자 정보 파일이 존재합니다")
                        print("Azure Blob에 담당자 정보 파일이 존재합니다")
                    else:
                        st.warning("⚠️ Azure Blob에 담당자 정보 파일이 없습니다")
                else:
                    st.info("ℹ️ Azure Blob이 비활성화되어 있습니다")
                    
            except Exception as e:
                st.warning(f"⚠️ 담당자 정보 상태 확인 중 오류: {str(e)}")
        else:
            # manager 권한이 없는 경우 안내 메시지
            # st.info("담당자 정보 관리 기능은 관리자 권한이 필요합니다.")
            print("관리자권한없음")
        
        # 구분선
        # st.markdown("---")
    
    # 소개 - 새로운 카테고리 구조에 맞게 수정
    st.markdown("이 챗봇은 다음과 같은 정보를 제공합니다:")

    with st.expander("질문 가이드"):
        st.markdown("""
    - 기본 정보 (성분, 함량, 용법, 용량, 적응증, 약가) 예시) 케이캡 정보 알려줘, MET,SU,DPP4,TZD 사용시 보험 인정되는 약제는?
    - 핵심 특장점 및 영업 포인트 예시) 케이캡 핵심 특장점 알려줘
    - 허가 적응증 안내
    - 제형별 차이점과 특징
    - 경쟁제품과의 효능/효과 비교 예시) 케이캡과 펙수클루 비교해줘
    - 케이캡 경쟁제품 비교해줘
    - 안전성 데이터
    - 논문 기반 Q&A 예시) 케이캡 임상 문헌 알려줘, 케이캡 유지요법 유지율
    - 담당자 문의 예시) 케이캡 담당자 알려줘, 다파엔 담당자
    - 대화 내용이 길어지면 한번씩 대화 초기화를 하면 답변이 잘 나옵니다.
        """)

    # with st.expander("경쟁제품/시장 비교"):
    #     st.markdown("""
    # - 경쟁제품과의 효능/효과 비교 ex) 케이캡과 펙수클루 비교해줘
    # - 케이캡 경쟁제품 비교해줘
    #     """)

#     with st.expander("임상/논문 근거"):
#         st.markdown("""
#     - 안전성 데이터
#     - 논문 기반 Q&A ex) 케이캡 임상 문헌 알려줘, 케이캡 유지요법 유지율
#         """)

#     with st.expander("실제 활용/상담 전략"):
#         st.markdown("""
#     - 담당자 문의 및 전환 설득법 ex) 케이캡 담당자 알려줘, 다파엔 담당자
#         """)
#     with st.expander("담당자 정보 파일 양식"):
#         st.markdown("""
#     - 엑셀파일(xlsx파일)로 좌측 사이드바에 업로드
#     - 엑셀 파일 양식
                    
# | 5계층명 | 5계층 | 4계층명 | PM사원 | 코스트센터 |
# |---------|-------|---------|--------|-------------|
#         """)
    # 채팅 히스토리 표시
    display_chat_history()
    

    
    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        reset_conversation()
        st.rerun()
    
    # 채팅 입력창 (st.chat_input 사용)
    if user_input := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지를 세션에 추가 (이전 질문이 사라지지 않도록)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 사용자 메시지 표시 (스트리밍하지 않음)
        with st.chat_message("user"):
            st.write(user_input)
        
        # 어시스턴트 메시지 스트리밍 표시
        with st.chat_message("assistant"):
            # 스트리밍 방식으로 답변 생성 및 표시
            response_container = st.empty()
            full_response = ""
            
            # 크롤링 과정을 스피너로 표시하고 임시 메시지 컨테이너 생성
            temp_message_container = st.empty()
            
            with st.spinner("답변 생성 중입니다..."):
                for chunk in process_user_input_streaming(user_input):
                    full_response = chunk
                    # 항상 HTML 렌더링 허용 (기존 스타일 유지)
                    response_container.markdown(full_response, unsafe_allow_html=True)
            
            # 답변 완료 시 임시 메시지 지우기
            temp_message_container.empty()
            
            # 최종 응답을 세션에 저장 (이미 process_user_input_streaming에서 저장됨)
            # 피드백 버튼 표시를 위해 메시지 인덱스 확인
            message_idx = len(st.session_state.messages) - 1
            
            # 피드백 버튼 (기존 로직과 동일)
            if message_idx not in st.session_state.feedback_given:
                # 피드백 상태 초기화
                if f"show_feedback_{message_idx}" not in st.session_state:
                    st.session_state[f"show_feedback_{message_idx}"] = False
                if f"feedback_type_{message_idx}" not in st.session_state:
                    st.session_state[f"feedback_type_{message_idx}"] = None
                    
                # 피드백 버튼 컨테이너
                col1, col2 = st.columns([1, 1])
                
                # 만족/불만족 버튼
                with col1:
                    if st.button("👍 만족", key=f"satisfied_{message_idx}"):
                        st.session_state[f"show_feedback_{message_idx}"] = True
                        st.session_state[f"feedback_type_{message_idx}"] = "satisfied"
                        st.rerun()
                        
                with col2:
                    if st.button("👎 불만족", key=f"unsatisfied_{message_idx}"):
                        st.session_state[f"show_feedback_{message_idx}"] = True
                        st.session_state[f"feedback_type_{message_idx}"] = "unsatisfied"
                        st.rerun()
                
                # 피드백 입력창 표시
                if st.session_state[f"show_feedback_{message_idx}"]:
                    feedback_text = st.text_area("의견을 자유롭게 작성해주세요", key=f"feedback_text_{message_idx}")
                    if st.button("제출", key=f"submit_{message_idx}"):
                        save_feedback(message_idx, 
                                    st.session_state[f"feedback_type_{message_idx}"] == "satisfied",
                                    feedback_text)
                        st.session_state.feedback_given.add(message_idx)
                        st.rerun()
        
        # 자동 스크롤을 위한 앵커
        st.markdown('<div id="bottom-anchor"></div>', unsafe_allow_html=True)
        
        # 자동 스크롤 JavaScript
        components.html("""
        <script>
        window.scrollTo(0, document.body.scrollHeight);
        </script>
        """, height=0)
    
    
    # 푸터
    st.markdown("---")
    st.markdown("© 2025 R:EDI (Ready+Detail Info). 모든 정보는 내부용으로만 사용하세요.")

if __name__ == "__main__":
    main()
