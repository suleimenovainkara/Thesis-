import html
import os
import random
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from PIL import Image


API_BASE_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Museum Recommender",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM STYLES
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8f5ef 0%, #f2ede3 100%);
    }
    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #2f241f;
        margin-bottom: 8px;
        line-height: 1.1;
    }
    .subtitle {
        font-size: 18px;
        color: #5d5048;
        margin-bottom: 26px;
    }
    .hero-box {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(120,100,85,0.15);
        border-radius: 28px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(60,40,20,0.08);
        margin-bottom: 24px;
    }
    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #322721;
        margin-top: 6px;
        margin-bottom: 8px;
    }
    .section-text {
        color: #5d5048;
        font-size: 16px;
        margin-bottom: 16px;
    }
    .art-card {
        background: rgba(255,255,255,0.88);
        border-radius: 22px;
        padding: 14px;
        border: 1px solid rgba(120,100,85,0.12);
        box-shadow: 0 6px 20px rgba(70,50,30,0.07);
        margin-bottom: 18px;
    }
    .meta-label {
        color: #7a6b60;
        font-size: 13px;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .meta-value {
        color: #2e241f;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .reason-box {
        background: rgba(239, 231, 219, 0.55);
        border-radius: 14px;
        padding: 10px 12px;
        margin-top: 10px;
        margin-bottom: 6px;
        color: #4b4038;
        font-size: 14px;
        border: 1px solid rgba(120,100,85,0.10);
    }
    .soft-divider {
        height: 1px;
        background: rgba(90,70,50,0.1);
        margin-top: 12px;
        margin-bottom: 18px;
    }
    .small-note {
        color: #7a6b60;
        font-size: 13px;
    }
    .welcome-box {
        background: rgba(255,255,255,0.72);
        border-radius: 18px;
        padding: 14px 16px;
        border: 1px solid rgba(120,100,85,0.12);
        margin-top: 10px;
        margin-bottom: 14px;
    }
    .feature-box {
        background: rgba(255,255,255,0.72);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(120,100,85,0.10);
        margin-bottom: 14px;
        min-height: 150px;
    }
    .feature-title {
        font-size: 18px;
        font-weight: 700;
        color: #2f241f;
        margin-bottom: 6px;
    }
    .mode-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #ede3d3;
        color: #3d302a;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #efe7db 0%, #e7ddcf 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #2f241f !important;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #2f241f;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# API HELPERS
# =========================================================

# ── Увеличен timeout: 120 сек для тяжёлых ML-запросов ──
DEFAULT_TIMEOUT = 120


def api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        params=params,
        timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def api_post(
    endpoint: str,
    json_data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=json_data,
        files=files,
        params=params,
        timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def parse_api_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            "❌ Не удалось подключиться к бэкенду. "
            "Убедись что FastAPI запущен: uvicorn main:app --reload"
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return (
            "⏱ Запрос превысил время ожидания. "
            "Попробуй ещё раз — возможно бэкенд загружает данные."
        )
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        if response is not None:
            try:
                data = response.json()
                if isinstance(data, dict) and "detail" in data:
                    return f"Ошибка {response.status_code}: {data['detail']}"
            except Exception:
                pass
            try:
                return f"Ошибка {response.status_code}: {response.text}"
            except Exception:
                return f"Ошибка {response.status_code}: бэкенд вернул ошибку."
    return f"Неожиданная ошибка: {str(exc)}"


def check_backend() -> bool:
    """Быстрая проверка что бэкенд живой — вызывай при старте страницы"""
    try:
        requests.get(f"{API_BASE_URL}/", timeout=3)
        return True
    except Exception:
        return False


# =========================================================
# SESSION STATE
# =========================================================
def init_session_state() -> None:
    defaults = {
        "current_page": "Home",
        #"page_selector": "Home",
        "user_id": None,
        "username": "",
        "is_authenticated": False,
        "onboarding_candidates": [],
        "onboarding_recommendations": [],
        "user_recommendations": [],
        "image_recommendations": [],
        "favorites": [],
        "backend_ok": None,   
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_recommendation_state() -> None:
    st.session_state["onboarding_candidates"] = []
    st.session_state["onboarding_recommendations"] = []
    st.session_state["user_recommendations"] = []
    st.session_state["image_recommendations"] = []
    st.session_state["favorites"] = []


def logout() -> None:
    st.session_state["user_id"] = None
    st.session_state["username"] = ""
    st.session_state["is_authenticated"] = False
    reset_recommendation_state()
    st.session_state["current_page"] = "Home"
    #st.session_state["page_selector"] = "Home"
    st.rerun()


init_session_state()


# =========================================================
# VALIDATION
# =========================================================
def validate_registration(username: str, password: str, confirm_password: str) -> Optional[str]:
    username = username.strip()
    if not username:
        return "Username cannot be empty."
    if len(username) < 3:
        return "Username must contain at least 3 characters."
    if len(username) > 50:
        return "Username is too long."
    if not password:
        return "Password cannot be empty."
    if len(password) < 6:
        return "Password must contain at least 6 characters."
    if password != confirm_password:
        return "Passwords do not match."
    return None


def validate_login(username: str, password: str) -> Optional[str]:
    username = username.strip()
    if not username:
        return "Please enter your username."
    if not password:
        return "Please enter your password."
    return None


# =========================================================
# UI HELPERS
# =========================================================
def set_page(page_name: str) -> None:
    st.session_state["current_page"] = page_name
    #st.session_state["page_selector"] = page_name
    #st.rerun()


def safe_text(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return html.escape(text) if text else default


def load_favorites() -> None:
    if not st.session_state["is_authenticated"] or st.session_state["user_id"] is None:
        st.session_state["favorites"] = []
        return
    try:
        result = api_get(f"/favorites/{st.session_state['user_id']}")
        st.session_state["favorites"] = result.get("favorites", [])
    except Exception:
        st.session_state["favorites"] = []


def add_to_favorites_ui(artwork_id: int) -> None:
    try:
        api_post(
            "/favorites/add",
            json_data={"user_id": st.session_state["user_id"], "artwork_id": artwork_id}
        )
        load_favorites()
        st.success("Artwork saved to favorites.")
    except Exception as e:
        st.error(parse_api_error(e))


def remove_from_favorites_ui(artwork_id: int) -> None:
    try:
        api_post(
            "/favorites/remove",
            json_data={"user_id": st.session_state["user_id"], "artwork_id": artwork_id}
        )
        load_favorites()
        st.success("Artwork removed from favorites.")
    except Exception as e:
        st.error(parse_api_error(e))


def get_favorite_artwork_ids() -> set:
    return {
        item.get("id")
        for item in st.session_state.get("favorites", [])
        if item.get("id") is not None
    }


def render_reason(item: Dict[str, Any]) -> None:
    reason = safe_text(
        item.get("reason", "Recommended based on visual similarity and your preference profile.")
    )
    st.markdown(
        f'<div class="reason-box"><b>Why recommended:</b> {reason}</div>',
        unsafe_allow_html=True
    )


def render_art_card(
    item: Dict[str, Any],
    show_similarity: bool = False,
    selectable: bool = False,
    checkbox_key: Optional[str] = None,
    allow_favorite_actions: bool = False,
    allow_remove_favorite: bool = False
):
    image_path = item.get("image_path", "")
    artist     = safe_text(item.get("artist", "Unknown"))
    style      = safe_text(item.get("style", "Unknown"))
    filename   = safe_text(item.get("filename", ""))
    similarity = item.get("similarity")
    checked    = False

    artwork_id  = item.get("id") or item.get("artwork_id")
    is_favorite = artwork_id in get_favorite_artwork_ids() if artwork_id is not None else False

    st.markdown('<div class="art-card">', unsafe_allow_html=True)

    if image_path and os.path.exists(image_path):
        try:
            st.image(Image.open(image_path), use_container_width=True)
        except Exception:
            st.info("Image preview not available.")
    else:
        st.info("Image preview not available.")

    st.markdown(f'<div class="meta-label">Artist</div><div class="meta-value">{artist}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta-label">Style</div><div class="meta-value">{style}</div>',  unsafe_allow_html=True)

    if show_similarity and similarity is not None:
        try:
            pct = float(similarity) * 100
            st.markdown(
                f'<div class="meta-label">Similarity</div>'
                f'<div class="meta-value">{pct:.1f}%</div>',
                unsafe_allow_html=True
            )
        except Exception:
            pass

    if item.get("reason"):
        render_reason(item)

    if filename:
        st.caption(filename)

    if selectable and checkbox_key:
        checked = st.checkbox("I like this artwork", key=checkbox_key)

    if st.session_state["is_authenticated"] and artwork_id is not None:
        if allow_favorite_actions:
            if is_favorite:
                st.button("Saved", key=f"saved_{artwork_id}_{checkbox_key or filename}",
                          disabled=True, use_container_width=True)
            else:
                if st.button("Save to Favorites",
                             key=f"save_{artwork_id}_{checkbox_key or filename}",
                             use_container_width=True):
                    add_to_favorites_ui(int(artwork_id))
                    st.rerun()

        if allow_remove_favorite:
            if st.button("Remove",
                         key=f"remove_{artwork_id}_{checkbox_key or filename}",
                         use_container_width=True):
                remove_from_favorites_ui(int(artwork_id))
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return checked


def render_gallery(
    items: List[Dict[str, Any]],
    selectable: bool = False,
    show_similarity: bool = False,
    prefix: str = "item",
    columns: int = 4,
    allow_favorite_actions: bool = False,
    allow_remove_favorite: bool = False
):
    selected_indices = []
    if not items:
        return selected_indices

    cols = st.columns(columns)
    for i, item in enumerate(items):
        with cols[i % columns]:
            checkbox_key = None
            if selectable:
                emb_idx      = item.get("embedding_index")
                checkbox_key = f"{prefix}_{emb_idx}_{i}"

            checked = render_art_card(
                item,
                show_similarity=show_similarity,
                selectable=selectable,
                checkbox_key=checkbox_key,
                allow_favorite_actions=allow_favorite_actions,
                allow_remove_favorite=allow_remove_favorite
            )

            if selectable and checked:
                emb_idx = item.get("embedding_index")
                if emb_idx is not None:
                    selected_indices.append(int(emb_idx))

    return selected_indices


# =========================================================
# SIDEBAR
# =========================================================
pages = [
    "Home",
    "Account",
    "Onboarding",
    "Personal Recommendations",
    "Favorites",
    "Search by Image"
]

with st.sidebar:
    st.markdown('<div class="sidebar-title">Museum Recommender</div>', unsafe_allow_html=True)

    # ── Статус бэкенда ──────────────────────────────────
    if st.session_state["backend_ok"] is None:
        st.session_state["backend_ok"] = check_backend()

    if st.session_state["backend_ok"]:
        st.markdown(
            '<div style="color:#2d6a2d;font-size:12px;margin-bottom:8px">● Backend online</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color:#c0392b;font-size:12px;margin-bottom:8px">'
            '● Backend offline — запусти: uvicorn main:app --reload</div>',
            unsafe_allow_html=True
        )
        if st.button("Проверить снова"):
            st.session_state["backend_ok"] = check_backend()
            st.rerun()

    selected_page = st.radio("Navigation", 
         pages, #key="page_selector"
         index=pages.index(st.session_state["current_page"]),
    )
    if selected_page != st.session_state["current_page"]:
      st.session_state["current_page"] = selected_page

    st.markdown("---")

    if st.session_state["is_authenticated"] and st.session_state["username"]:
        st.markdown(
            f'<div class="welcome-box"><b>Welcome, {html.escape(st.session_state["username"])}</b><br>'
            f'<span class="small-note">Your personal art profile is active.</span></div>',
            unsafe_allow_html=True
        )
        st.button("Log out", use_container_width=True, on_click=logout)
    else:
        st.markdown(
            '<div class="welcome-box"><b>Guest mode</b><br>'
            '<span class="small-note">Create an account or log in to save preferences '
            'and get personalized recommendations.</span></div>',
            unsafe_allow_html=True
        )

page = st.session_state["current_page"]


# =========================================================
# HOME
# =========================================================
if page == "Home":
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Discover Art Through Personal Taste</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">An intelligent museum assistant that recommends artworks '
        'based on visual preferences, onboarding choices, saved favorites, and uploaded images.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.15, 1])
    with col1:
        st.markdown('<div class="section-title">About the system</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-text">This prototype is designed to reduce museum fatigue and '
            'help visitors discover artworks in a more meaningful and personalized way. '
            'The system combines computer vision, interaction-based personalization, '
            'favorites, and visual similarity search.</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        feat1, feat2 = st.columns(2)
        with feat1:
            st.markdown('<div class="feature-box"><div class="feature-title">Onboarding-based cold start</div>'
                        'Select artworks you like and let the system build your initial taste profile.</div>',
                        unsafe_allow_html=True)
        with feat2:
            st.markdown('<div class="feature-box"><div class="feature-title">Personalized recommendations</div>'
                        'Receive artwork suggestions based on your evolving preference vector.</div>',
                        unsafe_allow_html=True)
        feat3, feat4 = st.columns(2)
        with feat3:
            st.markdown('<div class="feature-box"><div class="feature-title">Visual similarity search</div>'
                        'Upload an image and retrieve artworks with similar visual features.</div>',
                        unsafe_allow_html=True)
        with feat4:
            st.markdown('<div class="feature-box"><div class="feature-title">Favorites collection</div>'
                        'Save artworks you love and build your own curated museum-inspired collection.</div>',
                        unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">How to begin</div>', unsafe_allow_html=True)
        st.markdown("""
1. Open the **Account** page
2. Register or log in
3. Complete the **Onboarding** step
4. Explore your recommendations
5. Save your favorite artworks
6. Optionally upload an image for visual search
        """)
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        if not st.session_state["is_authenticated"]:
            st.button(
                  "Go to Account",
                  on_click=set_page,
                  args=("Account",),
                  use_container_width=True
               )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.button(
                  "Start Onboarding",
                  on_click=set_page,
                  args=("Onboarding",),
                  use_container_width=True
               )
                
               # if st.button("Start Onboarding", use_container_width=True):
               #   set_page("Onboarding")
            with col_b:
                st.button(
                  "Open Favorites",
                  on_click=set_page,
                  args=("Favorites",),
                  use_container_width=True
               )
                #if st.button("Open Favorites", use_container_width=True):
                 #   set_page("Favorites")


# =========================================================
# ACCOUNT
# =========================================================
elif page == "Account":
    st.markdown('<div class="section-title">Your Account</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">Sign in to store interactions, save favorites, '
        'build a stable preference profile, and receive personalized artwork recommendations.</div>',
        unsafe_allow_html=True
    )

    if st.session_state["is_authenticated"]:
        st.success(f"You are logged in as {st.session_state['username']}.")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Go to Onboarding",       use_container_width=True): set_page("Onboarding")
        with col_b:
            if st.button("Go to Recommendations",  use_container_width=True): set_page("Personal Recommendations")
        with col_c:
            if st.button("Go to Favorites",        use_container_width=True): set_page("Favorites")
    else:
        st.markdown('<div class="hero-box">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["Log in", "Register"])

        with tab_login:
            st.subheader("Log in")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")

            if st.button("Log in", use_container_width=True):
                err = validate_login(login_username, login_password)
                if err:
                    st.error(err)
                else:
                    try:
                        with st.spinner("Logging in…"):
                            result = api_post(
                                "/users/login",
                                json_data={
                                    "username": login_username.strip(),
                                    "password": login_password
                                }
                            )
                        st.session_state["user_id"]        = result["user_id"]
                        st.session_state["username"]       = result["username"]
                        st.session_state["is_authenticated"] = True
                        reset_recommendation_state()
                        st.success("Logged in successfully.")
                        set_page("Personal Recommendations")
                    except Exception as e:
                        st.error(parse_api_error(e))

        with tab_register:
            st.subheader("Create account")
            reg_username = st.text_input("Choose a username", key="reg_username")
            reg_password = st.text_input("Choose a password", type="password", key="reg_password")
            reg_confirm  = st.text_input("Confirm password",  type="password", key="reg_confirm")

            if st.button("Create account", use_container_width=True):
                err = validate_registration(reg_username, reg_password, reg_confirm)
                if err:
                    st.error(err)
                else:
                    try:
                        with st.spinner("Creating account…"):
                            result = api_post(
                                "/users/register",
                                json_data={
                                    "username": reg_username.strip(),
                                    "password": reg_password,
                                    "role": "user"
                                }
                            )
                        st.session_state["user_id"]          = result["user_id"]
                        st.session_state["username"]         = result["username"]
                        st.session_state["is_authenticated"] = True
                        reset_recommendation_state()
                        st.success("Account created successfully.")
                        set_page("Onboarding")
                    except Exception as e:
                        st.error(parse_api_error(e))

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ONBOARDING
# =========================================================
elif page == "Onboarding":
    st.markdown('<div class="section-title">Onboarding Preference Test</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">Select the artworks that appeal to you. '
        'The system will use them to create your initial preference profile.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state["is_authenticated"] or st.session_state["user_id"] is None:
        st.warning("Please log in first on the Account page.")
        if st.button("Go to Account", use_container_width=True):
            set_page("Account")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            total_n   = st.slider("Number of artworks",        min_value=6,  max_value=16, value=10)
        with col2:
            per_style = st.slider("Artworks per style",        min_value=1,  max_value=3,  value=1)
        with col3:
            top_n     = st.slider("Number of recommendations", min_value=4,  max_value=12, value=6)

        col_a, col_b = st.columns(2)
        with col_a:
            load_btn    = st.button("Load Onboarding Gallery", use_container_width=True)
        with col_b:
            shuffle_btn = st.button("Shuffle Gallery",         use_container_width=True)

        if load_btn or shuffle_btn:
            try:
                with st.spinner("Loading artworks…"):
                    result = api_get(
                        "/onboarding/candidates",
                        params={
                            "total_n":      total_n,
                            "per_style":    per_style,
                            "random_state": random.randint(1, 1_000_000)
                        }
                    )
                st.session_state["onboarding_candidates"]    = result.get("candidates", [])
                st.session_state["onboarding_recommendations"] = []
            except Exception as e:
                st.error(parse_api_error(e))

        candidates = st.session_state["onboarding_candidates"]
        if candidates:
            st.markdown("### Select the artworks you like")
            selected_indices = render_gallery(
                candidates,
                selectable=True,
                show_similarity=False,
                prefix="onboarding",
                columns=4,
                allow_favorite_actions=False
            )

            if st.button("Generate My Recommendations", use_container_width=True):
                if not selected_indices:
                    st.error("Please select at least one artwork.")
                else:
                    try:
                        with st.spinner("Building your taste profile and finding recommendations…"):
                            result = api_post(
                                "/onboarding/submit",
                                json_data={
                                    "user_id": st.session_state["user_id"],
                                    "selected_embedding_indices": selected_indices,
                                    "top_n": top_n
                                }
                            )
                        st.session_state["onboarding_recommendations"] = result.get("recommendations", [])
                        load_favorites()
                        st.success("Your first personalized recommendations are ready!")
                    except Exception as e:
                        st.error(parse_api_error(e))

        if st.session_state["onboarding_recommendations"]:
            st.markdown("### Your onboarding recommendations")
            render_gallery(
                st.session_state["onboarding_recommendations"],
                selectable=False,
                show_similarity=True,
                prefix="onb_rec",
                columns=4,
                allow_favorite_actions=True
            )


# =========================================================
# PERSONAL RECOMMENDATIONS
# =========================================================
elif page == "Personal Recommendations":
    st.markdown('<div class="section-title">Personal Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">Get recommendations based on your saved interactions, '
        'favorites, and evolving preference profile.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state["is_authenticated"] or st.session_state["user_id"] is None:
        st.warning("Please log in first on the Account page.")
        if st.button("Go to Account", use_container_width=True):
            set_page("Account")
    else:
        mode = st.radio(
            "Recommendation mode",
            ["Balanced", "More similar", "More diverse"],
            horizontal=True,
            key="recommendation_mode"
        )

        col1, col2 = st.columns(2)
        with col1:
            top_n_user = st.slider("Number of recommendations", min_value=4, max_value=12, value=6)
        with col2:
            use_weighted = st.checkbox("Use weighted preference profile", value=True)

        if mode == "More similar":
            style_bonus, max_per_style, max_per_artist = 0.00, 4, 2
            st.markdown('<div class="mode-chip">Focused on strong visual similarity</div>', unsafe_allow_html=True)
        elif mode == "Balanced":
            style_bonus, max_per_style, max_per_artist = 0.02, 2, 1
            st.markdown('<div class="mode-chip">Balanced between similarity and diversity</div>', unsafe_allow_html=True)
        else:
            style_bonus, max_per_style, max_per_artist = 0.05, 1, 1
            st.markdown('<div class="mode-chip">Encourages variety across styles and artists</div>', unsafe_allow_html=True)

        if st.button("Get Recommendations", use_container_width=True, key="get_user_recommendations_btn"):
            try:
                with st.spinner("Analysing your taste profile and finding matches…"):
                    result = api_post(
                        "/recommend/for-user",
                        json_data={
                            "user_id":              st.session_state["user_id"],
                            "top_n":                top_n_user,
                            "use_weighted_profile": use_weighted,
                            "style_bonus_value":    style_bonus,
                            "max_per_style":        max_per_style,
                            "max_per_artist":       max_per_artist,
                            "rebuild_profile":      True
                        }
                    )

                recs = result.get("recommendations", [])
                st.session_state["user_recommendations"] = recs

                if recs:
                    load_favorites()
                    st.success(f"Found {len(recs)} recommendations!")
                else:
                    st.warning(
                        "No recommendations yet. "
                        "Complete onboarding or add favourites first."
                    )

            except Exception as e:
                st.error(parse_api_error(e))

        if st.session_state["user_recommendations"]:
            render_gallery(
                st.session_state["user_recommendations"],
                selectable=False,
                show_similarity=True,
                prefix="user_rec",
                columns=4,
                allow_favorite_actions=True
            )


# =========================================================
# FAVORITES
# =========================================================
elif page == "Favorites":
    st.markdown('<div class="section-title">My Favorite Collection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">Save artworks you love and build a personal collection.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state["is_authenticated"] or st.session_state["user_id"] is None:
        st.warning("Please log in first on the Account page.")
        if st.button("Go to Account", use_container_width=True):
            set_page("Account")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Refresh Favorites",    use_container_width=True): load_favorites()
        with col_b:
            if st.button("Go to Recommendations", use_container_width=True): set_page("Personal Recommendations")

        load_favorites()
        favorites = st.session_state.get("favorites", [])

        if not favorites:
            st.info("Your collection is empty. Save artworks from your recommendations to see them here.")
        else:
            render_gallery(
                favorites,
                selectable=False,
                show_similarity=False,
                prefix="fav",
                columns=4,
                allow_favorite_actions=False,
                allow_remove_favorite=True
            )


# =========================================================
# SEARCH BY IMAGE
# =========================================================
elif page == "Search by Image":
    st.markdown('<div class="section-title">Search by Image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">Upload an image and the system will retrieve '
        'visually similar artworks from the dataset.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    top_n_image   = st.slider("Number of results", min_value=4, max_value=12, value=6)

    if uploaded_file is not None:
        try:
            st.image(Image.open(uploaded_file), caption="Uploaded image", width=320)
        except Exception:
            st.warning("Could not preview uploaded image.")

    if st.button("Find Similar Artworks", use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload an image first.")
        else:
            try:
                with st.spinner("Searching for visually similar artworks…"):
                    files  = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    result = api_post("/recommend/by-image", files=files, params={"top_n": top_n_image})
                st.session_state["image_recommendations"] = result.get("recommendations", [])
                load_favorites()
                st.success("Done!")
            except Exception as e:
                st.error(parse_api_error(e))

    if st.session_state["image_recommendations"]:
        st.markdown("### Similar artworks")
        render_gallery(
            st.session_state["image_recommendations"],
            selectable=False,
            show_similarity=True,
            prefix="img_rec",
            columns=4,
            allow_favorite_actions=True
        )