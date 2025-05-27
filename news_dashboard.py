# -*- coding: utf-8 -*-
"""
에스엔시스 뉴스 대시보드 (모던 UI/UX) – v2.2  
• BWMS 키워드 누락 해결 → 영문·숫자 토큰 허용(2–15자) & 빈 TF‑IDF 안전 처리  
• ‘NULL’ 토큰·불용어 추가 필터링, 키워드 없을 때 ‘–’ 출력  
• ThreadPool 경고 완전 억제(log 필터)  
• 기타 소소한 리팩터링
"""

from __future__ import annotations

# ------------------------------------------------------
# 경고 억제 설정 (Streamlit 로드 전에!)
# ------------------------------------------------------
import os, logging, warnings
os.environ["STREAMLIT_SUPPRESS_NO_SCRIPT_RUN_CONTEXT_WARNING"] = "true"  # 내부 옵션
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")  # fallback
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

# 표준 라이브러리
import re, json, time, hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

# 외부 라이브러리
import requests, pandas as pd, altair as alt, streamlit as st
from bs4 import BeautifulSoup
from feedparser import parse as rss_parse
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------
st.set_page_config(
    page_title="에스엔시스 뉴스 보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 기본 테마 스크립트
st.markdown(
    """
    <script>try{if(!localStorage.getItem('theme'))localStorage.setItem('theme','light');}catch(e){}</script>
    """,
    unsafe_allow_html=True,
)

# 커스텀 CSS
st.markdown(
    """
    <style>
      .stApp{background:#F8F9FA;font-family:'Segoe UI',sans-serif}
      [data-testid="stHeader"]{background:#2C3E50!important}
      [data-testid="stHeader"] h1{color:#fff!important}
      [data-testid="stSidebar"]{background:#fff;border-right:1px solid #E1E4E8}
      .stMetric{background:#fff;border:1px solid #E1E4E8;border-radius:8px;padding:1rem}
      .stExpander{background:#fff;border-radius:8px;margin-bottom:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# 로깅
# ------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("NewsBoard")

# ------------------------------------------------------
# 전역 상수
# ------------------------------------------------------
CACHE_FILE = Path.home() / ".news_cache.json"
CACHE_MAX_DAYS = 30
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

FIXED_QUERIES: Dict[str, str] = {
    "에스엔시스": "에스엔시스 OR S&SYS",
    "삼성중공업": "삼성중공업",
    "한화오션": "한화오션",
}

PRODUCT_QUERIES: List[tuple[str, List[str]]] = [
    ("BWMS", ["BWMS", "BWTS", "선박평형수"]),
    ("IAS", ["선박용 제어시스템", "선박 제어시스템", "콩스버그", "선박용 IAS"]),
    ("FGSS", ["FGSS", "LFSS", "선박이중연료시스템"]),
]

# 불용어(검색어·회사/제품명·일반 단어)
NOISE_WORDS: set[str] = {
    *(kw.lower() for kw in FIXED_QUERIES),
    *[w.lower() for _, syn in PRODUCT_QUERIES for w in syn],
    "null", "뉴스", "기사", "최근", "사진", "제공", "대한", "관련"
}

# ------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------

def _shorten(txt: str, width: int = 60) -> str:
    return txt if len(txt) <= width else txt[: width] + "…"


def parse_datetime(s: str) -> Optional[datetime]:
    """다양한 날짜 문자열 → KST tz-aware datetime"""
    if not s:
        return None
    s = s.strip()
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    # 상대표기
    rel_patterns = {
        r"(\d+)분 전": lambda m: now - timedelta(minutes=int(m[1])),
        r"(\d+)시간 전": lambda m: now - timedelta(hours=int(m[1])),
        r"(\d+)일 전": lambda m: now - timedelta(days=int(m[1])),
    }
    for pat, func in rel_patterns.items():
        if (m := re.match(pat, s)):
            return func(m)
    if s in ("어제", "하루 전"):
        return now - timedelta(days=1)
    if s == "오늘":
        return now
    # 절대표기
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y.%m.%d.",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            dt = dt.replace(tzinfo=ZoneInfo("Asia/Seoul")) if dt.tzinfo is None else dt.astimezone(ZoneInfo("Asia/Seoul"))
            return dt
        except ValueError:
            continue
    return None


def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_top_keywords(docs: List[str], top_n: int = 5, exclude: set[str] | None = None) -> List[str]:
    texts = [clean_text(d) for d in docs if d.strip()]
    if not texts:
        return []
    try:
        vect = TfidfVectorizer(
            token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,15}\b",
            ngram_range=(1, 3),
            max_features=500,
        )
        X = vect.fit_transform(texts)
    except ValueError:  # 빈 어휘
        return []
    scores = X.sum(axis=0).A1
    terms = vect.get_feature_names_out()
    josa = re.compile(r"(으로|로|와|과|이|가|은|는|도)$")
    exclude_low = {w.lower() for w in (exclude or set())} | NOISE_WORDS
    result: List[str] = []
    for term, score in sorted(zip(terms, scores), key=lambda x: -x[1]):
        ct = josa.sub("", term)
        if 2 <= len(ct) <= 15 and ct.lower() not in exclude_low:
            result.append(ct)
        if len(result) >= top_n:
            break
    return result

# ------------------------------------------------------
# 네트워크 요청 유틸
# ------------------------------------------------------

def _request(url: str, **kwargs):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"{url[:60]} … 재시도 ({e})")
            time.sleep(1)
    raise RuntimeError(f"요청 실패: {url}")

# ------------------------------------------------------
# 기사 수집 함수
# ------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_newsapi(q: str) -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    since = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q": q,
        "language": "ko",
        "sortBy": "publishedAt",
        "from": since,
        "apiKey": NEWS_API_KEY,
        "pageSize": 100,
    }
    try:
        arts = _request("https://newsapi.org/v2/everything", params=params).json().get("articles", [])
        for a in arts:
            a.setdefault("origins", []).append("newsapi")
            a["content"] = a.get("content", "") or ""
        return arts
    except Exception:
        logger.exception("NewsAPI 오류")
        return []

@st.cache_data(ttl=3600)
def fetch_rss(q: str) -> List[Dict]:
    out, seen = [], set()
    for term in re.split(r"\s+OR\s+", q):
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(term)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            feed = rss_parse(_request(url).text)
            for e in feed.entries:
                if e.link in seen:
                    continue
                seen.add(e.link)
                dt = time.strftime("%Y-%m-%d %H:%M", e.published_parsed) if getattr(e, "published_parsed", None) else ""
                out.append(
                    {
                        "title": e.title,
                        "url": e.link,
                        "publishedAt": dt,
                        "content": BeautifulSoup(e.get("summary", ""), "html.parser").get_text(),
                        "origins": ["rss"],
                    }
                )
        except Exception:
            logger.warning(f"RSS 오류: {url}")
    return out

@st.cache_data(ttl=3600)
def fetch_naver(q: str) -> List[Dict]:
    out = []
    try:
        url = f"https://search.naver.com/search.naver?where=news&query={requests.utils.quote(q)}&sort=1"
        soup = BeautifulSoup(_request(url).text, "html.parser")
        items = soup.select("li.bx") + soup.select("div.news_area")
        for it in items:
            a_tag = it.select_one("a.news_tit")
            if not a_tag:
                continue
            title = a_tag.get("title") or a_tag.get_text(strip=True)
            link = a_tag["href"]
            dt_tag = it.select_one("span.date") or it.select_one("span.info")
            dt = dt_tag.get_text(strip=True) if dt_tag else ""
            desc = it.select_one("a.api_txt_lines") or it.select_one("div.news_dsc")
            content = desc.get_text(strip=True) if desc else ""
            out.append(
                {
                    "title": title,
                    "url": link,
                    "publishedAt": dt,
                    "content": content,
                    "origins": ["naver"],
                }
            )
    except Exception:
        logger.exception("Naver 스크래핑 오류")
    terms = [t.lower() for t in re.split(r"\s+OR\s+", q) if t.strip()]
    return [a for a in out if any(t in (a["title"] + " " + a["content"]).lower() for t in terms)]

# ------------------------------------------------------
# 캐시 및 중복 관리
# ------------------------------------------------------

def _purge_old(cache: Dict[str, Dict]) -> Dict[str, Dict]:
    threshold = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=CACHE_MAX_DAYS)
    return {uid: a for uid, a in cache.items() if (parse_datetime(a.get("publishedAt", "")) or threshold) >= threshold}


def update_cache(arts: List[Dict]) -> None:
    cache: Dict[str, Dict] = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text("utf-8"))
        except json.JSONDecodeError:
            pass
    cache = _purge_old(cache)
    changed = False
    for a in arts:
        url = a.get("url", "")
        if not url:
            continue
        uid = hashlib.sha256(url.encode()).hexdigest()
        if uid not in cache:
            cache[uid] = a
            changed = True
    if changed:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")


def dedup(arts: List[Dict]) -> List[Dict]:
    seen, uniq = set(), []
    for a in arts:
        url = a.get("url")
        if url and url not in seen:
            uniq.append(a)
            seen.add(url)
    return uniq

# ------------------------------------------------------
# 통합 수집
# ------------------------------------------------------

def fetch_all(q: str, mode: str, use_nv: bool) -> List[Dict]:
    tasks = []
    if mode != "RSS만":
        tasks.append(("newsapi", lambda: fetch_newsapi(q)))
    if mode != "NewsAPI만":
        tasks.append(("rss", lambda: fetch_rss(q)))
    if use_nv:
        tasks.append(("naver", lambda: fetch_naver(q)))

    arts = []
    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(fn): name for name, fn in tasks}
        for fut in as_completed(futures):
            try:
                arts.extend(fut.result())
            except Exception:
                logger.exception(f"{futures[fut]} 수집 실패")
    arts = dedup(arts)
    update_cache(arts)
    return arts

# ------------------------------------------------------
# 추이 분석
# ------------------------------------------------------

def analyze_trends(arts: List[Dict], kw_map: Dict[str, List[str]], start: date, end: date) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    cmap = {d.strftime("%Y-%m-%d"): {c: 0 for c in kw_map} for d in dates}
    for it in arts:
        dt = parse_datetime(it.get("publishedAt", ""))
        if not dt:
            continue
        d = dt.strftime("%Y-%m-%d")
        if d not in cmap:
            continue
        txt = (it.get("title", "") + " " + it.get("content", "")).lower()
        for comp, kws in kw_map.items():
            if any(k.lower() in txt for k in kws):
                cmap[d][comp] += 1
    rows = [{"date": d, "company": c, "count": n} for d, v in cmap.items() for c, n in v.items()]
    df = pd.DataFrame(rows)
    df["date_fmt"] = pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    return df

# ------------------------------------------------------
# 메인 UI
# ------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.header("필터 설정")
    mode = st.sidebar.selectbox("뉴스 소스", ["전체 (네이버 포함)", "전체 (네이버 제외)", "RSS만", "NewsAPI만"], 0)
    use_nv = "포함" in mode
    cnt = st.sidebar.slider("기사 표시 건수", 5, 30, 10, 5)
    today = date.today(); default_start = today - timedelta(days=30)
    start_date, end_date = st.sidebar.date_input("분석 기간", [default_start, today])
    if start_date > end_date:
        st.sidebar.error("시작일은 종료일 이전이어야 합니다.")
    st.sidebar.markdown("---")
    comp1 = st.sidebar.text_input("회사1 (동적)", "한라IMS")
    comp2 = st.sidebar.text_input("회사2 (동적)", "파나시아")
    st.sidebar.markdown("---")
    all_comps = list(FIXED_QUERIES) + [comp1, comp2]
    selected = [c for c in all_comps if st.sidebar.checkbox(c, True)]
    if st.sidebar.button("🔄 새로고침"):
        st.cache_data.clear()
    if not NEWS_API_KEY:
        st.sidebar.warning("NEWS_API_KEY 미설정 – NewsAPI 기사 제외")

    # Title & metrics
    st.title("에스엔시스 뉴스 보드")
    cols = st.columns(len(FIXED_QUERIES) + 2)
    data_map: Dict[str, List[Dict]] = {}
    with st.spinner("뉴스 수집 중..."):
        for i, comp in enumerate(list(FIXED_QUERIES) + [comp1, comp2]):
            arts = [
                a for a in fetch_all(FIXED_QUERIES.get(comp, comp), mode, use_nv)
                if (dt := parse_datetime(a.get("publishedAt", ""))) and start_date <= dt.date() <= end_date
            ]
            data_map[comp] = arts
            cols[i].metric(f"{comp} 기사 수", len(arts))

    st.markdown("---")

    # 업체별 최신 뉴스
    tabs = st.tabs(list(data_map.keys()))
    for tab, comp in zip(tabs, data_map):
        with tab:
            st.subheader(f"{comp} 최신 뉴스 (상위 {cnt}건)")
            subset = sorted(data_map[comp], key=lambda x: parse_datetime(x["publishedAt"]) or datetime.min, reverse=True)[:cnt]
            if not subset:
                st.info("현재 조회할 기사가 없습니다.")
            for a in subset:
                ts = parse_datetime(a["publishedAt"]); ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(f"- [{_shorten(a['title'])}]({a['url']}) <span style='color:#6B7280;'>({ts_str})</span>", unsafe_allow_html=True)

    st.markdown("---")

    # 제품별 최신 뉴스
    st.subheader("제품별 최신 뉴스")
    p_tabs = st.tabs([t for t, _ in PRODUCT_QUERIES])
    for tab, (title, syns) in zip(p_tabs, PRODUCT_QUERIES):
        with tab:
            st.subheader(f"{title} 최신 뉴스 (상위 {cnt}건)")
            q = " OR ".join(syns)
            arts_display = sorted(
                fetch_all(q, mode, use_nv),
                key=lambda x: parse_datetime(x.get("publishedAt", "")) or datetime.min,
                reverse=True,
            )[:cnt]
            if not arts_display:
                st.info("현재 조회할 기사가 없습니다.")
            for a in arts_display:
                ts = parse_datetime(a["publishedAt"]); ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(f"- [{_shorten(a['title'])}]({a['url']}) <span style='color:#6B7280;'>({ts_str})</span>", unsafe_allow_html=True)

    st.markdown("---")

    # 기업별 키워드
    with st.expander("🔑 주요 키워드 분석 (상위 5개)", True):
        kcols = st.columns(len(data_map))
        for col, comp in zip(kcols, data_map):
            texts = [a["title"] + " " + a.get("content", "") for a in data_map[comp][:30]]
            kws = extract_top_keywords(texts, 5, {comp})
            col.markdown(f"**{comp}**")
            if kws:
                for w in kws:
                    col.write(f"- {w}")
            else:
                col.write("–")

    # 제품별 키워드
    with st.expander("🔑 제품별 주요 키워드 분석 (상위 5개)", False):
        pcols = st.columns(len(PRODUCT_QUERIES))
        for col, (title, syns) in zip(pcols, PRODUCT_QUERIES):
            q = " OR ".join(syns)
            arts = [
                a for a in fetch_all(q, mode, use_nv)
                if (dt := parse_datetime(a.get("publishedAt", ""))) and start_date <= dt.date() <= end_date
            ][:30]
            texts = [a["title"] + " " + a.get("content", "") for a in arts]
            kws = extract_top_keywords(texts, 5, set(s.lower() for s in syns))
            col.markdown(f"**{title}**")
            if kws:
                for w in kws:
                    col.write(f"- {w}")
            else:
                col.write("–")

    st.markdown("---")

    # 노출 추이
    st.subheader("노출 추이 분석")
    df = analyze_trends(
        sum(data_map.values(), []),
        {**{k: [k] for k in FIXED_QUERIES}, comp1: [comp1], comp2: [comp2]},
        start_date,
        end_date,
    )
    df = df[df["company"].isin(selected)]
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date_fmt:O", title="날짜"),
            y=alt.Y("count:Q", title="건수"),
            color=alt.Color("company:N", title="회사"),
            tooltip=["date_fmt", "company", "count"],
        )
        .properties(width="container", height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
