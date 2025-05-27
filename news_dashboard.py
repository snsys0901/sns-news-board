# -*- coding: utf-8 -*-
"""
에스엔시스 뉴스 대시보드  (v2.6, 2025-05-27)
────────────────────────────────────────────────────────
• BWMS / IAS / FGSS + 동적 2종(제품1·제품2) = 5개 제품 모니터링
• 키워드 추출 보강 → 항상 최대 5개, 제목 동일 토큰 제외
• 그래프 .interactive() 적용(휠 줌·드래그 팬 가능)
• 사이드바 라벨 명확화 (제품1 / 제품2)
"""

# ── 공통 import / 경고 억제 ──────────────────────────────
import os
import warnings
import logging
import re
import json
import time
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup
from feedparser import parse as rss_parse
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["STREAMLIT_SUPPRESS_NO_SCRIPT_RUN_CONTEXT_WARNING"] = "true"
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
logging.getLogger(
    "streamlit.runtime.scriptrunner.script_run_context"
).setLevel(logging.ERROR)

# ── 페이지 & 스타일 ─────────────────────────────────────
st.set_page_config(page_title="에스엔시스 뉴스 보드", layout="wide")
st.markdown(
    """<script>try{localStorage.setItem('theme','light');}catch(e){}</script>""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
.stApp{background:#F8F9FA;font-family:'Segoe UI',sans-serif}
[data-testid="stHeader"]{background:#2C3E50!important}
[data-testid="stHeader"] h1{color:#fff!important}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #E1E4E8}
.stMetric{background:#fff;border:1px solid #E1E4E8;border-radius:8px;padding:1rem}
.stExpander{background:#fff;border-radius:8px;margin-bottom:1rem}
</style>""",
    unsafe_allow_html=True,
)

# ── 로깅 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NewsBoard")

# ── 전역 상수 ───────────────────────────────────────────
CACHE_FILE = Path.home() / ".news_cache.json"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

FIXED_QUERIES = {
    "에스엔시스": "에스엔시스 OR S&SYS",
    "삼성중공업": "삼성중공업",
    "한화오션": "한화오션",
}

STATIC_PRODUCTS = [
    ("BWMS", ["BWMS", "BWTS", "선박평형수"]),
    ("IAS", [
        "IAS",
        "통합자동화시스템",
        "Integrated Automation System",
        "선박용 제어시스템",
        "선박 제어시스템",
        "콩스버그",
        "선박용 IAS",
    ]),
    ("FGSS", ["FGSS", "LFSS", "선박이중연료시스템"]),
]

DEFAULT_P1_SYNS = ["배전반", "수배전반", "전력배전반", "전력기기"]
DEFAULT_P2_SYNS = [
    "친환경", "친환경 선박", "그린십", "탈탄소", "저탄소 선박",
]

NOISE_WORDS = {
    "null", "뉴스", "기사", "사진", "최근",
    *FIXED_QUERIES,
}

# ── 유틸 함수 ───────────────────────────────────────────
def _shorten(t: str, w: int = 60) -> str:
    return t if len(t) <= w else t[:w] + "…"

def parse_datetime(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    for pat, unit in [(r"(\d+)분 전", "minutes"), (r"(\d+)시간 전", "hours")]:
        if m := re.match(pat, s):
            return now - timedelta(**{unit: int(m[1])})
    if s in ("오늘", "today"):
        return now
    if s in ("어제", "yesterday"):
        return now - timedelta(days=1)
    for fmt in ("%Y-%m-%d %H:%M", "%Y.%m.%d.", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=ZoneInfo("Asia/Seoul"))
        except ValueError:
            continue
    return None

def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _tfidf(texts: List[str], stop: set, top_n: int) -> List[str]:
    vect = TfidfVectorizer(
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b",
        ngram_range=(1, 3),
        max_features=500,
    )
    try:
        X = vect.fit_transform(texts)
    except ValueError:
        return []
    scores = X.sum(axis=0).A1
    terms = vect.get_feature_names_out()
    josa = re.compile(r"(으로|로|와|과|이|가|은|는|도|을|를)$")
    cand: List[str] = []
    for t, sc in sorted(zip(terms, scores), key=lambda x: -x[1]):
        tok = josa.sub("", t)
        if tok.lower() not in stop and not tok.isdigit():
            cand.append(tok)
        if len(cand) >= top_n:
            break
    return cand

def _freq_fallback(texts: List[str], stop: set, top_n: int) -> List[str]:
    freq = Counter()
    for line in texts:
        for tok in re.findall(r"[가-힣A-Za-z0-9]{2,}", line):
            if tok.lower() not in stop and not tok.isdigit():
                freq[tok] += 1
    return [w for w, _ in freq.most_common(top_n)]

def extract_top_keywords(
    docs: List[str],
    top_n: int,
    extra_stop: set,
    fallback_word: str,
) -> List[str]:
    texts = [clean_text(d) for d in docs if d.strip()]
    stop = NOISE_WORDS | extra_stop
    if not texts:
        return [fallback_word]
    kws = _tfidf(texts, stop, top_n)
    if len(kws) < top_n:
        more = _freq_fallback(texts, stop, top_n * 2)
        for w in more:
            if w not in kws:
                kws.append(w)
            if len(kws) >= top_n:
                break
    kws = [k for k in kws if k.lower() != fallback_word.lower()][:top_n]
    return kws or [fallback_word]

def dedup(lst: List[Dict]) -> List[Dict]:
    seen, uniq = set(), []
    for a in lst:
        url = a.get("url")
        if url and url not in seen:
            uniq.append(a)
            seen.add(url)
    return uniq

# ── 기사 수집 함수 ────────────────────────────────────
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
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        r.raise_for_status()
        arts = r.json().get("articles", [])
        for a in arts:
            a.setdefault("origins", []).append("newsapi")
            a["content"] = a.get("content", "") or ""
    except Exception:
        logger.exception("NewsAPI 오류")
        arts = []
    return arts

@st.cache_data(ttl=3600)
def fetch_rss(q: str) -> List[Dict]:
    out, seen = [], set()
    for term in re.split(r"\s+OR\s+", q):
        url = (
            f"https://news.google.com/rss/search?"
            f"q={requests.utils.quote(term)}&hl=ko&gl=KR&ceid=KR:ko"
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            feed = rss_parse(r.text)
            for e in feed.entries:
                if e.link in seen:
                    continue
                seen.add(e.link)
                summary = BeautifulSoup(e.get("summary", ""), "html.parser").get_text()
                dt = (time.strftime("%Y-%m-%d %H:%M", e.published_parsed)
                    if getattr(e, "published_parsed", None) else "")
                out.append({
                    "title": e.title,
                    "url": e.link,
                    "publishedAt": dt,
                    "content": summary,
                    "origins": ["rss"],
                })
        except Exception:
            logger.warning(f"RSS 오류: {url}")
    return out

@st.cache_data(ttl=3600)
def fetch_naver(q: str) -> List[Dict]:
    try:
        url = f"https://search.naver.com/search.naver?where=news&query={requests.utils.quote(q)}&sort=1"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception:
        logger.exception("Naver 스크래핑 오류")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for it in soup.select("li.bx") + soup.select("div.news_area"):
        a_tag = it.select_one("a.news_tit")
        if not a_tag:
            continue
        title = a_tag.get("title") or a_tag.get_text(strip=True)
        link = a_tag["href"]
        dt_tag = it.select_one("span.date") or it.select_one("span.info")
        dt = dt_tag.get_text(strip=True) if dt_tag else ""
        desc = it.select_one("a.api_txt_lines") or it.select_one("div.news_dsc")
        content = desc.get_text(strip=True) if desc else ""
        out.append({
            "title": title,
            "url": link,
            "publishedAt": dt,
            "content": content,
            "origins": ["naver"],
        })
    return out

# ── fetch_all & 캐시 ───────────────────────────────────
def fetch_all(q: str, mode: str, use_nv: bool) -> List[Dict]:
    funcs = []
    if mode != "RSS만":
        funcs.append(fetch_newsapi)
    if mode != "NewsAPI만":
        funcs.append(fetch_rss)
    if use_nv:
        funcs.append(fetch_naver)
    arts: List[Dict] = []
    with ThreadPoolExecutor() as ex:
        for fut in [ex.submit(fn, q) for fn in funcs]:
            arts.extend(fut.result())
    arts = dedup(arts)
    update_cache(arts)
    return arts

def update_cache(arts: List[Dict]) -> None:
    cache: Dict[str, Dict] = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text("utf-8"))
        except json.JSONDecodeError:
            pass
    changed = False
    for a in arts:
        url = a.get("url", "")
        if not url:
            continue
        uid = hashlib.sha256(url.encode()).hexdigest()
        if uid not in cache:
            cache[uid] = a
            changed = True
    purge_before = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=30)
    for uid, a in list(cache.items()):
        dt = parse_datetime(a.get("publishedAt", ""))
        if dt and dt < purge_before:
            del cache[uid]
            changed = True
    if changed:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")

# ── 추이 분석 ───────────────────────────────────────────
def analyze_trends(
    arts: List[Dict], kw_map: Dict[str, List[str]], start: date, end: date
) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    cmap = {d.strftime("%Y-%m-%d"): {c: 0 for c in kw_map} for d in dates}
    for it in arts:
        dt = parse_datetime(it.get("publishedAt", ""))
        if not dt:
            continue
        day = dt.strftime("%Y-%m-%d")
        if day not in cmap:
            continue
        txt = (it.get("title", "") + " " + it.get("content", "")).lower()
        for comp, kws in kw_map.items():
            if any(k.lower() in txt for k in kws):
                cmap[day][comp] += 1
    rows = [
        {"date": d, "company": c, "count": n}
        for d, v in cmap.items()
        for c, n in v.items()
    ]
    df = pd.DataFrame(rows)
    df["date_fmt"] = pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    return df

# ── Streamlit 메인 ─────────────────────────────────────
def main() -> None:
    # ── Sidebar 설정 ──────────────────────────────────
    # 1) 새로고침 버튼을 필터 설정 바로 위로 이동
    if st.sidebar.button("🔄 새로고침"):
        st.cache_data.clear()

    # 2) 필터 설정 헤더 및 위젯
    st.sidebar.header("필터 설정")
    mode = st.sidebar.selectbox(
        "뉴스 소스",
        ("전체 (네이버 포함)", "전체 (네이버 제외)", "RSS만", "NewsAPI만"),
        0,
    )
    use_nv = "포함" in mode
    cnt = st.sidebar.slider("기사 표시 건수", 5, 30, 10, 5)

    today = date.today()
    default_start = today - timedelta(days=30)
    start_date, end_date = st.sidebar.date_input(
        "분석 기간", (default_start, today)
    )
    if start_date > end_date:
        st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")

    st.sidebar.markdown("---")
    comp1 = st.sidebar.text_input("회사1 (동적)", "한라IMS")
    comp2 = st.sidebar.text_input("회사2 (동적)", "파나시아")
    st.sidebar.markdown("---")

    prod1_name = st.sidebar.text_input("제품1 (동적)", "배전반")
    prod2_name = st.sidebar.text_input("제품2 (동적)", "친환경")

    # ── 제품 쿼리 구성 ─────────────────────────────────
    def synonyms(name: str) -> List[str]:
        base = name.strip() or "제품"
        if base == "배전반":
            return DEFAULT_P1_SYNS
        if base == "친환경":
            return DEFAULT_P2_SYNS
        return [base]

    PRODUCT_QUERIES = STATIC_PRODUCTS + [
        (prod1_name.strip() or "제품1", synonyms(prod1_name)),
        (prod2_name.strip() or "제품2", synonyms(prod2_name)),
    ]

    # ── Title & Metrics ───────────────────────────────
    st.title("에스엔시스 뉴스 보드")
    comp_list = list(FIXED_QUERIES) + [comp1, comp2]
    cols = st.columns(len(comp_list))
    data_map: Dict[str, List[Dict]] = {}

    with st.spinner("뉴스 수집 중…"):
        for i, comp in enumerate(comp_list):
            arts = [
                a
                for a in fetch_all(FIXED_QUERIES.get(comp, comp), mode, use_nv)
                if (
                    (dt := parse_datetime(a.get("publishedAt", "")))
                    and start_date <= dt.date() <= end_date
                )
            ]
            data_map[comp] = arts
            cols[i].metric(f"{comp} 기사 수", len(arts))

    st.markdown("---")

    # ── 1) 업체별 최신 뉴스 ─────────────────────────
    for tab, comp in zip(st.tabs(data_map.keys()), data_map):
        with tab:
            st.subheader(f"{comp} 최신 뉴스 (상위 {cnt}건)")
            subset = sorted(
                data_map[comp],
                key=lambda x: parse_datetime(x["publishedAt"]) or datetime.min,
                reverse=True,
            )[:cnt]
            if not subset:
                st.info("현재 조회할 기사가 없습니다.")
            for a in subset:
                ts = parse_datetime(a["publishedAt"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(
                    f"- [{_shorten(a['title'])}]({a['url']}) "
                    f"<span style='color:#6B7280;'>({ts_str})</span>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── 2) 제품별 최신 뉴스 ─────────────────────────
    st.subheader("제품별 최신 뉴스")
    for tab, (title, syns) in zip(
        st.tabs([t for t, _ in PRODUCT_QUERIES]), PRODUCT_QUERIES
    ):
        with tab:
            st.subheader(f"{title} 최신 뉴스 (상위 {cnt}건)")
            q = " OR ".join(syns)
            arts = sorted(
                fetch_all(q, mode, use_nv),
                key=lambda x: parse_datetime(x.get("publishedAt", "")) or datetime.min,
                reverse=True,
            )[:cnt]
            if not arts:
                st.info("현재 조회할 기사가 없습니다.")
            for a in arts:
                ts = parse_datetime(a["publishedAt"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(
                    f"- [{_shorten(a['title'])}]({a['url']}) "
                    f"<span style='color:#6B7280;'>({ts_str})</span>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── 3) 기업별 키워드 ───────────────────────────
    with st.expander("🔑 주요 키워드 분석 (상위 5개)", True):
        kcols = st.columns(len(data_map))
        for col, comp in zip(kcols, data_map):
            texts = [
                a["title"] + " " + a.get("content", "")
                for a in data_map[comp][: cnt * 3]
            ]
            kws = extract_top_keywords(texts, 5, {comp.lower()}, comp)
            col.markdown(f"**{comp}**")
            for w in kws:
                col.write(f"- {w}")

    # ── 4) 제품별 키워드 ───────────────────────────
    with st.expander("🔑 제품별 주요 키워드 분석 (상위 5개)", False):
        pcols = st.columns(len(PRODUCT_QUERIES))
        for col, (title, syns) in zip(pcols, PRODUCT_QUERIES):
            q = " OR ".join(syns)
            arts = [
                a
                for a in fetch_all(q, mode, use_nv)
                if (
                    (dt := parse_datetime(a.get("publishedAt", "")))
                    and start_date <= dt.date() <= end_date
                )
            ][: cnt * 3]
            texts = [a["title"] + " " + a.get("content", "") for a in arts]
            kws = extract_top_keywords(texts, 5, {s.lower() for s in syns}, title)
            col.markdown(f"**{title}**")
            for w in kws:
                col.write(f"- {w}")

    st.markdown("---")

    # ── 5) 노출 추이 분석 ─────────────────────────
    st.subheader("노출 추이 분석")
    trend_df = analyze_trends(
        sum(data_map.values(), []),
        {**{k: [k] for k in FIXED_QUERIES}, comp1: [comp1], comp2: [comp2]},
        start_date,
        end_date,
    )
    selected = [
        comp
        for comp in comp_list
        if st.sidebar.checkbox(comp, True, key=f"cb_{comp}")
    ]
    trend_df = trend_df[trend_df["company"].isin(selected)]
    chart = (
        alt.Chart(trend_df)
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
