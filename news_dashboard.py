# -*- coding: utf-8 -*-
"""
에스엔시스 뉴스 대시보드 (모던 UI/UX)
* 기본 테마: 라이트 모드 강제 적용
* 업체별 최신 뉴스➕제품별 최신 뉴스➕기업/제품별 주요 키워드 분석 (TF–IDF 전용)
* 네이버 스크랩 + RSS/NewsAPI 지원
* parse_datetime 네이밍 통일 및 위치 수정
* 모던 테마: 커스텀 CSS, Metrics, Expander, Line 차트
"""

import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import requests
import pandas as pd
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup
from feedparser import parse as rss_parse
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------
# 1) 페이지 설정: 반드시 첫 Streamlit 명령으로 호출
# ------------------------------------------------------
st.set_page_config(
    page_title="에스엔시스 뉴스 보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------
# 2) 기본 테마 강제: 라이트 모드
# ------------------------------------------------------
st.markdown("""
    <script>
      try { window.localStorage.setItem("theme", "light"); }
      catch(e) {}
    </script>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 3) 커스텀 CSS
# ------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; font-family: 'Segoe UI', sans-serif; }
    header { background-color: #2C3E50 !important; }
    header .css-1v3fvcr h1 { color: #FFFFFF !important; }
    .css-1d391kg { background-color: #FFFFFF; border-right: 1px solid #E1E4E8; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #E1E4E8; border-radius: 8px; padding: 1rem; }
    .stExpander { background-color: #FFFFFF; border-radius: 8px; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 로깅 설정
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NewsBoard")

# ------------------------------------------------------
# 전역 설정
# ------------------------------------------------------
CACHE_FILE    = Path.home() / ".news_cache.json"
FIXED_QUERIES = {
    "에스엔시스": "에스엔시스 OR S&SYS",
    "삼성중공업": "삼성중공업",
    "한화오션":   "한화오션",
}
NOISE_WORDS   = {"rss", "news", "google", "https", "http", "com", "href", "color", "nbsp"}
NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "")

# ------------------------------------------------------
# 제품별 검색 설정
# ------------------------------------------------------
PRODUCT_QUERIES = [
    ("BWMS", ["BWMS", "BWTS", "선박평형수"]),
    ("IAS",  ["IAS",  "ICMS", "설비제어시스템"]),
    ("FGSS", ["FGSS", "LFSS", "선박이중연료시스템"]),
]

# ------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------
def _shorten(text: str, width: int = 60) -> str:
    return text if len(text) <= width else text[:width] + "…"

def parse_datetime(s: str) -> Optional[datetime]:
    if not s: return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %z"):
        try: return datetime.strptime(s, fmt)
        except ValueError: pass
    if m := re.match(r"(\d+)시간 전", s):
        return datetime.now() - timedelta(hours=int(m.group(1)))
    if m := re.match(r"(\d+)분 전", s):
        return datetime.now() - timedelta(minutes=int(m.group(1)))
    return None

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^가-힣A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_top_keywords(docs: List[str], top_n: int = 5) -> List[str]:
    """TF–IDF 점수 합계를 기준으로 상위 n개 n-gram 추출"""
    texts = [clean_text(d).lower() for d in docs if d.strip()]
    if not texts:
        return []
    # 한글+영문 단어 2자 이상, 1~2그램, max_features 충분히 크게
    vect = TfidfVectorizer(
        token_pattern=r"(?u)\b[가-힣A-Za-z]{2,}\b",
        ngram_range=(1,2),
        max_features=200
    )
    X = vect.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vect.get_feature_names_out()
    # NOISE_WORDS 제거
    filtered = [
        (terms[i], scores[i]) for i in scores.argsort()[::-1]
        if terms[i] not in NOISE_WORDS
    ]
    top_terms = [t for t,_ in filtered[:top_n]]
    return top_terms

# 이하 fetch_*, update_cache, analyze_trends 등은 이전과 동일
@st.cache_data(ttl=3600)
def fetch_newsapi(q: str) -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    since = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = dict(q=q, language="ko", sortBy="publishedAt",
                  from_=since, apiKey=NEWS_API_KEY, pageSize=100)
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        r.raise_for_status()
        arts = r.json().get("articles", [])
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
            r = requests.get(url, timeout=10); r.raise_for_status()
            feed = rss_parse(r.text)
            for e in feed.entries:
                if e.link in seen:
                    continue
                seen.add(e.link)
                content = BeautifulSoup(e.get("summary", ""), "html.parser").get_text()
                dt = (time.strftime("%Y-%m-%d %H:%M", e.published_parsed)
                      if hasattr(e, "published_parsed") else e.get("published", ""))
                out.append({
                    "title": e.title, "url": e.link,
                    "publishedAt": dt, "content": content,
                    "origins": ["rss"]
                })
        except Exception:
            logger.warning(f"RSS 오류: {url}")
    return out

@st.cache_data(ttl=3600)
def fetch_naver(q: str) -> List[Dict]:
    out = []
    try:
        url = f"https://search.naver.com/search.naver?where=news&query={requests.utils.quote(q)}&sort=1"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select("li.bx") + soup.select("div.news_area")
        for it in items:
            a_tag = it.select_one("a.news_tit")
            if not a_tag: continue
            title = a_tag.get("title") or a_tag.get_text(strip=True)
            link  = a_tag["href"]
            dt_tag = it.select_one("span.date") or it.select_one("span.info")
            dt = dt_tag.get_text(strip=True) if dt_tag else ""
            desc = it.select_one("a.api_txt_lines") or it.select_one("div.news_dsc")
            content = desc.get_text(strip=True) if desc else ""
            out.append({
                "title": title, "url": link,
                "publishedAt": dt, "content": content,
                "origins": ["naver"]
            })
    except Exception:
        logger.exception("Naver 스크래핑 오류")
    terms = [t.strip().lower() for t in re.split(r"\s+OR\s+", q) if t.strip()]
    return [
        art for art in out
        if any(term in (art["title"]+" "+art["content"]).lower() for term in terms)
    ]

def update_cache(arts: List[Dict]) -> None:
    cache = {}
    if CACHE_FILE.exists():
        try: cache = json.loads(CACHE_FILE.read_text("utf-8"))
        except: pass
    changed = False
    for a in arts:
        url = a.get("url","")
        if not url: continue
        uid = hashlib.sha256(url.encode()).hexdigest()
        if uid not in cache:
            cache[uid] = a; changed = True
    if changed:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2),"utf-8")

def fetch_all(q: str, mode: str, use_nv: bool) -> List[Dict]:
    arts = []
    if mode!="RSS만":   arts += fetch_newsapi(q)
    if mode!="NewsAPI만": arts += fetch_rss(q)
    if use_nv:        arts += fetch_naver(q)
    update_cache(arts)
    return arts

def analyze_trends(arts: List[Dict], kw_map: Dict[str,List[str]],
                   start: date, end: date) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    cmap = {d.strftime("%Y-%m-%d"):{c:0 for c in kw_map} for d in dates}
    for it in arts:
        dt = parse_datetime(it.get("publishedAt",""))
        if not dt: continue
        day = dt.strftime("%Y-%m-%d")
        if day not in cmap: continue
        txt = (it.get("title","")+" "+it.get("content","")).lower()
        for comp, kws in kw_map.items():
            if any(kw.lower() in txt for kw in kws):
                cmap[day][comp]+=1
    rows=[]
    for d,counts in cmap.items():
        for comp,c in counts.items():
            rows.append({"date":d,"company":comp,"count":c})
    df=pd.DataFrame(rows)
    df["date_fmt"]=pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    return df

# ------------------------------------------------------
# 메인
# ------------------------------------------------------
def main():
    # 사이드바
    st.sidebar.header("🔎 필터 설정")
    mode = st.sidebar.selectbox("뉴스 소스",
        ["전체 (네이버 포함)","전체 (네이버 제외)","RSS만","NewsAPI만"], index=0)
    use_nv = "포함" in mode
    cnt = st.sidebar.slider("기사 표시 건수",5,20,10,step=5)

    today = date.today()
    default_start = today - timedelta(days=30)
    start_date, end_date = st.sidebar.date_input("분석 기간",
                                                 [default_start,today])
    if start_date>end_date:
        st.sidebar.error("시작일은 종료일 이전이어야 합니다.")
    st.sidebar.markdown("---")
    comp1 = st.sidebar.text_input("회사1 (동적)","한라IMS")
    comp2 = st.sidebar.text_input("회사2 (동적)","파나시아")
    st.sidebar.markdown("---")
    all_comps = list(FIXED_QUERIES)+[comp1,comp2]
    selected = [c for c in all_comps if st.sidebar.checkbox(c, True)]
    if st.sidebar.button("🔄 새로고침"):
        st.cache_data.clear()

    # 타이틀 & 메트릭
    st.title("에스엔시스 뉴스 보드")
    cols = st.columns(len(FIXED_QUERIES)+2)
    data_map={}
    for i,comp in enumerate(list(FIXED_QUERIES)+[comp1,comp2]):
        arts=[a for a in fetch_all(FIXED_QUERIES.get(comp,comp),mode,use_nv)
              if (dt:=parse_datetime(a.get("publishedAt","")))
                 and start_date<=dt.date()<=end_date]
        data_map[comp]=arts
        cols[i].metric(f"{comp} 기사 수", len(arts))

    st.markdown("---")

    # 1) 업체별 최신 뉴스
    tabs = st.tabs(list(data_map.keys()))
    for tab,comp in zip(tabs,data_map):
        with tab:
            st.subheader(f"{comp} 최신 뉴스 (상위 {cnt}건)")
            subset = sorted(
                data_map[comp],
                key=lambda x: parse_datetime(x["publishedAt"]) or datetime.min,
                reverse=True
            )[:cnt]
            if not subset:
                st.info("현재 조회할 기사가 없습니다.")
            for a in subset:
                ts = parse_datetime(a["publishedAt"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(
                    f"- [{_shorten(a['title'])}]({a['url']}) "
                    f"<span style='color:#6B7280;'>({ts_str})</span>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # 2) 제품별 최신 뉴스
    st.subheader("제품별 최신 뉴스")
    p_tabs = st.tabs([t for t,_ in PRODUCT_QUERIES])
    for tab,(title, syns) in zip(p_tabs, PRODUCT_QUERIES):
        with tab:
            st.subheader(f"{title} 최신 뉴스 (상위 {cnt}건)")
            q = " OR ".join(syns)
            arts = sorted(
                fetch_all(q,mode,use_nv),
                key=lambda x: parse_datetime(x.get("publishedAt","")) or datetime.min,
                reverse=True
            )[:cnt]
            if not arts:
                st.info("현재 조회할 기사가 없습니다.")
            for a in arts:
                ts = parse_datetime(a["publishedAt"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(
                    f"- [{_shorten(a['title'])}]({a['url']}) "
                    f"<span style='color:#6B7280;'>({ts_str})</span>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # 3) 기업별 키워드 (TF–IDF 전용)
    with st.expander("🔑 주요 키워드 분석 (상위 5개)", expanded=True):
        kcols = st.columns(len(data_map))
        for col, comp in zip(kcols, data_map):
            texts = [a["title"]+" "+a.get("content","") for a in data_map[comp][:cnt]]
            kws = extract_top_keywords(texts, top_n=5)
            col.markdown(f"**{comp}**")
            for w in kws:
                col.write(f"- {w}")

    # 4) 제품별 키워드 (TF–IDF 전용)
    with st.expander("🔑 제품별 주요 키워드 분석 (상위 5개)", expanded=False):
        pcols = st.columns(len(PRODUCT_QUERIES))
        for col,(title,syns) in zip(pcols, PRODUCT_QUERIES):
            q = " OR ".join(syns)
            arts = [
                a for a in fetch_all(q,mode,use_nv)
                if (dt:=parse_datetime(a.get("publishedAt","")))
                   and start_date<=dt.date()<=end_date
            ][:cnt]
            texts = [a["title"]+" "+a.get("content","") for a in arts]
            kws = extract_top_keywords(texts, top_n=5)
            col.markdown(f"**{title}**")
            for w in kws:
                col.write(f"- {w}")

    st.markdown("---")

    # 5) 노출 추이 차트
    st.subheader("노출 추이 분석")
    df = analyze_trends(
        sum(data_map.values(), []),
        {**{k:[k] for k in FIXED_QUERIES}, comp1:[comp1], comp2:[comp2]},
        start_date, end_date
    )
    df = df[df["company"].isin(selected)]
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("date_fmt:O", title="날짜"),
        y=alt.Y("count:Q", title="건수"),
        color=alt.Color("company:N", title="회사"),
        tooltip=["date_fmt","company","count"],
    ).properties(width="container", height=400)
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
