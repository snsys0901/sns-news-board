# -*- coding: utf-8 -*-
"""
📗 에스엔시스 뉴스 대시보드
- 제목+내용 기반 전처리 키워드 강화: TF–IDF + 빈도 백업
- 네이버 스크랩(li.bx + div.news_area) + RSS/NewsAPI 지원
- parse_datetime 네이밍 통일 및 위치 수정
- fetch_naver: 검색어가 제목 또는 본문에 포함된 기사만 반환하도록 필터링
"""

import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

import requests
import pandas as pd
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup
from feedparser import parse as rss_parse
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# 로깅 설정
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NewsBoard")

# -----------------------------
# 전역 설정
# -----------------------------
CACHE_FILE    = Path.home() / ".news_cache.json"
FIXED_QUERIES = {
    "에스엔시스": "에스엔시스 OR S&SYS",
    "삼성중공업": "삼성중공업",
    "한화오션": "한화오션",
}
FIXED_KEYWORDS = {k: [k] for k in FIXED_QUERIES}
NOISE_WORDS   = {"rss", "news", "google", "https", "http", "com", "href", "color", "nbsp"}

NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
SMTP_SERVER    = os.getenv("SMTP_SERVER", "smtp.example.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 587))
SMTP_USER      = os.getenv("SMTP_USER", "user@example.com")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "password")

# -----------------------------
# 유틸 함수
# -----------------------------
def _shorten(text: str, width: int = 60) -> str:
    return text if len(text) <= width else text[:width] + "…"

def parse_datetime(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
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
    """
    TF–IDF와 빈도 기반으로 상위 키워드 추출
    - 불필요한 접속사(JOSA) 제거
    - 고정 키워드 및 노이즈 단어 제외
    - 회사명 포함 토큰 배제
    """
    texts = [clean_text(d).lower() for d in docs if d.strip()]
    if not texts:
        return []

    # 1) TF–IDF 후보
    try:
        vect = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100,
            token_pattern=r"(?u)\b[가-힣A-Za-z]{2,}\b"
        )
        X = vect.fit_transform(texts)
        scores = X.sum(axis=0).A1
        terms = vect.get_feature_names_out()
        tfidf_candidates = [terms[i] for i in scores.argsort()[::-1][:top_n*2]]
    except Exception:
        logger.warning("TF–IDF 처리 실패, 빈도 기반으로 전환")
        tfidf_candidates = []

    # 2) 빈도 기반 후보
    words = []
    for t in texts:
        words += re.findall(r"(?u)\b[가-힣A-Za-z]{2,}\b", t)
    freq_candidates = [w for w, _ in Counter(words).most_common(top_n*2)]

    # 접속사(JOSA) 제거용 패턴
    josa_pattern = re.compile(r"(으로|로|와|과|은|는|이|가|도)$")
    excludes = NOISE_WORDS | {k.lower() for k in FIXED_QUERIES}

    # 3) 후보 필터링 및 정제
    result = []
    for w in tfidf_candidates + freq_candidates:
        w_clean = josa_pattern.sub("", w).strip()
        if len(w_clean) < 2:
            continue
        lw = w_clean.lower()
        if lw in excludes or any(fq.lower() in lw for fq in FIXED_QUERIES):
            continue
        if lw in result:
            continue
        result.append(w_clean)
        if len(result) == top_n:
            break

    return result

# -----------------------------
# 캐시 관리
# -----------------------------
def _load_cache() -> Dict[str, Dict]:
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text("utf-8"))
    except Exception:
        logger.exception("캐시 로드 실패")
    return {}

def update_cache(articles: List[Dict]) -> None:
    cache = _load_cache()
    changed = False
    for a in articles:
        url = a.get("url", "")
        if not url:
            continue
        uid = hashlib.sha256(url.encode()).hexdigest()
        if uid not in cache:
            cache[uid] = a
            changed = True
        else:
            prev = cache[uid]
            origins = list(dict.fromkeys(prev.get("origins", []) + a.get("origins", [])))
            if origins != prev.get("origins", []):
                cache[uid]["origins"] = origins
                changed = True
            pa = a.get("publishedAt")
            if pa and pa != prev.get("publishedAt"):
                cache[uid]["publishedAt"] = pa
                changed = True
    if changed:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
        logger.info("캐시 업데이트 완료")

# -----------------------------
# 뉴스 수집
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_newsapi(q: str) -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q": q, "language": "ko", "sortBy": "publishedAt",
        "from": since, "apiKey": NEWS_API_KEY, "pageSize": 50,
    }
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
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            feed = rss_parse(r.text)
            for e in feed.entries:
                if e.link in seen: continue
                seen.add(e.link)
                content = BeautifulSoup(e.get("summary", ""), "html.parser").get_text()
                dt = (time.strftime("%Y-%m-%d %H:%M", e.published_parsed)
                      if hasattr(e, "published_parsed") else e.get("published", ""))
                out.append({
                    "title": e.title, "url": e.link,
                    "publishedAt": dt, "content": content,
                    "origins": ["rss"],
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
            a = it.select_one("a.news_tit")
            if not a: continue
            title = a.get("title") or a.get_text(strip=True)
            link  = a["href"]
            dt_tag = it.select_one("span.date") or it.select_one("span.info")
            dt     = dt_tag.get_text(strip=True) if dt_tag else ""
            desc   = it.select_one("a.api_txt_lines") or it.select_one("div.news_dsc")
            content= desc.get_text(strip=True) if desc else ""
            out.append({
                "title":title, "url":link,
                "publishedAt":dt, "content":content,
                "origins":["naver"],
            })
    except Exception:
        logger.exception("Naver 스크래핑 오류")

    # ——————————————————————————
    # 본문 또는 제목에 검색어 포함된 기사만 반환
    # ——————————————————————————
    terms = [t.strip().lower() for t in re.split(r"\s+OR\s+", q) if t.strip()]
    filtered = []
    for art in out:
        combined = (art["title"] + " " + art["content"]).lower()
        if any(term in combined for term in terms):
            filtered.append(art)
    return filtered

def fetch_all(q: str, mode: str, use_nv: bool) -> List[Dict]:
    arts = []
    if mode != "RSS만":     arts += fetch_newsapi(q)
    if mode != "NewsAPI만": arts += fetch_rss(q)
    if use_nv:              arts += fetch_naver(q)
    update_cache(arts)
    return arts

# -----------------------------
# 노출 추이 분석
# -----------------------------
def analyze_trends(arts: List[Dict], kw_map: Dict[str,List[str]]) -> pd.DataFrame:
    dates = [(date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(30, -1, -1)]
    cmap  = {d:{c:0 for c in kw_map} for d in dates}
    for itm in arts:
        dt = parse_datetime(itm.get("publishedAt",""))
        if not dt: continue
        d0 = dt.strftime("%Y-%m-%d")
        if d0 not in cmap: continue
        txt = (itm.get("title","")+" "+itm.get("content","")).lower()
        for comp,kws in kw_map.items():
            if any(kw.lower() in txt for kw in kws):
                cmap[d0][comp]+=1
    recs=[]
    for d in dates:
        for c in kw_map:
            recs.append({"date":d,"company":c,"count":cmap[d][c]})
    df=pd.DataFrame(recs)
    df["date_fmt"]=pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="뉴스 대시보드", layout="wide")
    st.title("📗 에스엔시스 뉴스 보드")

    mode   = st.sidebar.selectbox("뉴스 소스",
             ["전체 (네이버 포함)","전체 (네이버 제외)","RSS만","NewsAPI만"])
    use_nv = "포함" in mode
    cnt    = st.sidebar.selectbox("표시 건수",[5,10,15],index=1)

    st.sidebar.markdown("---")
    comp1 = st.sidebar.text_input("회사1 (동적)","한라IMS")
    comp2 = st.sidebar.text_input("회사2 (동적)","파나시아")

    st.sidebar.markdown("---")
    st.sidebar.write("**노출 추이에 포함할 회사**")
    all_comps = list(FIXED_QUERIES)+[comp1,comp2]
    selected  = [c for c in all_comps if st.sidebar.checkbox(c,True)]

    if st.sidebar.button("🔄 새로고침"):
        st.cache_data.clear()
        # 자동 rerun 지원

    queries = {**FIXED_QUERIES, comp1:comp1, comp2:comp2}
    kw_map  = {**FIXED_KEYWORDS, comp1:[comp1], comp2:[comp2]}
    tabs    = st.tabs(list(queries.keys()))
    all_data:Dict[str,List[Dict]]={}

    for tab,comp in zip(tabs,queries):
        with tab:
            arts = fetch_all(queries[comp],mode,use_nv)
            all_data[comp]=arts
            st.subheader(f"📰 {comp} 뉴스")
            if not arts:
                st.info("기사 없음")
            for a in sorted(arts,
                            key=lambda x: parse_datetime(x["publishedAt"]) or datetime.min,
                            reverse=True)[:cnt]:
                ts=parse_datetime(a["publishedAt"])
                ts=ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                st.markdown(f"- [{_shorten(a['title'])}]({a['url']}) ({ts})")

    st.markdown("---")
    st.header("🔑 주요 키워드 분석 (상위 5개)")
    cols = st.columns(len(queries))
    for col,comp in zip(cols,queries):
        with col:
            texts=[a["title"]+" "+a.get("content","") for a in all_data[comp][:cnt]]
            kws=extract_top_keywords(texts,top_n=5)
            if kws:
                for w in kws: st.write(f"- {w}")
            else:
                st.write("키워드 없음")

    st.markdown("---")
    st.header("📈 노출 추이 분석")
    merged   = sum(all_data.values(),[])
    df_trend = analyze_trends(merged,kw_map)
    df_trend = df_trend[df_trend["company"].isin(selected)]

    chart=(
        alt.Chart(df_trend)
           .mark_bar()
           .encode(
               x=alt.X("date_fmt:N",title="날짜"),
               y=alt.Y("count:Q",title="건수"),
               color=alt.Color("company:N",title="회사"),
               tooltip=["date_fmt","company","count"],
           )
           .properties(width="container",height=350)
    )
    st.altair_chart(chart,use_container_width=True)

if __name__=="__main__":
    main()
