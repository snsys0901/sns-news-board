# -*- coding: utf-8 -*-
"""
에스엔시스 뉴스 대시보드  (v2.7.7, 2025-06-26)
────────────────────────────────────────────────────────
• UX 최적화: 카드 레이아웃, 멀티컬럼, 헤더에 로고 백그라운드 방식 삽입
• 모듈화: 렌더링 함수 분리
• 사이드바 기간·소스·건수·동적 필터 강화
• 버그 수정: 캐시 업데이트 Windows 권한 문제 해결, 차트 선택 빈 리스트 처리, 제품 탭 렌더링 방식 수정
• 기능 추가: 화면 상단 헤더 배경에 로고 삽입 (왼쪽)
• 기능 추가: 헤더 하단에 푸른색 그라데이션 테두리 적용
"""

import os
import warnings
import logging
import re
import json
import time
import hashlib
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
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

# ── 환경 설정 ─────────────────────────────────────────────
os.environ["STREAMLIT_SUPPRESS_NO_SCRIPT_RUN_CONTEXT_WARNING"] = "true"
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NewsBoard")

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(page_title="에스엔시스 뉴스 보드", layout="wide")
st.markdown(
    """<script>try{localStorage.setItem('theme','light');}catch(e){};</script>""",
    unsafe_allow_html=True,
)

# ── 헤더 로고 배경 삽입 및 그라데이션 테두리 적용 ─────────────────────────────────────
LOGO_URL = "https://pds.saramin.co.kr/company/logo/201903/20/pon8bu_ngqk-ya4cjo_logo.jpg"
st.markdown(
    f"""
    <style>
    [data-testid="stHeader"] {{
        /* 배경은 순수 흰색 + 로고 이미지 */
        background-color: #ffffff;
        background-image: url('{LOGO_URL}');
        background-repeat: no-repeat;
        background-position: left center;
        background-size: auto 60px;
        padding-left: 100px;

        /* 그라데이션 테두리 */
        border-bottom: 6px solid transparent;
        border-image-source: linear-gradient(to right, #1E90FF, #00BFFF);
        border-image-slice: 1;
    }}
    [data-testid="stHeader"] h1 {{
        /* 흰 바탕 대비 잘 보이도록 다크 그레이로 변경 */
        color: #333 !important;
        margin-left: 100px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ── 전역 스타일 정의 ─────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background: #F8F9FA; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #E1E4E8; }
    .stMetric { background: #fff; border: 1px solid #E1E4E8; border-radius: 8px; padding: 1rem; }
    .news-card { background: #fff; border: 1px solid #E1E4E8; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .news-card a { color: #2C3E50; text-decoration: none; font-weight: bold; }
    .news-card .timestamp { color: #6B7280; font-size: 0.8rem; margin-top: 0.5rem; }
    .news-card .desc { margin-top: 0.5rem; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 상수 정의 ─────────────────────────────────────────────
CACHE_FILE = Path.home() / ".news_cache.json"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

FIXED_QUERIES = {
    "에스엔시스": "에스엔시스 OR S&SYS",
    "삼성중공업": "삼성중공업",
    "한화오션": "한화오션",
}
STATIC_PRODUCTS = [
    ("BWMS", ["BWMS", "BWTS", "선박평형수"]),
    ("IAS", ["IAS","통합자동화시스템","Integrated Automation System","선박용 제어시스템","선박 제어시스템","콩스버그","선박용 IAS"]),
    ("FGSS", ["FGSS","LFSS","선박이중연료시스템"]),
]
DEFAULT_P1_SYNS = ["배전반", "수배전반", "전력배전반", "전력기기"]
DEFAULT_P2_SYNS = ["친환경", "친환경 선박", "그린십", "탈탄소", "저탄소 선박"]
NOISE_WORDS = {"null","뉴스","기사","사진","최근", *FIXED_QUERIES}

# ── 유틸 함수 ─────────────────────────────────────────────
def _shorten(text: str, width: int = 80) -> str:
    return text if len(text) <= width else text[:width] + "…"

def parse_datetime(s: str) -> Optional[datetime]:
    if not s: return None
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    for pat, unit in [(r"(\d+)분 전", "minutes"), (r"(\d+)시간 전", "hours")]:
        m = re.match(pat, s)
        if m: return now - timedelta(**{unit: int(m.group(1))})
    if s in ("오늘", "today"): return now
    if s in ("어제", "yesterday"): return now - timedelta(days=1)
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

def extract_top_keywords(docs: List[str], top_n: int, extra_stop: set, fallback: str) -> List[str]:
    texts = [clean_text(d) for d in docs if d.strip()]
    stop = NOISE_WORDS | extra_stop
    if not texts: return [fallback]
    vect = TfidfVectorizer(token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b", ngram_range=(1,3), max_features=500)
    try:
        X = vect.fit_transform(texts)
    except ValueError:
        return [fallback]
    scores = X.sum(axis=0).A1; terms = vect.get_feature_names_out()
    ranked = sorted(zip(scores, terms), reverse=True)
    clean_terms = [re.sub(r"(으로|로|와|과|이|가|은|는|도|을|를)$", "", t) for _, t in ranked]
    filtered = [w for w in clean_terms if w.lower() not in stop and not w.isdigit()]
    return [w for w in filtered if w.lower() != fallback.lower()][:top_n] or [fallback]

def dedup(items: List[Dict]) -> List[Dict]:
    seen, unique = set(), []
    for a in items:
        u = a.get("url")
        if u and u not in seen:
            unique.append(a); seen.add(u)
    return unique
# ── 데이터 수집 ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_newsapi(q: str) -> List[Dict]:
    arts=[]
    if NEWS_API_KEY:
        since=(datetime.utcnow()-timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params={"q":q,"language":"ko","sortBy":"publishedAt","from":since,"apiKey":NEWS_API_KEY,"pageSize":100}
        try:
            r=requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            r.raise_for_status()
            for a in r.json().get("articles", []):
                a.setdefault("origins",[]).append("newsapi")
                a["content"]=a.get("content", "") or ""
                arts.append(a)
        except Exception:
            logger.exception("NewsAPI 오류")
    return arts

@st.cache_data(ttl=3600)
def fetch_rss(q: str) -> List[Dict]:
    out, seen = [], set()
    for term in q.split(" OR "):
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(term)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            r=requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            feed=rss_parse(r.text)
            for e in feed.entries:
                if e.link in seen: continue
                seen.add(e.link)
                summary=BeautifulSoup(e.get("summary",""),"html.parser").get_text()
                dt=time.strftime("%Y-%m-%d %H:%M", e.published_parsed) if getattr(e,"published_parsed", None) else ""
                out.append({"title":e.title,"url":e.link,"publishedAt":dt,"content":summary,"origins":["rss"]})
        except Exception:
            logger.warning(f"RSS 오류: {url}")
    return out

@st.cache_data(ttl=3600)
def fetch_naver(q: str) -> List[Dict]:
    out=[]
    try:
        url=f"https://search.naver.com/search.naver?where=news&query={requests.utils.quote(q)}&sort=1"
        r=requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception:
        logger.exception("Naver 스크래핑 오류")
        return out
    soup=BeautifulSoup(r.text,"html.parser")
    for it in soup.select("li.bx") + soup.select("div.news_area"):
        a_tag=it.select_one("a.news_tit")
        if not a_tag: continue
        title=a_tag.get("title") or a_tag.get_text(strip=True)
        link=a_tag["href"]
        dt_tag=it.select_one("span.date") or it.select_one("span.info")
        dt=dt_tag.get_text(strip=True) if dt_tag else ""
        desc=it.select_one("a.api_txt_lines") or it.select_one("div.news_dsc")
        content=desc.get_text(strip=True) if desc else ""
        out.append({"title":title,"url":link,"publishedAt":dt,"content":content,"origins":["naver"]})
    return out


def fetch_all(q: str, mode: str, use_nv: bool) -> List[Dict]:
    funcs=[]
    if mode != "RSS만": funcs.append(fetch_newsapi)
    if mode != "NewsAPI만": funcs.append(fetch_rss)
    if use_nv: funcs.append(fetch_naver)
    arts=[]
    with ThreadPoolExecutor() as ex:
        for fut in [ex.submit(fn, q) for fn in funcs]:
            try: arts.extend(fut.result())
            except Exception as e: logger.warning(f"fetch error: {e}")
    arts=dedup(arts)
    update_cache(arts)
    return arts


def update_cache(arts: List[Dict]) -> None:
    cache={}
    if CACHE_FILE.exists():
        try: cache=json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except:
            try: CACHE_FILE.unlink(missing_ok=True)
            except: pass
            cache={}
    changed=False
    for a in arts:
        url=a.get("url","")
        if not url: continue
        uid=hashlib.sha256(url.encode()).hexdigest()
        if uid not in cache: cache[uid]=a; changed=True
    purge_before=datetime.now(ZoneInfo("Asia/Seoul"))-timedelta(days=30)
    for uid,a in list(cache.items()):
        dt=parse_datetime(a.get("publishedAt",""))
        if dt and dt<purge_before: del cache[uid]; changed=True
    if changed:
        try:
            tmp=tempfile.NamedTemporaryFile("w", delete=False, dir=CACHE_FILE.parent, encoding="utf-8")
            json.dump(cache, tmp, ensure_ascii=False, indent=2)
            tmp.flush(); tmp.close()
            if CACHE_FILE.exists(): CACHE_FILE.unlink()
            os.rename(tmp.name, CACHE_FILE)
        except Exception as e:
            logger.warning(f"Cache update failed: {e}")


def analyze_trends(arts: List[Dict], kw_map: Dict[str, List[str]], start: date, end: date) -> pd.DataFrame:
    dates=pd.date_range(start, end)
    cmap={d.strftime("%Y-%m-%d"): {k: 0 for k in kw_map} for d in dates}
    for a in arts:
        dt=parse_datetime(a.get("publishedAt",""))
        if not dt: continue
        day=dt.strftime("%Y-%m-%d")
        if day not in cmap: continue
        txt=(a.get("title","")+" "+a.get("content","")).lower()
        for comp,kws in kw_map.items():
            if any(kw.lower() in txt for kw in kws): cmap[day][comp]+=1
    rows=[{"date":d,"company":c,"count":n} for d,v in cmap.items() for c,n in v.items()]
    df=pd.DataFrame(rows)
    df["date_fmt"]=pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    return df


def display_article_cards(articles: List[Dict], cols_per_row: int = 3):
    for i in range(0, len(articles), cols_per_row):
        batch=articles[i:i+cols_per_row]
        cols=st.columns(len(batch))
        for col,art in zip(cols,batch):
            ts=parse_datetime(art.get("publishedAt",""))
            ts_str=ts.strftime("%Y-%m-%d %H:%M") if ts else ""
            desc=(art.get("content","")[:100]+"…") if art.get("content") else ""
            html=(
                f"<div class='news-card'>"
                f"<a href=\"{art['url']}\" target=\"_blank\">{_shorten(art['title'])}</a>"
                f"<div class='timestamp'>{ts_str}</div>"
                f"<div class='desc'>{desc}</div>"
                "</div>"
            )
            with col: st.markdown(html, unsafe_allow_html=True)


def main():
    if st.sidebar.button("🔄 새로고침"): st.cache_data.clear()
    st.sidebar.header("필터 설정")
    mode=st.sidebar.selectbox("뉴스 소스", ("전체 (네이버 포함)", "전체 (네이버 제외)", "RSS만", "NewsAPI만"), 0)
    use_nv="포함" in mode
    cnt=st.sidebar.slider("기사 표시 건수", 5, 30, 10, 5)
    today=date.today()
    dates=st.sidebar.date_input("분석 기간", (today-timedelta(days=30), today))
    if isinstance(dates, tuple): start_date,end_date=dates
    else: start_date=end_date=dates
    if start_date>end_date: st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
    comp1=st.sidebar.text_input("회사1 (동적)", "한라IMS")
    comp2=st.sidebar.text_input("회사2 (동적)", "파나시아")
    prod1=st.sidebar.text_input("제품1 (동적)", "배전반")
    prod2=st.sidebar.text_input("제품2 (동적)", "친환경")

    st.title("에스엔시스 뉴스 보드")
    comp_list=list(FIXED_QUERIES)+[comp1,comp2]
    cols_metrics=st.columns(len(comp_list))
    data_map={}
    with st.spinner("뉴스 수집 중…"):
        for i,comp in enumerate(comp_list):
            arts=[a for a in fetch_all(FIXED_QUERIES.get(comp,comp),mode,use_nv)
                  if (dt:=parse_datetime(a.get("publishedAt",""))) and start_date<=dt.date()<=end_date]
            data_map[comp]=arts
            cols_metrics[i].metric(f"{comp} 기사 수", len(arts))

    st.markdown("---")
    for tab,comp in zip(st.tabs(list(data_map.keys())), data_map):
        with tab:
            st.subheader(f"{comp} 최신 뉴스 (상위 {cnt}건)")
            subset=sorted(data_map[comp], key=lambda x: parse_datetime(x.get("publishedAt","")) or datetime.min, reverse=True)[:cnt]
            if not subset: st.info("현재 조회할 기사가 없습니다.")
            else: display_article_cards(subset)

    st.markdown("---")
    st.subheader("제품별 최신 뉴스")
    prod_tabs = st.tabs([t for t,_ in STATIC_PRODUCTS] + [prod1 or "제품1", prod2 or "제품2"] )
    prod_queries = STATIC_PRODUCTS + [
        (prod1 or "제품1", DEFAULT_P1_SYNS if prod1=="배전반" else [prod1]),
        (prod2 or "제품2", DEFAULT_P2_SYNS if prod2=="친환경" else [prod2]),
    ]
    for tab, (title, syns) in zip(prod_tabs, prod_queries):
        with tab:
            st.subheader(f"{title} 최신 뉴스 (상위 {cnt}건)")
            arts=sorted(fetch_all(" OR ".join(syns),mode,use_nv),
                        key=lambda x: parse_datetime(x.get("publishedAt","")) or datetime.min, reverse=True)[:cnt]
            if not arts: st.info("현재 조회할 기사가 없습니다.")
            else: display_article_cards(arts)

    st.markdown("---")
    with st.expander("🔑 주요 키워드 분석 (회사별 상위 5개)", True):
        cols_kw=st.columns(len(data_map))
        for col, comp in zip(cols_kw, data_map):
            texts=[a["title"]+a.get("content","") for a in data_map[comp][:cnt*3]]
            kws=extract_top_keywords(texts,5,{comp.lower()},comp)
            col.markdown(f"**{comp}**")
            for w in kws: col.write(f"- {w}")

    st.markdown("---")
    with st.expander("🔑 제품별 주요 키워드 분석 (상위 5개)", False):
        cols_kwp=st.columns(len(prod_queries))
        for col, (title, syns) in zip(cols_kwp, prod_queries):
            texts=[a["title"]+a.get("content","") for a in sorted(fetch_all(" OR ".join(syns),mode,use_nv),
                        key=lambda x: parse_datetime(x.get("publishedAt","")) or datetime.min)][:cnt*3]
            kws=extract_top_keywords(texts,5,{s.lower() for s in syns},title)
            col.markdown(f"**{title}**")
            for w in kws: col.write(f"- {w}")

    st.markdown("---")
    trend_df=analyze_trends(sum(data_map.values(),[]), {**{k:[k] for k in FIXED_QUERIES}, comp1:[comp1], comp2:[comp2]}, start_date, end_date)
    selected=[c for c in comp_list if st.sidebar.checkbox(c, True, key=f"cb_{c}")]
    if not selected: selected=comp_list
    chart=(
        alt.Chart(trend_df[trend_df["company"].isin(selected)])
        .mark_line(point=True)
        .encode(
            x=alt.X("date_fmt:O", title="날짜"),
            y=alt.Y("count:Q", title="건수"),
            color=alt.Color("company:N", title="회사"),
            tooltip=["date_fmt","company","count"],
        )
        .properties(width="container", height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

if __name__=="__main__": main()
