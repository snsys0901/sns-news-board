# -*- coding: utf-8 -*-
"""
에스엔시스 뉴스 대시보드 (파나시아 포함, 최종)
- Requests 기반 NewsAPI, Google RSS, Naver 크롤링
- TF–IDF 키워드 추출
- Streamlit 캐시 처리
- 스케줄러(메일 발송) 유지
"""

import os
import re
import json
import time
import hashlib
import logging
import threading
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
import schedule
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# 로깅 설정
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("NewsBoard")

# -----------------------------
# 전역 설정
# -----------------------------
CACHE_FILE = Path.home() / ".news_cache.json"
QUERIES = {
    "에스엔시스": "에스엔시스 OR S&SYS",
    "삼성중공업": "삼성중공업",
    "한화오션": "한화오션",
    "한라IMS": "한라IMS",
    "파나시아": "파나시아"
}
KEYWORDS = {
    "에스엔시스": ["에스엔시스", "s&sys"],
    "삼성중공업": ["삼성중공업"],
    "한화오션": ["한화오션"],
    "한라IMS": ["한라ims"],
    "파나시아": ["파나시아"]
}

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
SMTP_SERVER   = os.getenv("SMTP_SERVER",   "smtp.example.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT",  587))
SMTP_USER     = os.getenv("SMTP_USER",     "user@example.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "password")

# -----------------------------
# 유틸 함수
# -----------------------------
def _shorten(text: str, width: int = 60) -> str:
    return text if len(text) <= width else text[:width] + "…"

def parse_datetime_naive(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%SZ",
                "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except:
            continue
    return None

def extract_top_keywords(docs: List[str], top_n: int = 5) -> List[str]:
    docs = [d for d in docs if d.strip()]
    if not docs:
        return []
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=100)
    X = vect.fit_transform(docs)
    scores = X.sum(axis=0).A1
    terms  = vect.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:top_n]
    return [terms[i] for i in top_idx]

# -----------------------------
# 캐시 로드/업데이트
# -----------------------------
def _load_cache() -> Dict[str, Dict]:
    try:
        if CACHE_FILE.exists():
            return json.loads(CACHE_FILE.read_text("utf-8"))
    except:
        logger.exception("캐시 로드 실패, 초기화")
    return {}

def update_cache(arts: List[Dict]) -> None:
    cache = _load_cache()
    changed = False
    for a in arts:
        url = a.get("url","")
        if not url: continue
        uid = hashlib.sha256(url.encode()).hexdigest()
        if uid not in cache:
            cache[uid] = a; changed = True
        else:
            prev = cache[uid]
            orgs = prev.get("origins",[])
            new_orgs = list(dict.fromkeys(orgs + a.get("origins",[])))
            if new_orgs != orgs:
                cache[uid]["origins"] = new_orgs; changed = True
            new_pa = a.get("publishedAt")
            if new_pa and new_pa != prev.get("publishedAt"):
                cache[uid]["publishedAt"] = new_pa; changed = True
    if changed:
        try:
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
            logger.info("캐시 업데이트 완료")
        except:
            logger.exception("캐시 저장 실패")

# -----------------------------
# 동기식 뉴스 수집
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_newsapi_sync(q: str) -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    try:
        since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q":q, "language":"ko", "sortBy":"publishedAt",
            "from":since, "apiKey":NEWS_API_KEY, "pageSize":50
        }, timeout=15)
        r.raise_for_status()
        arts = r.json().get("articles",[])
        for a in arts:
            a.setdefault("origins",[]).append("newsapi")
            a.setdefault("source",{})["name"] = a.get("source",{}).get("name","NewsAPI")
        return arts
    except:
        logger.exception("NewsAPI 수집 오류")
        return []

@st.cache_data(ttl=3600)
def fetch_rss_sync(q: str) -> List[Dict]:
    out=[]; seen=set()
    for term in re.split(r"\s+OR\s+", q):
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(term)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            r = requests.get(url, timeout=15); r.raise_for_status()
            feed = rss_parse(r.text)
            for e in feed.entries:
                if e.link in seen: continue
                seen.add(e.link)
                dt = time.strftime("%Y-%m-%d %H:%M", e.published_parsed) if getattr(e,"published_parsed",None) else e.get("published","")
                out.append({"title":e.title,"url":e.link,"publishedAt":dt,"source":{"name":"Google RSS"},"origins":["rss"]})
        except:
            logger.warning(f"RSS 수집 실패: {url}")
    return out

@st.cache_data(ttl=3600)
def fetch_naver(q: str) -> List[Dict]:
    try:
        r = requests.get(
            f"https://search.naver.com/search.naver?where=news&query={requests.utils.quote(q)}&sort=1",
            headers={"User-Agent":"Mozilla/5.0"},
            timeout=15
        ); r.raise_for_status()
        soup=BeautifulSoup(r.text,"html.parser"); out=[]
        for li in soup.select("li.bx"):
            tag=li.select_one("a.news_tit")
            if not tag: continue
            raw = li.select_one("span.info").get_text(strip=True) if li.select_one("span.info") else ""
            out.append({"title":tag["title"].strip(),"url":tag["href"].strip(),
                        "publishedAt":raw,"source":{"name":"Naver"},"origins":["naver"]})
        return out
    except:
        logger.exception("Naver 수집 오류")
        return []

def fetch_news(q: str, mode: str, use_naver: bool) -> List[Dict]:
    arts=[]
    if mode!="RSS만": arts+=fetch_newsapi_sync(q)
    if mode!="NewsAPI만": arts+=fetch_rss_sync(q)
    if use_naver:      arts+=fetch_naver(q)
    update_cache(arts)
    return arts

# -----------------------------
# 이메일 기능
# -----------------------------
def generate_email_content(data: Dict[str,List[Dict]]) -> str:
    lines=[]
    for comp, arts in data.items():
        lines.append(f"## {comp}\n")
        for a in sorted(arts, key=lambda x: parse_datetime_naive(x["publishedAt"]) or datetime.min, reverse=True)[:10]:
            ts=parse_datetime_naive(a["publishedAt"])
            ts=ts.strftime("%Y-%m-%d %H:%M") if ts else "미확인"
            lines.append(f"- {a['title']} ({ts})\n  {a['url']}\n")
        lines.append("\n")
    return "".join(lines)

def send_email(subject: str, content: str, tos: List[str]) -> None:
    if not tos: return
    try:
        msg=MIMEText(content,_charset="utf-8")
        msg["Subject"]=subject; msg["From"]=SMTP_USER; msg["To"]=", ".join(tos)
        with smtplib.SMTP(SMTP_SERVER,SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER,SMTP_PASSWORD)
            smtp.sendmail(SMTP_USER,tos,msg.as_string())
        logger.info(f"이메일 발송: {tos}")
    except:
        logger.exception("이메일 발송 오류")

def send_daily_email() -> None:
    data={comp:fetch_news(q,mode="전체 (네이버 포함)",use_naver=True) for comp,q in QUERIES.items()}
    content=generate_email_content(data)
    tos=[st.session_state.get(f"email{i}") for i in range(1,6) if st.session_state.get(f"email{i}")]
    send_email("뉴스 보드 요약",content,tos)

# -----------------------------
# Streamlit UI
# -----------------------------
def render_articles(comp: str, arts: List[Dict], cnt: int) -> None:
    st.subheader(f"📰 {comp} 뉴스")
    if not arts:
        st.info("새로운 뉴스가 없습니다.")
        return
    for a in sorted(arts,key=lambda x: parse_datetime_naive(x["publishedAt"]) or datetime.min,reverse=True)[:cnt]:
        ts=parse_datetime_naive(a["publishedAt"]) or datetime.min
        ts=ts.strftime("%Y-%m-%d %H:%M")
        st.markdown(f"- [{_shorten(a['title'])}]({a['url']}) ({ts})")

def analyze_trends(src: str) -> None:
    st.markdown("---")
    st.header(f"📈 노출 이력 (31일) — {src}")
    cache=_load_cache()
    today=date.today()
    dates=[(today-timedelta(days=i)).strftime("%Y-%m-%d") for i in reversed(range(31))]
    cmap={d:{c:0 for c in KEYWORDS} for d in dates}
    for itm in cache.values():
        dt=parse_datetime_naive(itm["publishedAt"])
        if not dt: continue
        d0=dt.strftime("%Y-%m-%d")
        if d0 not in cmap: continue
        tl=itm["title"].lower()
        for comp,kws in KEYWORDS.items():
            if any(kw in tl for kw in kws):
                cmap[d0][comp]+=1
    rec=[{"date":d,"company":c,"count":cmap[d][c]} for d in dates for c in KEYWORDS]
    df=pd.DataFrame(rec)
    df["date_fmt"]=pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    chart=(alt.Chart(df).mark_bar()
           .encode(x=alt.X("date_fmt:N",title="날짜"),
                   y=alt.Y("count:Q",title="건수"),
                   color=alt.Color("company:N",title="회사"),
                   tooltip=["date_fmt","company","count"])
           .properties(width="container",height=300))
    st.altair_chart(chart,use_container_width=True)

def main() -> None:
    st.set_page_config(page_title="에스엔시스 뉴스 보드",layout="wide")
    st.title("📗 에스엔시스 뉴스 보드")

    # ─ 사이드바 ─
    src=st.sidebar.selectbox("뉴스 소스",
        ["전체 (네이버 포함)","전체 (네이버 제외)","RSS만","NewsAPI만"])
    use_nv="포함" in src
    cnt=st.sidebar.selectbox("표시 뉴스 건수",[5,10,12,15],index=1)

    st.sidebar.header("📧 메일 설정")
    t=st.sidebar.selectbox("메일 발송 시간",["08:00","10:00","14:00","16:00"],key="send_time")
    for i in range(1,6):
        st.sidebar.text_input(f"이메일 {i}",key=f"email{i}")
    if st.sidebar.button("메일 스케줄 저장"):
        schedule.clear("emailjob")
        schedule.every().day.at(t).do(send_daily_email).tag("emailjob")
        st.sidebar.success("저장되었습니다.")

    st.sidebar.header("노출 이력 설정")
    ts=st.sidebar.selectbox("소스 선택",["전체","네이버만","구글 RSS만"],index=0)

    # 스케줄러
    if "sched" not in st.session_state:
        def _run():
            while True:
                schedule.run_pending(); time.sleep(1)
        threading.Thread(target=_run,daemon=True).start()
        st.session_state["sched"]=True

    if st.sidebar.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()

    # ─ 탭별 뉴스 ─
    tabs=st.tabs(list(QUERIES.keys()))
    for tab,comp in zip(tabs,QUERIES):
        with tab:
            arts=fetch_news(QUERIES[comp],src,use_nv)
            render_articles(comp,arts,cnt)

    # ─ 키워드 분석 ─
    st.markdown("---"); st.header("🔑 주요 키워드 분석")
    cols=st.columns(len(QUERIES))
    for col,comp in zip(cols,QUERIES):
        with col:
            arts=fetch_news(QUERIES[comp],src,use_nv)
            titles=[a["title"] for a in sorted(arts,key=lambda x: parse_datetime_naive(x["publishedAt"]) or datetime.min,reverse=True)[:cnt]]
            kws=extract_top_keywords(titles,5)
            if kws:
                for kw in kws: st.markdown(f"- {kw}")
            else:
                st.write("키워드 없음")

    # ─ 노출 이력 ─
    analyze_trends(ts)

if __name__ == "__main__":
    main()
