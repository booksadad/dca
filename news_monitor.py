import yfinance as yf
import google.generativeai as genai
import json
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("QuantHQ_News")

class NewsMonitor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
            # Use gemini-3.1-flash-lite as requested by the user
            self.model = genai.GenerativeModel('gemini-3.1-flash-lite', generation_config={"response_mime_type": "application/json"})
        else:
            self.model = None

    def fetch_latest_news(self, ticker: str, hours_back: int = 24):
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news
            if not news_items:
                return []
            
            recent_news = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            for item in news_items:
                if 'providerPublishTime' in item:
                    pub_time = datetime.fromtimestamp(item['providerPublishTime'], timezone.utc)
                elif 'pubDate' in item:
                    try:
                        pub_time = datetime.fromisoformat(item['pubDate'].replace('Z', '+00:00'))
                    except:
                        pub_time = datetime.now(timezone.utc)
                else:
                    pub_time = datetime.now(timezone.utc)

                if pub_time >= cutoff_time:
                    content = item.get('content', {})
                    recent_news.append({
                        'title': content.get('title', item.get('title', '')),
                        'summary': content.get('summary', item.get('summary', '')),
                        'link': item.get('link', content.get('clickThroughUrl', {}).get('url', '')),
                        'time': pub_time
                    })
            
            return recent_news
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def analyze_sentiment(self, ticker: str, news_title: str, news_summary: str):
        if not self.model:
            return None

        prompt = f"""
        You are a financial AI analyzing news for an institutional quant fund.
        Stock Ticker: {ticker}
        News Title: {news_title}
        News Summary: {news_summary}

        CRITICAL INSTRUCTIONS:
        1. Analyze if this news is Bullish, Bearish, or Neutral for {ticker}.
        2. Give an impact score from 1 to 10 (10 being market-moving).
        3. Write a brief explanation (2-3 sentences max) in THAI language.
        
        Return ONLY valid JSON in the exact format:
        {{
            "sentiment": "Bullish" | "Bearish" | "Neutral",
            "impact_score": int,
            "thai_summary": "string"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"AI Sentiment Analysis failed for {ticker}: {e}")
            return None

    def scan_portfolio(self, tickers: list[str], hours_back: int = 24):
        results = []
        for t in tickers:
            news_items = self.fetch_latest_news(t, hours_back)
            for n in news_items:
                analysis = self.analyze_sentiment(t, n['title'], n['summary'])
                if analysis:
                    results.append({
                        'ticker': t,
                        'title': n['title'],
                        'link': n['link'],
                        'sentiment': analysis.get('sentiment', 'Neutral'),
                        'impact_score': analysis.get('impact_score', 0),
                        'thai_summary': analysis.get('thai_summary', '')
                    })
        # Sort by impact score descending
        results.sort(key=lambda x: x['impact_score'], reverse=True)
        return results
