import streamlit.components.v1 as components


def render_tradingview_chart(tv_symbol, key, height=480):
    container_id = f"tv_{key}_{height}"
    tradingview_html = f"""
    <div class="tradingview-widget-container">
      <div id="{container_id}" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      (function() {{
        const target = document.getElementById('{container_id}');
        if (!target) return;
        target.innerHTML = '';
        new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{tv_symbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "th",
          "toolbar_bg": "#131722",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "container_id": "{container_id}"
        }});
      }})();
      </script>
    </div>
    """
    components.html(tradingview_html, height=height + 20)
