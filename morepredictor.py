from __future__ import annotations

from flask import Flask, render_template, request
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass
from datetime import datetime, timedelta, date
import time
import re
from typing import Optional, Dict, Any, Tuple

app = Flask(__name__)

# =============================
# HTTP session with retries
# =============================
def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            # Use a non-placeholder contact if you can (helps with Nominatim policy).
            "User-Agent": "MorelMushroomPredictor/1.0 (contact: you@example.com)",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return s

HTTP = build_session()

# =============================
# Tiny in-memory cache
# =============================
@dataclass
class CacheItem:
    value: Any
    expires_at: float

_CACHE: Dict[str, CacheItem] = {}

def cache_get(key: str) -> Any:
    item = _CACHE.get(key)
    if not item:
        return None
    if time.time() > item.expires_at:
        _CACHE.pop(key, None)
        return None
    return item.value

def cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    _CACHE[key] = CacheItem(value=value, expires_at=time.time() + ttl_seconds)

# =============================
# Helpers
# =============================
ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")
STREET_HINT_RE = re.compile(
    r"\b(\d{1,6})\b.*\b(st|street|rd|road|ave|avenue|blvd|boulevard|ln|lane|dr|drive|hwy|highway|ct|court|trl|trail|pkwy|parkway|cir|circle|way)\b",
    re.IGNORECASE,
)

def clean_location(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def safe_json(r: requests.Response) -> Optional[dict]:
    try:
        return r.json()
    except Exception:
        return None

def safe_average(values) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)

def estimate_soil_temp(avg_day: float, avg_night: float) -> float:
    return ((avg_day + avg_night) / 2.0) - 5.0

def is_probably_street_address(q: str) -> bool:
    q = q.strip()
    if STREET_HINT_RE.search(q):
        return True
    if re.search(r"\b\d{1,6}\b", q) and ("," in q):
        return True
    return False

# =============================
# Geocoding (ZIP/city + full address)
# =============================
def geocode_open_meteo(query: str) -> Optional[Tuple[float, float]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": query,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    if ZIP_RE.match(query):
        params["countryCode"] = "US"

    r = HTTP.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return None
    data = safe_json(r) or {}
    results = data.get("results") or []
    if not results:
        return None
    best = max(results, key=lambda x: x.get("population") or 0)
    return float(best["latitude"]), float(best["longitude"])

def geocode_nominatim(query: str) -> Optional[Tuple[float, float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}

    headers = dict(HTTP.headers)
    headers["Referer"] = "https://example.com"

    r = HTTP.get(url, params=params, headers=headers, timeout=12)
    if r.status_code != 200:
        return None
    data = safe_json(r)
    if not data:
        return None
    if isinstance(data, list) and data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None

def get_coordinates(user_input: str) -> Optional[Tuple[float, float]]:
    q = clean_location(user_input)
    if not q or len(q) < 2:
        return None

    cache_key = f"geo:{q.lower()}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    coords = None

    # Street address → Nominatim first
    if is_probably_street_address(q):
        coords = geocode_nominatim(q) or geocode_open_meteo(q)
    else:
        # City/ZIP → Open-Meteo first
        coords = geocode_open_meteo(q) or geocode_nominatim(q)

    if coords:
        cache_set(cache_key, coords, ttl_seconds=60 * 60 * 24 * 14)
    return coords

# =============================
# Weather (NEVER FAILS)
# =============================
def weather_from_archive(lat: float, lon: float, start: date, end: date, include_soil: bool = True) -> Optional[Dict[str, float]]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    daily_fields = "temperature_2m_max,temperature_2m_min,rain_sum"
    if include_soil:
        daily_fields += ",soil_temperature_0_to_10cm_mean"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": daily_fields,
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }

    r = HTTP.get(url, params=params, timeout=14)
    if r.status_code != 200:
        return None

    data = safe_json(r) or {}
    daily = data.get("daily") or {}

    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    rain = daily.get("rain_sum") or []

    avg_day = safe_average(tmax)
    avg_night = safe_average(tmin)
    total_rain = float(sum([x for x in rain if x is not None])) if rain else 0.0

    soil_vals = daily.get("soil_temperature_0_to_10cm_mean") or []
    avg_soil = safe_average(soil_vals)

    if avg_day is None or avg_night is None:
        return None

    if avg_soil is None:
        avg_soil = estimate_soil_temp(avg_day, avg_night)

    return {
        "avg_day_temp": float(avg_day),
        "avg_night_temp": float(avg_night),
        "avg_soil_temp": float(avg_soil),
        "total_rain": float(total_rain),
        "_source": "archive" if include_soil else "archive_no_soil",
    }

def weather_from_forecast(lat: float, lon: float, include_soil: bool = True) -> Optional[Dict[str, float]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "past_days": 7,
    }
    if include_soil:
        params["hourly"] = "soil_temperature_0_to_10cm"

    r = HTTP.get(url, params=params, timeout=14)
    if r.status_code != 200:
        return None

    data = safe_json(r) or {}
    daily = data.get("daily") or {}
    hourly = data.get("hourly") or {}

    avg_day = safe_average(daily.get("temperature_2m_max") or [])
    avg_night = safe_average(daily.get("temperature_2m_min") or [])
    total_rain = float(sum([x for x in (daily.get("precipitation_sum") or []) if x is not None]))

    if avg_day is None or avg_night is None:
        return None

    avg_soil = None
    if include_soil:
        soil_hourly = hourly.get("soil_temperature_0_to_10cm") or []
        avg_soil = safe_average(soil_hourly)

    if avg_soil is None:
        avg_soil = estimate_soil_temp(avg_day, avg_night)

    return {
        "avg_day_temp": float(avg_day),
        "avg_night_temp": float(avg_night),
        "avg_soil_temp": float(avg_soil),
        "total_rain": float(total_rain),
        "_source": "forecast" if include_soil else "forecast_no_soil",
    }

def get_weather_data(lat: float, lon: float) -> Dict[str, float]:
    """
    Never returns None.
    Tries multiple upstream options, then returns safe defaults.
    """
    cache_key = f"wx:{round(lat, 3)}:{round(lon, 3)}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    end = (datetime.today().date() - timedelta(days=1))
    start = end - timedelta(days=6)

    wx = weather_from_archive(lat, lon, start, end, include_soil=True)
    if wx is None:
        wx = weather_from_archive(lat, lon, start, end, include_soil=False)
    if wx is None:
        wx = weather_from_forecast(lat, lon, include_soil=True)
    if wx is None:
        wx = weather_from_forecast(lat, lon, include_soil=False)

    # Absolute last resort: never fail UI
    if wx is None:
        wx = {
            "avg_day_temp": 65.0,
            "avg_night_temp": 45.0,
            "avg_soil_temp": 50.0,
            "total_rain": 0.0,
            "_source": "fallback_defaults",
        }

    cache_set(cache_key, wx, ttl_seconds=60 * 30)
    return wx

# =============================
# Probability
# =============================
def calculate_probability(weather: Dict[str, float], trees: str) -> int:
    trees = (trees or "no").strip().lower()

    score = 0
    soil = weather.get("avg_soil_temp", 50.0)
    rain = weather.get("total_rain", 0.0)
    day = weather.get("avg_day_temp", 65.0)
    night = weather.get("avg_night_temp", 45.0)

    if 45 <= soil <= 55:
        score += 30
    if 0.5 <= rain <= 2.0:
        score += 25
    if 60 <= day <= 75:
        score += 20
    if night > 40:
        score += 15
    if trees == "yes":
        score += 10

    return max(0, min(int(score), 100))

# =============================
# Route (never 500, never "weather unavailable")
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    weather = None
    probability = None
    location = ""
    trees = "yes"

    try:
        if request.method == "POST":
            location = clean_location(request.form.get("location"))
            trees = (request.form.get("trees") or "yes").strip().lower()

            if not location or len(location) < 2:
                error = "Please enter a City/ZIP or full street address."
            else:
                coords = get_coordinates(location)
                if coords is None:
                    error = (
                        "I couldn't find that location. Try:\n"
                        "• City, State (ex: Stigler, OK)\n"
                        "• ZIP (ex: 74462)\n"
                        "• Full address with city/state"
                    )
                else:
                    lat, lon = coords
                    weather = get_weather_data(lat, lon)  # NEVER None
                    probability = calculate_probability(weather, trees)

    except Exception:
        app.logger.exception("Unhandled error")
        error = "Server hiccup—please try again."

    return render_template(
        "index.html",
        error=error,
        weather=weather,
        probability=probability,
        location=location,
        trees=trees
    )

if __name__ == "__main__":
    app.run(debug=True)