"""
Morel Mushroom Predictor — launch-ready Flask app (robust + "foolproof"-ish)

✅ Fixes:
- Reliable geocoding (Open-Meteo primary, Nominatim fallback)
- Weather fallback (Archive → Forecast w/ past_days)
- Avoids "today" archive edge cases (uses yesterday)
- Handles missing soil temperature gracefully (estimate)
- Strong input validation + safe defaults
- Timeouts + retries + HTTP status handling
- Simple in-memory caching to reduce API calls + rate-limit pain
- Structured error messages (no silent failures)
"""

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

# -----------------------------
# App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# HTTP session with retries
# -----------------------------
def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Use a realistic User-Agent (important for some services)
    session.headers.update(
        {
            "User-Agent": "MorelMushroomPredictor/1.0 (+https://example.com; contact: you@example.com)",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


HTTP = build_session()

# -----------------------------
# Tiny in-memory cache
# -----------------------------
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


# -----------------------------
# Helpers
# -----------------------------
ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")


def clean_location(s: str) -> str:
    s = (s or "").strip()
    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def safe_average(values) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def estimate_soil_temp(avg_day: float, avg_night: float) -> float:
    """
    Simple heuristic: soil lags behind air temps a bit.
    Keeps the app functional when soil data isn't returned.
    """
    return ((avg_day + avg_night) / 2.0) - 5.0


# -----------------------------
# Geocoding (reliable)
# -----------------------------
def geocode_open_meteo(location: str) -> Optional[Tuple[float, float]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": location, "count": 1, "language": "en", "format": "json"}
    r = HTTP.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return None
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    return float(results[0]["latitude"]), float(results[0]["longitude"])


def geocode_nominatim(location: str) -> Optional[Tuple[float, float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1, "addressdetails": 0}
    # Nominatim can be picky; a proper UA helps.
    r = HTTP.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def get_coordinates(location: str) -> Optional[Tuple[float, float]]:
    """
    Launch-ready geocoder:
    - cache results
    - Open-Meteo first (more forgiving)
    - fallback to Nominatim
    """
    location = clean_location(location)
    if not location:
        return None

    cache_key = f"geo:{location.lower()}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    # If it's a ZIP, still let the geocoder handle it; Open-Meteo usually works.
    coords = geocode_open_meteo(location)
    if coords is None:
        coords = geocode_nominatim(location)

    if coords:
        cache_set(cache_key, coords, ttl_seconds=60 * 60 * 24 * 14)  # 14 days
    return coords


# -----------------------------
# Weather (Archive → Forecast fallback)
# -----------------------------
def weather_from_archive(lat: float, lon: float, start: date, end: date) -> Optional[Dict[str, float]]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,rain_sum,soil_temperature_0_to_10cm_mean",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    r = HTTP.get(url, params=params, timeout=14)
    if r.status_code != 200:
        return None

    data = r.json()
    daily = data.get("daily") or {}

    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    rain = daily.get("rain_sum") or []
    soil = daily.get("soil_temperature_0_to_10cm_mean") or []

    avg_day = safe_average(tmax)
    avg_night = safe_average(tmin)
    total_rain = float(sum([x for x in rain if x is not None])) if rain else 0.0
    avg_soil = safe_average(soil)

    # Archive sometimes has temps but no soil; estimate soil if needed
    if avg_soil is None and avg_day is not None and avg_night is not None:
        avg_soil = estimate_soil_temp(avg_day, avg_night)

    if avg_day is None or avg_night is None:
        return None

    return {
        "avg_day_temp": float(avg_day),
        "avg_night_temp": float(avg_night),
        "avg_soil_temp": float(avg_soil) if avg_soil is not None else 50.0,
        "total_rain": float(total_rain),
    }


def weather_from_forecast(lat: float, lon: float) -> Optional[Dict[str, float]]:
    """
    Forecast endpoint can provide past_days and sometimes hourly soil temp.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "hourly": "soil_temperature_0_to_10cm",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "past_days": 7,
    }
    r = HTTP.get(url, params=params, timeout=14)
    if r.status_code != 200:
        return None

    data = r.json()
    daily = data.get("daily") or {}
    hourly = data.get("hourly") or {}

    avg_day = safe_average(daily.get("temperature_2m_max", []))
    avg_night = safe_average(daily.get("temperature_2m_min", []))
    total_rain = float(sum([x for x in (daily.get("precipitation_sum") or []) if x is not None]))

    # soil hourly can be long; average it
    soil_hourly = hourly.get("soil_temperature_0_to_10cm") or []
    avg_soil = safe_average(soil_hourly)

    if avg_day is None or avg_night is None:
        return None

    if avg_soil is None:
        avg_soil = estimate_soil_temp(avg_day, avg_night)

    return {
        "avg_day_temp": float(avg_day),
        "avg_night_temp": float(avg_night),
        "avg_soil_temp": float(avg_soil),
        "total_rain": float(total_rain),
    }


def get_weather_data(lat: float, lon: float) -> Optional[Dict[str, float]]:
    """
    Launch-ready weather fetch:
    - cache results per lat/lon for 30 minutes
    - use yesterday as archive end (avoids incomplete "today")
    - Archive first, Forecast fallback
    """
    cache_key = f"wx:{round(lat, 3)}:{round(lon, 3)}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    # Use yesterday to avoid archive "today" issues
    end = (datetime.today().date() - timedelta(days=1))
    start = end - timedelta(days=6)

    wx = weather_from_archive(lat, lon, start, end)
    if wx is None:
        wx = weather_from_forecast(lat, lon)

    if wx is None:
        return None

    cache_set(cache_key, wx, ttl_seconds=60 * 30)  # 30 min
    return wx


# -----------------------------
# Probability
# -----------------------------
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

    # Clamp 0–100
    return max(0, min(int(score), 100))


# -----------------------------
# Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    weather = None
    probability = None

    location = ""
    trees = "yes"

    if request.method == "POST":
        location = clean_location(request.form.get("location"))
        trees = (request.form.get("trees") or "yes").strip().lower()

        # Basic validation
        if not location:
            error = "Please enter a City, State or ZIP."
        elif len(location) < 2:
            error = "That location looks too short—try City, State or ZIP."
        else:
            coords = get_coordinates(location)
            if coords is None:
                error = "Location not found. Try 'City, State' or a 5-digit ZIP."
            else:
                lat, lon = coords
                weather = get_weather_data(lat, lon)
                if weather is None:
                    error = (
                        "Weather data is temporarily unavailable for that location. "
                        "Try again in a moment, or try a nearby city/ZIP."
                    )
                else:
                    probability = calculate_probability(weather, trees)

    return render_template(
        "index.html",
        error=error,
        weather=weather,
        probability=probability,
        location=location,
        trees=trees
    )


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    # For launch, run behind a real server like gunicorn:
    #   gunicorn -w 2 -b 0.0.0.0:8000 app:app
    app.run(debug=True)