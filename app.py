from flask import Flask, render_template, request
import requests
import json
import os
from statistics import mean

app = Flask(__name__)

COUNTER_FILE = "counter.json"


def get_total_uses():
    if not os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "w", encoding="utf-8") as f:
            json.dump({"total_uses": 0}, f)
        return 0

    try:
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return int(data.get("total_uses", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        return 0


def increment_total_uses():
    total = get_total_uses() + 1
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        json.dump({"total_uses": total}, f)
    return total


def geocode_location(location):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        raise ValueError("Could not find that location. Try a city, ZIP code, or fuller address.")

    first = results[0]
    return {
        "latitude": first["latitude"],
        "longitude": first["longitude"],
        "name": first.get("name", location),
        "admin1": first.get("admin1", ""),
        "country": first.get("country", "")
    }


def get_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "rain_sum",
            "soil_temperature_0cm"
        ]),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "forecast_days": 7
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    rain = daily.get("rain_sum", [])
    soil = daily.get("soil_temperature_0cm", [])

    if not max_temps or not min_temps:
        raise ValueError("Weather data was unavailable for that location.")

    soil_values = [x for x in soil if x is not None]
    avg_soil_temp = mean(soil_values) if soil_values else 0

    return {
        "avg_day_temp": mean(max_temps),
        "avg_night_temp": mean(min_temps),
        "total_rain": sum(x for x in rain if x is not None),
        "avg_soil_temp": avg_soil_temp
    }


def calculate_probability(weather, trees):
    score = 0

    soil = weather["avg_soil_temp"]
    rain = weather["total_rain"]
    day = weather["avg_day_temp"]
    night = weather["avg_night_temp"]

    # Soil temp
    if 45 <= soil <= 55:
        score += 30
    elif 40 <= soil < 45 or 55 < soil <= 60:
        score += 18
    elif soil > 0:
        score += 8

    # Rain
    if 0.5 <= rain <= 2.0:
        score += 25
    elif 0.2 <= rain < 0.5 or 2.0 < rain <= 3.0:
        score += 15
    elif rain > 0:
        score += 8

    # Day temp
    if 60 <= day <= 75:
        score += 20
    elif 55 <= day < 60 or 75 < day <= 80:
        score += 12
    else:
        score += 5

    # Night temp
    if night >= 40:
        score += 15
    elif 35 <= night < 40:
        score += 8
    else:
        score += 3

    # Trees bonus
    if trees == "yes":
        score += 10

    return max(0, min(100, round(score)))


@app.route("/", methods=["GET", "POST"])
def index():
    total_uses = get_total_uses()

    location = ""
    trees = "yes"
    error = None
    weather = None
    probability = None

    if request.method == "POST":
        location = request.form.get("location", "").strip()
        trees = request.form.get("trees", "yes")

        if not location:
            error = "Please enter a city, ZIP code, or address."
        else:
            try:
                geo = geocode_location(location)
                weather = get_weather_data(geo["latitude"], geo["longitude"])
                probability = calculate_probability(weather, trees)

                display_location_parts = [
                    geo.get("name", ""),
                    geo.get("admin1", ""),
                    geo.get("country", "")
                ]
                location = ", ".join(part for part in display_location_parts if part)

                total_uses = increment_total_uses()
            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        total_uses=total_uses,
        location=location,
        trees=trees,
        error=error,
        weather=weather,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)