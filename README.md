# 🌧️ Weather-Aided AI Calibration of Monsoon Rainfall

> Improving satellite rainfall estimates with machine learning and ground-based 
> weather data to support farmers across India.

## 📖 Overview

Monsoon tracking in India faces a critical gap: satellites offer broad coverage 
but lack the accuracy needed for operational use, and Northwestern India suffers 
from a shortage of precipitation radars. This project tackles that problem by 
using **machine learning to dynamically calibrate satellite rainfall data** 
against ground-truth measurements from CUNY-installed weather stations.

The result? More reliable, real-time rainfall estimates that can power flash 
flood warnings and smarter irrigation planning.

## 🎯 The Problem

- 🛰️ Satellite rainfall estimates aren't consistently accurate enough for real-world decisions
- 📉 Northwestern India lacks sufficient precipitation radar coverage
- 🌪️ Traditional physical models struggle with data limitations and built-in assumptions
- 👨‍🌾 Farmers in regions like Maharashtra and Odisha need better tools to plan around the monsoon

## 💡 Our Approach

Unlike previous point-by-point calibration methods, our ML model brings in the 
**spatial pattern** of satellite data alongside local weather parameters from 
ground stations.

**Inputs**
- Microwave satellite rainfall estimates (~20 km resolution)
- Vertical weather profiles: wind and humidity at multiple elevations
- Ground-truth rainfall from CUNY weather stations

**Training**
- ~300 training samples & ~60 validation samples per monsoon season
- 5 station locations per region (Maharashtra & Odisha)
- Rotating validation stations across training sessions

**Output**
- Farmer-accessible, calibrated rainfall data for irrigation and flood planning

## 🚀 Future Directions

- 🌏 Expand to other regions of India and global datasets (e.g., Australia, similar latitudes)
- 🔁 Annual model retraining for continuous accuracy improvement
- 📱 Language-localized tools to deliver bite-sized forecasts to farmers
- ⏱️ Interpolating satellite gaps using weather model data between 3-hour intervals

## 👥 Team Big Apple

Built by a cross-CUNY team:
- **Victor Carrion** — Bronx Community College
- **Trishtan Balkaran** — Baruch College
- **Aaryan Nair** — Graduate Center
- **Kazi Tasin** — NYC College of Technology
- **Sue-Moura Burke** — Bronx Community College

With support from Prof. Neal Phillip, Prof. Paramita Sen, and Prof. Brian Vant-Hull.

## 🏷️ Topics

`machine-learning` `climate-tech` `satellite-data` `weather-forecasting` 
`monsoon` `precipitation` `agriculture-tech` `python` `data-science`
