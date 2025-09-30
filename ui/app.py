# -*- coding: utf-8 -*-
"""Streamlit UI for IAM marketing platform."""
from __future__ import annotations

import html
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME") or "IAMistralbot"
SEGMENTATION_PATH = Path(os.getenv("SEGMENTATION_PATH", "segmentationRAG/segmentation_sms.csv"))
SMARTPHONES_PATH = Path(os.getenv("SMARTPHONES_PATH", "segmentationRAG/data/smartphones.csv"))
LIVE_LLM_URL = os.getenv("LIVE_LLM_URL", "")
LIVE_LLM_API_KEY = os.getenv("LIVE_LLM_API_KEY", "")
DEFAULT_LIVE_LLM_MODEL_ID = "AymenKhomsi/mistral7b-iam-sms-lora-4bit"

FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "OPPORTUNITE_Achat_Equipement": (
        "Includes customers whose data usage or phone model suggests they are prime candidates for a hardware upgrade. "
        "They are either using outdated devices, are fans of new technology, or are ready for a high-performance smartphone."
    ),
    "USAGE_Divertissement": (
        "These are customers who actively use their mobile lines for entertainment purposes. This family includes avid mobile gamers "
        "and subscribers to premium content services like video streaming and music apps."
    ),
    "USAGE_Internet": (
        "This family consists of heavy data users. It includes customers who consume large amounts of general internet data as well "
        "as those who focus specifically on social media platforms."
    ),
    "USAGE_Mixte": (
        "A versatile group of customers with varied behaviors. This family includes budget-conscious users, professionals with high-value "
        "subscriptions, and flexible users who switch between different types of offers based on their needs."
    ),
    "USAGE_Roaming": (
        "This family is composed of customers who use their phones while traveling abroad. It includes both frequent and occasional "
        "travelers, with needs ranging from data-only passes to comprehensive voice, SMS, and data packages."
    ),
    "USAGE_SMS": (
        "These are traditional users who still rely heavily on SMS for their communication needs. They are regular purchasers of SMS "
        "bundles and represent a classic mobile usage pattern."
    ),
    "USAGE_Voix": (
        "This family includes customers whose primary mobile activity is making phone calls. It covers profiles from heavy national "
        "callers to loyal Maroc Telecom customers and frequent international dialers."
    ),
}

PERSONA_DESCRIPTIONS: Dict[str, str] = {
    "OPPORTUNITE_AchatSmartphone": "Customers moving up from basic or aging devices who respond to accessible upgrade bundles and starter smartphones.",
    "OPPORTUNITE_PerformanceSmartphone": "Power users who push their hardware to the limit and expect flagship specs, premium connectivity, and generous data allowances.",
    "OPPORTUNITE_AchatNouveaute": "Trendsetters and early adopters eager for the latest handset launches, limited editions, and exclusive drops.",
    "PROFIL_Econome": "This customer is highly budget-conscious, making small but frequent recharges to maximise value across minutes and data.",
    "PROFIL_Flexible": "Needs change frequently. This persona relies on versatile bundles that adapt on demand between voice and data.",
    "PROFIL_Professionnel": "A high-value customer whose usage resembles that of a business pro: multi-SIM devices, premium spend, and reliability requirements.",
    "PROFIL_ServicesPremium": "Consumes premium digital content and regularly purchases subscriptions to services like STARZPLAY, music, or video platforms.",
    "PROFIL_Gamer": "Data usage is heavily concentrated on mobile gaming, making latency and stability critical to the experience.",
    "OPPORTUNITE_InternetCher": "Heavy data users who still pay out-of-bundle rates. They are prime targets for education on dedicated data passes.",
    "PROFIL_Internet": "Classic power browsers who already understand the value of data bundles and top up frequently with large passes.",
    "PROFIL_ReseauxSociaux": "Social media addicts whose consumption is concentrated on apps like TikTok, Instagram, and WhatsApp.",
    "OPPORTUNITE_VoyageurOccasionnel": "Occasional travellers who used roaming without a pass and need guidance to avoid high standard fees next time.",
    "PROFIL_VoyageurComplet": "Seasoned travellers who regularly activate comprehensive roaming packages combining data, voice, and SMS.",
    "PROFIL_Sms": "Traditional communicators who still rely on SMS bundles as their primary channel.",
    "PROFIL_FideleOnNet": "Voice users loyal to the Maroc Telecom network, generating high on-net call volumes that suit IAM-centric promotions.",
    "PROFIL_VoixNational": "Heavy national callers who dial across all Moroccan operators and appreciate inclusive minute bundles.",
    "PROFIL_CommunicantInternational": "Globally connected callers who need competitive international rates to stay in touch abroad.",
}


def _friendly_label(value: Optional[str]) -> str:
    if not value:
        return ""
    formatted = value.replace("_", " ")
    formatted = formatted.replace("OPPORTUNITE", "Opportunité").replace("PROFIL", "Profil").replace("USAGE", "Usage")
    return formatted


def render_contextual_insight(famille: Optional[str], persona: Optional[str]) -> None:
    sections: List[str] = []
    if famille:
        desc = FAMILY_DESCRIPTIONS.get(famille)
        if desc:
            sections.append(f"<strong>{_friendly_label(famille)}</strong> — {desc}")
    if persona:
        desc = PERSONA_DESCRIPTIONS.get(persona)
        if desc:
            sections.append(f"<strong>{_friendly_label(persona)}</strong> — {desc}")
    if not sections:
        st.markdown("""
            <div class='insight-text-empty' style='text-align:center; padding:24px 16px'>
                <div style='font-size:36px; margin-bottom:12px; opacity:0.6'>🧠</div>
                <p style='margin:0 0 16px; font-weight:600; color:var(--text-soft)'>Smart Insights Awaiting</p>
                <p style='margin:0; font-size:12px; opacity:0.8'>Select a usage type and persona to unlock contextual insights</p>
                <div style='display:flex; justify-content:center; gap:8px; margin-top:16px; font-size:20px; opacity:0.4'>
                    <span style='background:rgba(255,255,255,0.1); padding:6px 10px; border-radius:10px'>👥</span>
                    <span style='background:rgba(255,255,255,0.1); padding:6px 10px; border-radius:10px'>📊</span>
                    <span style='background:rgba(255,255,255,0.1); padding:6px 10px; border-radius:10px'>💡</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return
    st.markdown(
        """
        <div class='insight-text'>
            {sections_html}
        </div>
        """.format(sections_html="<br/><br/>".join(sections)),
        unsafe_allow_html=True,
    )


def build_sms_preview_html(sms_text: str, llm_input: Dict) -> str:
    offer_ctx = llm_input.get("offer_context", {}) or {}
    price, volume, validity = extract_offer_metrics(offer_ctx)

    model = (offer_ctx.get("modele") or "").strip()
    capacity = (offer_ctx.get("capacite") or "").strip()
    offer_name = (offer_ctx.get("offre") or "").strip()
    cta = (llm_input.get("cta") or "").strip()

    title = model or offer_name or cta or "IAM Campaign"
    label = "Model" if model else ("Offer" if offer_name else "CTA")

    detail_lines: List[str] = []
    if capacity:
        detail_lines.append(f"Capacity: {capacity}")
    if not model and volume:
        detail_lines.append(f"Data: {volume}")
    if validity:
        detail_lines.append(f"Validity: {validity}")
    minutes = offer_ctx.get("minutes")
    if minutes:
        detail_lines.append(f"Minutes: {minutes}")

    price_display = price or volume or validity or "--"
    info_block = "".join(
        f"<div class='product-meta'>{html.escape(line)}</div>" for line in detail_lines
    )

    orientation_class = "landscape"  # forced landscape orientation
    return f"""
    <div class='phone-mockup {orientation_class}'>
        <div class='phone-screen'>
            <div class='sms-bubble'>{html.escape(sms_text)}</div>
            <div class='product-info'>
                <div>
                    <div class='product-meta'>{html.escape(label)}</div>
                    <div class='product-name'>{html.escape(title)}</div>
                    {info_block}
                </div>
                <div class='product-price'>{html.escape(price_display)}</div>
            </div>
        </div>
    </div>
    """

if TELEGRAM_BOT_USERNAME == "IAMistralbot" and TELEGRAM_BOT_TOKEN:
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
            timeout=5,
        )
        data = resp.json()
        TELEGRAM_BOT_USERNAME = data.get("result", {}).get("username") or "IAMistralbot"
    except Exception:
        TELEGRAM_BOT_USERNAME = "IAMistralbot"

st.set_page_config(page_title="IAM Marketing Platform", page_icon="📡", layout="wide")

if "LIVE_LLM_URL" not in st.session_state and LIVE_LLM_URL and LIVE_LLM_URL.strip():
    st.session_state["LIVE_LLM_URL"] = LIVE_LLM_URL.strip()
if "LIVE_LLM_MODEL_ID" not in st.session_state:
    st.session_state["LIVE_LLM_MODEL_ID"] = DEFAULT_LIVE_LLM_MODEL_ID

st.markdown("""
<style>
/* ==============================
   PREMIUM ENTERPRISE UI SYSTEM
============================== */

/* ------ CSS CUSTOM PROPERTIES ------ */
:root {
  /* Primary Brand Colors */
  --primary-50: #f0f7ff;
  --primary-100: #e1efff;
  --primary-200: #b3d7ff;
  --primary-300: #7db8ff;
  --primary-400: #4a94ff;
  --primary-500: #1a70ff;
  --primary-600: #0056e6;
  --primary-700: #0043b3;
  --primary-800: #003480;
  --primary-900: #002654;
  
  /* Accent & Status Colors */
  --accent-gradient: linear-gradient(135deg, #ff4081 0%, #ff1744 100%);
  --success: #00c853;
  --warning: #ffa726;
  --error: #f44336;
  --info: var(--primary-500);
  
  /* Sophisticated Background System */
  --bg-primary: #0a0d1b;
  --bg-secondary: #0f1319;
  --bg-tertiary: #151922;
  --bg-card: rgba(21, 25, 34, 0.85);
  --bg-card-hover: rgba(25, 30, 40, 0.92);
  --bg-glass: rgba(255, 255, 255, 0.03);
  --bg-glass-hover: rgba(255, 255, 255, 0.06);
  
  /* Premium Borders */
  --border-light: rgba(255, 255, 255, 0.08);
  --border-medium: rgba(255, 255, 255, 0.15);
  --border-strong: rgba(255, 255, 255, 0.25);
  --border-accent: rgba(26, 112, 255, 0.4);
  
  /* Typography Scale */
  --text-primary: #ffffff;
  --text-secondary: #e2e8f0;
  --text-tertiary: #94a3b8;
  --text-quaternary: #64748b;
  --text-disabled: #475569;
  
  /* Shadow System */
  --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.12);
  --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.18);
  --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.24);
  --shadow-xl: 0 24px 64px rgba(0, 0, 0, 0.32);
  --shadow-primary: 0 8px 32px rgba(26, 112, 255, 0.25);
  
  /* Spacing System */
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-md: 1rem;
  --space-lg: 1.5rem;
  --space-xl: 2rem;
  --space-2xl: 3rem;
  
  /* Border Radius */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 24px;
  --radius-2xl: 32px;
  
  /* Transitions */
  --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-normal: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-slow: 0.35s cubic-bezier(0.4, 0, 0.2, 1);
  --transition-bounce: 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

/* ------ ADVANCED KEYFRAME ANIMATIONS ------ */
@keyframes float {
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  25% { transform: translateY(-8px) rotate(2deg); }
  50% { transform: translateY(-4px) rotate(-1deg); }
  75% { transform: translateY(-12px) rotate(1deg); }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

@keyframes pulse-glow {
  0%, 100% { 
    box-shadow: 0 0 20px rgba(26, 112, 255, 0.3);
    transform: scale(1);
  }
  50% { 
    box-shadow: 0 0 40px rgba(26, 112, 255, 0.6);
    transform: scale(1.02);
  }
}

@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(30px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes morphGradient {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

@keyframes ripple {
  0% {
    transform: scale(0);
    opacity: 1;
  }
  100% {
    transform: scale(4);
    opacity: 0;
  }
}

@keyframes loadingSpinner {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* ------ GLOBAL FOUNDATION ------ */
* {
  box-sizing: border-box;
}

body {
  background: 
    radial-gradient(ellipse at top, rgba(26, 112, 255, 0.08) 0%, transparent 50%),
    radial-gradient(ellipse at bottom, rgba(255, 64, 129, 0.06) 0%, transparent 50%),
    linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-primary) 100%);
  color: var(--text-primary);
  font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
  font-feature-settings: 'kern' 1, 'liga' 1, 'calt' 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
  min-height: 100vh;
  overflow-x: hidden;
}

.block-container {
  max-width: 1200px !important;
  padding: var(--space-lg) var(--space-md) var(--space-2xl) !important;
  margin: 0 auto !important;
  animation: slideInUp 0.8s var(--transition-normal);
}

.main > div {
  padding-top: 0 !important;
}

/* ------ PREMIUM HERO SECTION ------ */
.hero {
  position: relative;
  background: 
    linear-gradient(135deg, var(--primary-600) 0%, var(--primary-800) 100%);
  border-radius: var(--radius-2xl);
  padding: var(--space-2xl) var(--space-xl);
  text-align: center;
  box-shadow: var(--shadow-primary);
  margin-bottom: var(--space-2xl);
  overflow: hidden;
  animation: slideInUp 0.6s var(--transition-normal);
}

.hero::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -20%;
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
  animation: float 8s ease-in-out infinite;
  z-index: 1;
}

.hero::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
  background-size: 200% 200%;
  animation: shimmer 3s ease-in-out infinite;
  z-index: 2;
}

.hero h1 {
  margin: 0 0 var(--space-sm);
  font-size: clamp(2rem, 4vw, 2.5rem);
  font-weight: 800;
  background: linear-gradient(135deg, #ffffff 0%, #e1efff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  position: relative;
  z-index: 3;
  letter-spacing: -0.02em;
}

.hero p {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 500;
  opacity: 0.9;
  position: relative;
  z-index: 3;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ------ ENTERPRISE GRID SYSTEM ------ */
.enterprise-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-xl);
  margin-bottom: var(--space-2xl);
}

/* ------ PREMIUM CARD SYSTEM ------ */
.premium-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  padding: var(--space-xl);
  position: relative;
  overflow: hidden;
  transition: all var(--transition-slow);
  box-shadow: var(--shadow-md);
  animation: slideInUp 0.8s var(--transition-normal) backwards;
}

.premium-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--primary-500), transparent);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.premium-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: var(--shadow-xl);
  border-color: var(--border-accent);
  background: var(--bg-card-hover);
}

.premium-card:hover::before {
  opacity: 1;
}

.premium-card:nth-child(2) { animation-delay: 0.1s; }
.premium-card:nth-child(3) { animation-delay: 0.2s; }
.premium-card:nth-child(4) { animation-delay: 0.3s; }

/* ------ CARD HEADERS ------ */
.card-header {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  margin-bottom: var(--space-lg);
}

.card-icon {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  background: linear-gradient(135deg, var(--primary-500), var(--primary-700));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  box-shadow: var(--shadow-primary);
  animation: pulse-glow 3s ease-in-out infinite;
}

.card-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0;
  letter-spacing: -0.01em;
}

.card-subtitle {
  font-size: 0.875rem;
  color: var(--text-tertiary);
  margin: var(--space-xs) 0 0;
  font-weight: 500;
}

/* ------ PREMIUM FORM CONTROLS ------ */
.stSelectbox > div > div,
.stTextInput > div > div > input,
[data-baseweb="select"],
[data-baseweb="input"] {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-weight: 500 !important;
  transition: all var(--transition-normal) !important;
  backdrop-filter: blur(10px) !important;
  min-height: 48px !important;
}

.stSelectbox > div > div:hover,
.stTextInput > div > div > input:hover,
[data-baseweb="select"]:hover,
[data-baseweb="input"]:hover {
  border-color: var(--border-accent) !important;
  background: var(--bg-glass-hover) !important;
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-primary) !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
[data-baseweb="select"]:focus-within,
[data-baseweb="input"]:focus {
  border-color: var(--primary-500) !important;
  box-shadow: 0 0 0 3px rgba(26, 112, 255, 0.2) !important;
  background: var(--bg-glass-hover) !important;
}

/* ------ RADIO BUTTONS ------ */
.stRadio > div {
  display: flex !important;
  gap: var(--space-sm) !important;
  flex-wrap: wrap !important;
}

.stRadio > div > label {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-sm) var(--space-md) !important;
  font-weight: 600 !important;
  transition: all var(--transition-normal) !important;
  cursor: pointer !important;
  position: relative !important;
  overflow: hidden !important;
}

.stRadio > div > label::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  background: rgba(26, 112, 255, 0.3);
  border-radius: 50%;
  transition: all var(--transition-fast);
  transform: translate(-50%, -50%);
}

.stRadio > div > label:active::before {
  width: 100px;
  height: 100px;
  animation: ripple 0.6s ease-out;
}

.stRadio > div > label:hover {
  border-color: var(--border-accent) !important;
  background: var(--bg-glass-hover) !important;
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-sm) !important;
}

.stRadio > div > label[data-checked="true"] {
  background: linear-gradient(135deg, var(--primary-500), var(--primary-700)) !important;
  color: white !important;
  border-color: var(--primary-500) !important;
  box-shadow: var(--shadow-primary) !important;
}

/* ------ PREMIUM BUTTONS ------ */
.stButton > button {
  background: var(--accent-gradient) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  color: white !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  padding: var(--space-md) var(--space-xl) !important;
  min-height: 48px !important;
  transition: all var(--transition-normal) !important;
  box-shadow: var(--shadow-md) !important;
  position: relative !important;
  overflow: hidden !important;
  letter-spacing: 0.02em !important;
  width: 100% !important;
}

.stButton > button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.8s ease;
}

.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: var(--shadow-lg) !important;
}

.stButton > button:hover::before {
  left: 100%;
}

.stButton > button:active {
  transform: translateY(-1px) scale(0.98) !important;
}

/* ------ SMS PREVIEW PHONE ------ */
.phone-mockup {
  max-width: 320px;
  margin: 0 auto;
  background: linear-gradient(145deg, #1a1f35, #151a2e);
  border-radius: var(--radius-2xl);
  padding: var(--space-lg);
  box-shadow: var(--shadow-xl);
  position: relative;
  overflow: hidden;
}

.phone-mockup::before {
  content: '';
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 4px;
  background: var(--text-quaternary);
  border-radius: 2px;
}

.phone-screen {
  background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
  border-radius: var(--radius-lg);
  padding: var(--space-lg);
  min-height: 240px;
  position: relative;
}

.sms-bubble {
  background: linear-gradient(135deg, var(--primary-500), var(--primary-700));
  color: white;
  padding: var(--space-md) var(--space-lg);
  border-radius: 18px 18px 4px 18px;
  margin-bottom: var(--space-md);
  box-shadow: var(--shadow-primary);
  animation: slideInRight 0.6s var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.sms-bubble::after {
  content: '';
  position: absolute;
  bottom: -2px;
  right: -2px;
  width: 8px;
  height: 8px;
  background: var(--primary-700);
  transform: rotate(45deg);
}

/* ------ LANDSCAPE PHONE VARIANT ------ */
.phone-mockup.landscape {
    max-width: 560px;
    padding: var(--space-md) var(--space-lg);
}

.phone-mockup.landscape .phone-screen {
    min-height: 200px;
    display: flex;
    flex-direction: row;
    gap: var(--space-lg);
    align-items: stretch;
    padding: var(--space-lg) var(--space-xl);
}

.phone-mockup.landscape::before {
    top: 50%;
    left: 8px;
    transform: translateY(-50%);
    width: 4px;
    height: 60px;
}

.phone-mockup.landscape .sms-bubble {
    flex: 1 1 68%;
    max-height: 100%;
    overflow: hidden; /* remove scrollbar track */
    margin-bottom: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

/* Hide webkit scrollbar specifically within landscape sms bubble */
.phone-mockup.landscape .sms-bubble::-webkit-scrollbar {
    width: 0;
    height: 0;
}

.phone-mockup.landscape .product-info {
    flex: 1 1 32%;
    margin-top: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* Responsive fallback: revert to portrait under 640px */
@media (max-width: 640px) {
    .phone-mockup.landscape {
        max-width: 360px;
    }
    .phone-mockup.landscape .phone-screen {
        flex-direction: column;
    }
    .phone-mockup.landscape .sms-bubble {
        margin-bottom: var(--space-md);
    }
    .phone-mockup.landscape .product-info {
        margin-top: 0;
    }
    .phone-mockup.landscape::before {
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 4px;
    }
}

/* ------ PRODUCT INFO CARDS ------ */
.product-info {
  background: var(--bg-glass);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  margin-top: var(--space-md);
  transition: all var(--transition-normal);
}

.product-info:hover {
  background: var(--bg-glass-hover);
  border-color: var(--border-medium);
}

.product-name {
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: var(--space-xs);
}

.product-price {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--success);
}

.product-meta {
  font-size: 0.75rem;
  color: var(--text-tertiary);
  margin-bottom: 2px;
}

/* ------ INSIGHT SECTIONS ------ */
.insight-container {
  background: var(--bg-glass);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
  margin-top: var(--space-md);
  backdrop-filter: blur(10px);
}

.insight-text {
  color: var(--text-secondary);
  line-height: 1.7;
  font-size: 1.02rem;
  background: var(--bg-glass);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
  margin-top: var(--space-md);
}

.insight-text strong {
  color: var(--primary-400);
  font-weight: 700;
}

.insight-text-empty {
  color: var(--text-tertiary);
  text-align: center;
  font-style: italic;
  padding: var(--space-xl);
  border: 2px dashed var(--border-light);
  border-radius: var(--radius-md);
  background: var(--bg-glass);
  margin-top: var(--space-md);
}

/* ------ TELEGRAM SECTION ------ */
.telegram-section {
  background: linear-gradient(135deg, rgba(26, 112, 255, 0.1), rgba(26, 112, 255, 0.05));
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-xl);
  padding: var(--space-xl);
  text-align: center;
  backdrop-filter: blur(10px);
}

/* ------ LOADING STATES ------ */
.loading-spinner {
  width: 24px;
  height: 24px;
  border: 3px solid var(--border-light);
  border-top: 3px solid var(--primary-500);
  border-radius: 50%;
  animation: loadingSpinner 1s linear infinite;
  margin: 0 auto;
}

/* ------ RESPONSIVE DESIGN ------ */
@media (max-width: 768px) {
  .block-container {
    padding: var(--space-md) var(--space-sm) var(--space-xl) !important;
  }
  
  .enterprise-grid {
    grid-template-columns: 1fr;
    gap: var(--space-lg);
  }
  
  .hero {
    padding: var(--space-xl) var(--space-lg);
  }
  
  .hero h1 {
    font-size: 1.75rem;
  }
  
  .premium-card {
    padding: var(--space-lg);
  }
}

/* ------ ACCESSIBILITY IMPROVEMENTS ------ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* ------ STREAMLIT OVERRIDES ------ */
.stMarkdown { margin-bottom: 0 !important; }
.element-container { margin: 0 !important; }
[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="element-container"] { margin-bottom: var(--space-md) !important; }
[data-testid="column"] { padding: 0 var(--space-sm) !important; }

/* ------ SUCCESS STATES ------ */
.success-message {
  background: linear-gradient(135deg, rgba(0, 200, 83, 0.1), rgba(0, 200, 83, 0.05));
  border: 1px solid rgba(0, 200, 83, 0.3);
  border-radius: var(--radius-md);
  padding: var(--space-md);
  color: var(--success);
  font-weight: 600;
  animation: slideInUp 0.5s var(--transition-normal);
}

/* ------ FOOTER ------ */
footer {
  text-align: center;
  margin-top: var(--space-2xl);
  padding: var(--space-lg) 0;
  border-top: 1px solid var(--border-light);
  color: var(--text-tertiary);
  font-size: 0.875rem;
}
</style>
""", unsafe_allow_html=True)


DIVERTISSEMENT_TAGS = {
    "PROFIL_Gamer": ["*88"],
    "PROFIL_ServicesPremium": ["*9"],
}

@st.cache_data
def load_segmentation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {path}")
    df = pd.read_csv(path, delimiter=";")
    df = df.fillna("")
    return df

@st.cache_data
def load_smartphones(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Smartphones file not found: {path}")
    return pd.read_csv(path)

SEG_DF = load_segmentation(SEGMENTATION_PATH)
SEG_DF = SEG_DF[SEG_DF["famille"].str.upper() != "RISQUE_CHURN"].copy()
SMARTPHONE_DF = load_smartphones(SMARTPHONES_PATH)

_brand_lookup: Dict[str, str] = {}
for raw_brand in SMARTPHONE_DF.get("marque", []):
    if isinstance(raw_brand, str) and raw_brand.strip():
        upper = raw_brand.strip().upper()
        _brand_lookup.setdefault(upper, raw_brand.strip())

SMARTPHONE_MODEL_TO_BRAND: Dict[str, str] = {}
for _, row in SMARTPHONE_DF.iterrows():
    model_name = str(row.get("modele") or "").strip().upper()
    brand_name = str(row.get("marque") or "").strip()
    if model_name and brand_name:
        SMARTPHONE_MODEL_TO_BRAND.setdefault(model_name, brand_name)

SMARTPHONE_BRANDS = sorted(_brand_lookup.values())
SMARTPHONE_BRANDS_UPPER = set(_brand_lookup.keys())
FAMILLES = sorted([fam for fam in SEG_DF["famille"].unique() if fam])

@dataclass
class Selection:
    famille: str
    persona: str
    value: str
    is_equipment: bool

def normalise_brand(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip()

def dependent_dropdowns() -> Optional[Selection]:
    famille = st.selectbox("Customer Usage Type", FAMILLES, key="famille_select")
    subset = SEG_DF[SEG_DF["famille"] == famille]
    personas = sorted([p for p in subset["persona"].unique() if p])
    if famille == "USAGE_Divertissement":
        extras = [p for p in DIVERTISSEMENT_TAGS.keys() if p not in personas]
        personas.extend(sorted(extras))
    persona = st.selectbox("Persona", personas, key="persona_select")

    persona_rows = subset[subset["persona"] == persona]
    is_equipment = famille == "OPPORTUNITE_Achat_Equipement"

    if is_equipment:
        if persona in ["OPPORTUNITE_AchatSmartphone", "OPPORTUNITE_PerformanceSmartphone"]:
            # No brand choice for these personas
            label = None
            value = ""
        elif persona == "OPPORTUNITE_AchatNouveaute":
            label = "Handset brand"
            values = SMARTPHONE_BRANDS
        else:
            label = "Handset brand"
            raw_values = sorted({normalise_brand(val) for val in persona_rows["hset_brand"] if val})
            filtered: List[str] = []
            seen: set[str] = set()
            for val in raw_values:
                upper = val.upper()
                if upper in SMARTPHONE_BRANDS_UPPER:
                    display = _brand_lookup.get(upper, val)
                    if display not in seen:
                        seen.add(display)
                        filtered.append(display)
            values = filtered or SMARTPHONE_BRANDS
    else:
        label = "CTA"
        values = sorted({val for val in persona_rows["tag_offre"] if val})
        if famille == "USAGE_Divertissement":
            presets = DIVERTISSEMENT_TAGS.get(persona)
            if presets:
                values = presets

    if label is None:
        # No third dropdown
        pass
    elif not values:
        st.warning("No options available for this persona. Choose another combination.")
        return None
    else:
        value = st.selectbox(label, values, key="value_select")
    return Selection(famille=famille, persona=persona, value=value, is_equipment=is_equipment)

def call_compose(selection: Selection) -> Dict:
    endpoint = "/compose/smartphone" if selection.is_equipment else "/compose/offre"
    if selection.is_equipment:
        payload: Dict[str, str] = {
            "persona": selection.persona,
            "famille": selection.famille,
            "hset_brand": selection.value,
        }
    else:
        payload = {
            "persona": selection.persona,
            "famille": selection.famille,
            "tag_offre": selection.value,
        }
    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()

def mock_llm(llm_input: Dict) -> str:
    """
    Marketing-style French mock writer inspired by IAM tone.
    For each (famille, persona, cta/brand) triplet, pick one of 5 adapted
    templates and fill it with values from llm_input_json.
    """
    llm_input = llm_input or {}
    offer = llm_input.get("offer_context", {})
    persona = llm_input.get("persona", "CLIENT_IAM")
    famille = llm_input.get("famille", "")
    cta = (llm_input.get("cta") or "").strip()
    deadline = llm_input.get("deadline", "fin du mois")
    link = llm_input.get("links", {}).get("details", "https://iam.ma")

    # Extract smartphone brand if present
    model = (offer.get("modele") or "").strip()
    capacity = (offer.get("capacite") or "").strip()
    raw_brand = (
        offer.get("marque")
        or llm_input.get("brand")
        or llm_input.get("hset_brand")
        or ""
    )
    if not raw_brand and model:
        raw_brand = SMARTPHONE_MODEL_TO_BRAND.get(model.upper())
    if not raw_brand and model:
        raw_brand = model.split()[0]
    brand_key = (raw_brand or "").strip().upper()
    brand = _brand_lookup.get(brand_key, (raw_brand or "").strip())
    if not brand and raw_brand:
        brand = str(raw_brand).strip()
    if not brand and model:
        brand = "Smartphone"

    # Normalize price/volume for display
    price = offer.get("prix_dh")
    try:
        price_txt = f"{float(price):.0f} MAD" if price is not None else ""
    except (TypeError, ValueError):
        price_txt = f"{price} MAD" if price else ""
    volume = offer.get("volume") or ""
    minutes = offer.get("minutes") or ""
    sms_qty = offer.get("sms") or ""
    validity = offer.get("validite") or ""
    destinations = offer.get("destinations") or ""

    # Template registry: keys cascade (exact triplet → persona/family → family default)
    # Use concise, assertive IAM-like tone with CTA and deadline.
    def base_fields():
        """Build template-safe dictionary with graceful fallbacks."""
        return {
            "persona": persona,
            "famille": famille,
            "cta": cta,
            "deadline": deadline,
            "link": link,
            "offer_name": offer.get("offre", "Offre spéciale"),
            "price": price_txt,
            "price_monthly": offer.get("prix_mensuel") or llm_input.get("price_monthly", ""),
            "old_price": offer.get("ancien_prix") or llm_input.get("old_price", ""),
            "volume": volume,
            "base_volume": offer.get("base_volume") or llm_input.get("base_volume", ""),
            "bonus_volume": offer.get("bonus_volume") or offer.get("bonus_volume") or llm_input.get("bonus_volume", ""),
            "bonus_data": offer.get("bonus_data") or llm_input.get("bonus_data", ""),
            "minutes": minutes,
            "sms": sms_qty,
            "sms_count": offer.get("sms_count") or llm_input.get("sms_count", sms_qty),
            "validity": validity,
            "base_validity": offer.get("base_validite") or offer.get("base_validity") or llm_input.get("base_validity", ""),
            "bonus_credit": offer.get("bonus_credit") or llm_input.get("bonus_credit", ""),
            "dest": destinations,
            "destinations": destinations,
            "model": model,
            "capacity": capacity,
            "brand": brand,
            "brand_key": brand_key,
            "service_name": llm_input.get("service_name", ""),
            "serie_name": llm_input.get("serie_name", ""),
            "game_name": llm_input.get("game_name", ""),
            "zone": llm_input.get("zone", ""),
            "new_app": llm_input.get("new_app", ""),
        }
    
    TEMPLATES: Dict[Tuple[str, str, str], List[str]] = {
        # ===================================================================
        # Famille: OPPORTUNITE_Achat_Equipement
        # ===================================================================

        # --- Persona: OPPORTUNITE_AchatNouveaute (Ton: Exclusivité, Nouveauté) ---
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatNouveaute","APPLE"):[
            "NOUVEAU ! Réservez l' {model} {capacity} dès aujourd'hui chez Maroc Telecom et profitez d'une offre de lancement exclusive. Réservez-le ici : {link}",
            "Fan d'Apple ? Le nouvel  {model} est arrivé chez Maroc Telecom. En tant que client fidèle, soyez le premier à en profiter. Rendez-vous sur {link}",
            "EXCLUSIVITE Maroc Telecom ! Le nouvel {model} {capacity} est disponible. Commandez-le dès maintenant en Agence ou sur notre e-boutique : {link}",
            "Découvrez la puissance de l' {model} avec le réseau n°1 de Maroc Telecom. Disponible à partir de {price}. Plus d'infos sur {link}",
            "Vous exigez le meilleur. Ça tombe bien. Le nouvel {model} est chez Maroc Telecom. Commandez le vôtre dès aujourd'hui : {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatNouveaute","SAMSUNG"): [
            "NOUVEAU ! Réservez le  {model} chez Maroc Telecom et profitez du doublement de la capacité de stockage OFFERT ! Cliquez ici : {link}",
            "La nouvelle série Galaxy {model} est disponible chez Maroc Telecom ! En tant que connaisseur, précommandez le vôtre dès maintenant sur {link}",
            "Passez au niveau supérieur avec le {model} {capacity}. Profitez de son écran immersif sur le réseau 4G+ de Maroc Telecom. Dès {price} sur {link}",
            "Offre de lancement Maroc Telecom sur le nouveau  {model} ! Bénéficiez d'un bonus de reprise sur votre ancien smartphone. Rendez-vous en agence.",
            "EXCLUSIVITE IAM : le nouveau  {model} {capacity} est à {price}. Commandez-le sur notre e-boutique avec livraison gratuite : {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatNouveaute","HONOR"): [
            "EXCLUSIVITE Maroc Telecom ! Le nouveau  {model} {capacity} est disponible à {price}. Commandez-le dès maintenant sur notre e-boutique : {link}",
            "Nouveauté HONOR chez Maroc Telecom ! Découvrez le {model} et son design élégant. Offre de lancement avec {bonus_data} offerts. Plus d'infos sur {link}",
            "Ne manquez pas le lancement du  {model} chez Maroc Telecom. Un style premium et des performances de pointe. Commandez-le sur {link}",
            "Réservez votre nouveau  {model} chez Maroc Telecom et recevez une montre connectée offerte ! Offre limitée. Tous les détails sur {link}",
            "Le nouveau  {model} est arrivé chez Maroc Telecom. Profitez de son autonomie record pour seulement {price}. Découvrez-le sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatNouveaute","OPPO"): [
            "EXCLUSIVITE Maroc Telecom ! Le nouveau {model} {capacity} est disponible à {price}. Commandez-le dès maintenant sur notre e-boutique : {link}",
            "Nouveauté OPPO chez Maroc Telecom ! Découvrez le {model} et son design élégant. Offre de lancement avec {bonus_data} offerts. Plus d'infos sur {link}",
            "Ne manquez pas le lancement du  {model} chez Maroc Telecom. Un style premium et des performances de pointe. Commandez-le sur {link}",
            "Réservez votre nouveau {model} chez Maroc Telecom et recevez une montre connectée offerte ! Offre limitée. Tous les détails sur {link}",
            "Le nouveau {model} est arrivé chez Maroc Telecom. Profitez de son autonomie record pour seulement {price}. Découvrez-le sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatNouveaute","TECNO"): [
            "EXCLUSIVITE Maroc Telecom ! Le nouveau  {model} {capacity} est disponible à {price}. Commandez-le dès maintenant sur notre e-boutique : {link}",
            "Nouveauté TECNO chez Maroc Telecom ! Découvrez le {model} et son design élégant. Offre de lancement avec {bonus_data} offerts. Plus d'infos sur {link}",
            "Ne manquez pas le lancement du {model} chez Maroc Telecom. Un style premium et des performances de pointe. Commandez-le sur {link}",
            "Réservez votre nouveau {model} chez Maroc Telecom et recevez une montre connectée offerte ! Offre limitée. Tous les détails sur {link}",
            "Le nouveau {model} est arrivé chez Maroc Telecom. Profitez de son autonomie record pour seulement {price}. Découvrez-le sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatNouveaute","XIAOMI"): [
            "EXCLUSIVITE Maroc Telecom ! Le nouveau {model} {capacity} est disponible à {price}. Commandez-le dès maintenant sur notre e-boutique : {link}",
            "Nouveauté Xiaomi chez Maroc Telecom ! Découvrez le {model} et son processeur ultra-performant. Offre de lancement avec {bonus_data} offerts. Plus d'infos sur {link}",
            "Ne manquez pas le lancement du {model} chez Maroc Telecom. Un design premium et des performances de pointe. Commandez-le sur {link}",
            "Réservez votre nouveau {model} chez Maroc Telecom et recevez des écouteurs sans fil offerts ! Offre limitée. Tous les détails sur {link}",
            "Le nouveau {model} est arrivé chez Maroc Telecom. Profitez de son autonomie record pour seulement {price}. Découvrez-le sur {link}"
        ],

        # --- Persona: OPPORTUNITE_AchatSmartphone (Ton: Valeur, Accessibilité) ---
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatSmartphone","APPLE"): [
            "L'iPhone reconditionné est disponible chez Maroc Telecom ! À partir de {price} seulement, avec une garantie de 12 mois. Commandez-le vite sur {link}",
            "Passez à l'iPhone avec Maroc Telecom ! Découvrez nos offres sur l' {model} {capacity} à partir de {price}. L'expérience Apple à un prix accessible. {link}",
            "Offre spéciale Maroc Telecom : l' {model} à un prix exceptionnel. Idéal pour découvrir l'écosystème Apple. Tous les détails sur {link}",
            "Pour votre premier smartphone, choisissez la sécurité Apple avec Maroc Telecom. L' {model} est à {price} avec {bonus_data} de bienvenue. {link}",
            "Un iPhone pour tous chez Maroc Telecom ! Profitez de nos offres sur les modèles neufs ou reconditionnés. L' {model} dès {price}. {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatSmartphone","SAMSUNG"): [
            "Les bonnes affaires, c'est chez Maroc Telecom ! Profitez du {model} à partir de {price} au lieu de {old_price}. Commandez-le sur {link}",
            "Passez à la vitesse supérieure avec Maroc Telecom ! Découvrez le {model} {capacity} à seulement {price}. Plus d'infos sur {link}",
            "Offre Spéciale Maroc Telecom : Le {model} est à un prix exceptionnel. Idéal pour WhatsApp et vos applis préférées. {link}",
            "Pour votre premier smartphone, choisissez la fiabilité Samsung. Le Galaxy {model} est à {price} avec {bonus_data} offerts par Maroc Telecom. {link}",
            "Simplifiez-vous la vie avec le {model} de Maroc Telecom. Accédez à tous vos services en ligne pour seulement {price}. Livraison gratuite sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatSmartphone","HONOR"): [
            "Les bonnes affaires continuent chez Maroc Telecom ! Le {model} {capacity} à {price} avec livraison gratuite. Commandez dès maintenant sur {link}",
            "Un smartphone performant et accessible : le {model} à {price}, recommandé par Maroc Telecom. Parfait pour débuter en 4G. {link}",
            "Maroc Telecom vous offre {bonus_data}/mois pendant 3 mois à l'achat d'un Pack Smartphone HONOR à partir de {price}. Rendez-vous en agence.",
            "Offre imbattable chez Maroc Telecom sur le {model} : {price} avec 3 mois de Pass Réseaux Sociaux offerts. Plus d'infos sur {link}",
            "Découvrez la performance HONOR à petit prix avec Maroc Telecom. Le {model} {capacity} est à {price}. Commandez-le sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatSmartphone","OPPO"): [
            "Les bonnes affaires continuent chez Maroc Telecom ! Le {model} {capacity} à {price} avec livraison gratuite. Commandez dès maintenant sur {link}",
            "Un smartphone performant et accessible : le {model} à {price}, recommandé par Maroc Telecom. Parfait pour débuter en 4G. {link}",
            "Maroc Telecom vous offre {bonus_data}/mois pendant 3 mois à l'achat d'un Pack Smartphone OPPO à partir de {price}. Rendez-vous en agence.",
            "Offre imbattable chez Maroc Telecom sur le {model} : {price} avec 3 mois de Pass Réseaux Sociaux offerts. Plus d'infos sur {link}",
            "Découvrez la performance OPPO à petit prix avec Maroc Telecom. Le {model} {capacity} est à {price}. Commandez-le sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatSmartphone","TECNO"): [
            "Les bonnes affaires continuent chez Maroc Telecom ! Le {model} {capacity} à {price} avec livraison gratuite. Commandez dès maintenant sur {link}",
            "Un smartphone performant et accessible : le {model} à {price}, recommandé par Maroc Telecom. Parfait pour débuter en 4G. {link}",
            "Maroc Telecom vous offre {bonus_data} par mois pendant 3 mois à l'achat d'un Pack Smartphone TECNO à partir de {price}. Rendez-vous en agence.",
            "Offre imbattable chez Maroc Telecom sur le {model} : {price} avec 3 mois de Pass Réseaux Sociaux offerts. Plus d'infos sur {link}",
            "Découvrez la performance TECNO à petit prix avec Maroc Telecom. Le {model} {capacity} est à {price}. Commandez-le sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_AchatSmartphone","XIAOMI"): [
            "Smartphones jusqu'à -70%' chez Maroc Telecom ! Le {model} {capacity} est à {price} seulement. Profitez-en sur {link}",
            "Un smartphone performant et accessible : le {model} à {price}, recommandé par Maroc Telecom. Parfait pour débuter en 4G. {link}",
            "Les bonnes affaires continuent chez Maroc Telecom ! Le {model} à {price} avec livraison gratuite. Commandez dès maintenant sur {link}",
            "Offre imbattable chez Maroc Telecom sur le {model} : {price} avec 3 mois de Pass Réseaux Sociaux offerts. Plus d'infos sur {link}",
            "Découvrez la performance Xiaomi à petit prix avec Maroc Telecom. Le {model} {capacity} est à {price}. Commandez-le sur {link}"
        ],

        # --- Persona: OPPORTUNITE_PerformanceSmartphone (Ton: Puissance, Upgrade) ---
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_PerformanceSmartphone","APPLE"): [
            "Votre usage intensif mérite un équipement d'exception. Découvrez l'{model} chez Maroc Telecom et vivez une expérience sans latence. Dès {price_monthly}/mois sur {link}",
            "Vous êtes un Power User. Votre téléphone actuel vous freine. Libérez votre potentiel avec la puissance de l'{model} sur le réseau Maroc Telecom. {link}",
            "Ne laissez plus une batterie faible gâcher votre expérience. Il est temps de passer à l'{model} {capacity}. Offre de reprise disponible chez Maroc Telecom.",
            "Offre de reprise spéciale Maroc Telecom ! Échangez votre ancien mobile et obtenez une réduction immédiate sur l'achat d'un nouvel {model}. Voir conditions en agence.",
            "En tant que client à haute valeur, Maroc Telecom vous propose des facilités de paiement exclusives pour l'achat de l'{model} {capacity}. Rendez-vous sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_PerformanceSmartphone","SAMSUNG"): [
            "Votre performance est limitée par votre smartphone ? Passez au {model} {capacity} avec Maroc Telecom et redécouvrez la vitesse. Dès {price}. {link}",
            "Pour vos usages les plus exigeants, le {model} est le choix de la performance. Offre spéciale Maroc Telecom à {price}. {link}",
            "Libérez votre potentiel ! L'écran surpuissant du Galaxy {model} vous attend chez Maroc Telecom. Plus d'infos sur {link}",
            "Ne vous contentez plus de moins. Le {model}, recommandé par Maroc Telecom, est conçu pour les utilisateurs qui exigent le meilleur. {link}",
            "Votre passion pour la technologie mérite le meilleur. Maroc Telecom vous propose le {model} avec un bonus de data exclusif. {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_PerformanceSmartphone","HONOR"): [
            "Votre usage intensif mérite un équipement à la hauteur. Découvrez le {model} et sa puce surpuissante chez Maroc Telecom. Dès {price}. {link}",
            "Vous êtes un Power User. Votre téléphone actuel vous ralentit. Passez au {model} {capacity} avec Maroc Telecom pour une fluidité absolue. {link}",
            "Ne laissez plus une batterie faible limiter votre journée. Le {model} est conçu pour une autonomie extrême. Découvrez-le chez Maroc Telecom. {link}",
            "Offre de reprise spéciale Maroc Telecom : échangez votre ancien mobile et obtenez une réduction immédiate sur le {model}. Conditions en agence.",
            "En tant que client à haute valeur, Maroc Telecom vous propose le {model} {capacity} avec des facilités de paiement. Rendez-vous sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_PerformanceSmartphone","OPPO"): [
            "Votre usage intensif mérite un équipement à la hauteur. Découvrez le {model} et sa puce surpuissante chez Maroc Telecom. Dès {price}. {link}",
            "Vous êtes un Power User. Votre téléphone actuel vous ralentit. Passez au {model} {capacity} avec Maroc Telecom pour une fluidité absolue. {link}",
            "Ne laissez plus une batterie faible limiter votre journée. Le {model} est conçu pour une autonomie extrême. Découvrez-le chez Maroc Telecom. {link}",
            "Offre de reprise spéciale Maroc Telecom : échangez votre ancien mobile et obtenez une réduction immédiate sur le {model}. Conditions en agence.",
            "En tant que client à haute valeur, Maroc Telecom vous propose le {model} {capacity} avec des facilités de paiement. Rendez-vous sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_PerformanceSmartphone","TECNO"): [
            "Votre usage intensif mérite un équipement à la hauteur. Découvrez le {model} et sa puce surpuissante chez Maroc Telecom. Dès {price}. {link}",
            "Vous êtes un Power User. Votre téléphone actuel vous ralentit. Passez au {model} {capacity} avec Maroc Telecom pour une fluidité absolue. {link}",
            "Ne laissez plus une batterie faible limiter votre journée. Le {model} est conçu pour une autonomie extrême. Découvrez-le chez Maroc Telecom. {link}",
            "Offre de reprise spéciale Maroc Telecom : échangez votre ancien mobile et obtenez une réduction immédiate sur le {model}. Conditions en agence.",
            "En tant que client à haute valeur, Maroc Telecom vous propose le {model} {capacity} avec des facilités de paiement. Rendez-vous sur {link}"
        ],
        ("OPPORTUNITE_Achat_Equipement","OPPORTUNITE_PerformanceSmartphone","XIAOMI"): [
            "Votre usage intensif mérite un équipement à la hauteur. Découvrez le {model} et sa puce surpuissante chez Maroc Telecom. Dès {price}. {link}",
            "Vous êtes un Power User. Votre téléphone actuel vous ralentit. Passez au {model} {capacity} avec Maroc Telecom pour une fluidité absolue. {link}",
            "Ne laissez plus une batterie faible limiter votre journée. Le {model} est conçu pour une autonomie extrême. Découvrez-le chez Maroc Telecom. {link}",
            "Offre de reprise spéciale Maroc Telecom : échangez votre ancien mobile et obtenez une réduction immédiate sur le {model}. Conditions en agence.",
            "En tant que client à haute valeur, Maroc Telecom vous propose le {model} {capacity} avec des facilités de paiement. Rendez-vous sur {link}"
        ],
        
        # ===================================================================
        # Autres Familles
        # ===================================================================
        ("USAGE_Internet","OPPORTUNITE_InternetCher","*3"): [
            "Arrêtez de payer trop cher pour surfer ! Avec Maroc Telecom, profitez de {volume} pendant {validity} pour seulement {price}. Composez {cta} pour en profiter.",
            "Info Maroc Telecom : Saviez-vous que votre usage internet vous coûterait moins cher ? {volume} pour {price} avec le Pass Internet. Composez {cta} pour activer.",
            "Exclusif Maroc Telecom : ne gaspillez plus votre recharge ! Le Pass {cta} vous offre {volume} de data pour {price}. La solution intelligente pour surfer. Composez {cta}.",
            "Votre consommation data mérite un meilleur prix. Maroc Telecom vous propose le Pass {offer_name} : {volume} valables {validity} à {price}. Composez {cta} pour recharger.",
            "Payez moins, surfez plus avec Maroc Telecom ! {volume} de connexion 4G+ pour seulement {price}. C'est le moment de passer au Pass. Composez {cta}."
        ],
        ("USAGE_Internet","PROFIL_Internet","*3"): [
            "La vitesse vous appelle ! Avec Maroc Telecom, profitez de {volume} en très haut débit 4G+ pour {price}. Composez {cta} pour activer.",
            "Pour nos navigateurs experts : votre Pass Internet est boosté par Maroc Telecom ! Profitez de {volume} pour {price}. Offre valable jusqu'au {deadline}. Rechargez sur {link}",
            "Ne soyez jamais à court de data. Avec Maroc Telecom, rechargez votre Pass Internet à tout moment et profitez de {volume} pour {price}. Composez {cta}.",
            "Votre passion pour Internet mérite le meilleur réseau. Avec Maroc Telecom, téléchargez et streamez sans interruption. {volume} pour {price}. Composez {cta}.",
            "Flash promo Maroc Telecom ! Le Pass Internet de {price} vous offre {volume} valables {validity}. Composez {cta} pour en profiter."
        ],
        ("USAGE_Internet","PROFIL_ReseauxSociaux","*6"): [
            "Vos stories n'attendent pas ! Avec le Pass Réseaux Sociaux de Maroc Telecom, partagez sans compter sur TikTok et Instagram. {volume} pour {price}. Composez {cta}.",
            "Restez connecté à vos communautés avec Maroc Telecom ! Le Pass {cta} vous donne {volume} pour {price}. Composez {cta} pour l'activer.",
            "Exclusif Social Media Addict ! Pour votre prochaine recharge Pass {cta}, Maroc Telecom vous offre un bonus de {bonus_volume} valable sur YouTube. Rechargez sur {link}",
            "Ne manquez plus aucune notification. Le Pass {cta} de Maroc Telecom est maintenant valable {validity} pour {price}. Composez {cta} pour activer.",
            "Partagez sans limites avec Maroc Telecom ! Le Pass {offer_name} vous donne {volume} sur vos réseaux préférés. Composez {cta} pour seulement {price}."
        ],
        ("USAGE_Divertissement","PROFIL_Gamer","*88"): [
            "Le champ de bataille vous attend ! Rechargez avec le Pass Gaming {cta} de Maroc Telecom et bénéficiez d'une connexion optimisée. {volume} pour {price}. Composez {cta}.",
            "Niveau supérieur ! Le Pass Gaming {cta} de Maroc Telecom inclut des bonus exclusifs pour le jeu '{game_name}'. Composez {cta} pour en profiter.",
            "Ne soyez jamais à court de crédits en jeu. Utilisez votre recharge *9 de Maroc Telecom pour acheter des devises et recevez un bonus. Composez {cta} pour le gaming.",
            "Pour les gamers exigeants, Maroc Telecom recommande de combiner le Pass {cta} avec un Pass Internet *3. Composez {cta} pour l'activer.",
            "Votre fidélité de gamer est récompensée par Maroc Telecom ! Recevez un item exclusif avec votre prochaine recharge {cta} de {price}. Composez {cta}."
        ],
        ("USAGE_Divertissement","PROFIL_ServicesPremium","*9"): [
            "Votre divertissement, à votre façon. Avec Maroc Telecom, payez vos abonnements STARZPLAY et Anghami facilement avec la recharge {cta}. Plus d'infos sur {link}",
            "Nouveau sur {service_name} ! Découvrez la dernière saison de '{serie_name}'. Abonnez-vous avec votre recharge {cta} de Maroc Telecom et le premier mois est à -50%.",
            "Ne manquez aucun match de la Botola ! Abonnez-vous à MT FOOT avec votre recharge {cta} de Maroc Telecom. Composez {cta}.",
            "Simplifiez vos paiements digitaux avec Maroc Telecom. La recharge {cta} est la solution la plus sûre et rapide pour vos abonnements. {link}",
            "En tant qu'utilisateur premium, Maroc Telecom vous invite à découvrir notre nouveau service de livres audio. Premier livre offert via {cta}."
        ],
        ("USAGE_Mixte","PROFIL_Econome","*5"): [
            "Maîtrisez votre budget avec Maroc Telecom ! Avec le Pass {cta}, profitez de {minutes} d'appels OU {volume} de data pour seulement {price}. Composez {cta}.",
            "Chaque dirham compte. Maroc Telecom vous propose ses offres à {price} : {minutes} d'appels ou {volume} de réseaux sociaux. Composez {cta}.",
            "Nouveau chez Maroc Telecom ! Le Pass {cta} de {price} vous donne maintenant {minutes} + {volume}. Plus de valeur pour le même prix. Composez {cta}.",
            "Pour votre budget, chaque recharge est importante. Profitez de la recharge x15 de {price} et obtenez {bonus_credit} de bonus avec Maroc Telecom. Composez {cta}.",
            "Continuez à communiquer sans vous ruiner. Maroc Telecom vous propose le Pass {cta} : {minutes} ou {volume} à {price}. Composez {cta}."
        ],
        ("USAGE_Mixte","PROFIL_Flexible","*5"): [
            "Voix ou Data ? Ne choisissez plus ! Le Pass Tout en Un {cta} de Maroc Telecom s'adapte à vos besoins. {minutes} OU {volume} pour {price}. Composez {cta}.",
            "Pour vos besoins variés, Maroc Telecom vous recommande le Pass *2. Profitez du meilleur des deux mondes pour {price}. Composez {cta}.",
            "Votre style de communication est unique. Avec le service Switch de Maroc Telecom, convertissez votre crédit selon vos envies. Composez {cta}.",
            "Un imprévu ? Avec le Pass {cta} de Maroc Telecom, votre recharge n'est jamais perdue. {minutes} ou {volume} pour {price}. Composez {cta}.",
            "En tant qu'utilisateur flexible, Maroc Telecom vous offre un bonus de 10% de crédit en plus sur le Pass {cta}. Composez {cta}."
        ],
        ("USAGE_Mixte","PROFIL_Professionnel","Forfait_Business"): [
            "Concentrez-vous sur votre métier, Maroc Telecom s'occupe du reste. Passez à un forfait Business et recevez une seule facture. Plus d'infos sur {link}",
            "Votre usage professionnel mérite une offre adaptée. Découvrez nos forfaits Business Control avec appels illimités. Contactez le 777.",
            "Optimisez vos coûts professionnels avec Maroc Telecom. Nos forfaits postpayés vous offrent des tarifs plus avantageux. Détails sur {link}",
            "Valorisez votre image professionnelle avec une ligne dédiée. Maroc Telecom vous offre la portabilité de votre numéro Jawal vers un forfait Business.",
            "Votre téléphone double SIM est un outil de travail. Associez-lui une offre Business de Maroc Telecom. Découvrez nos offres sur {link}"
        ],
        ("USAGE_Roaming","OPPORTUNITE_VoyageurOccasionnel","*78"): [
            "Bienvenue à l'étranger ! Maroc Telecom vous recommande d'activer un Pass Roaming pour éviter les frais standards. Le Pass {cta} vous offre {volume} pour {price}. Composez {cta}.",
            "Ne laissez pas une facture de roaming gâcher vos vacances. Avec les pass Maroc Telecom à partir de {price}, contrôlez votre budget. Composez {cta}.",
            "Le saviez-vous ? Un Pass Roaming Maroc Telecom peut vous faire économiser jusqu'à 80%. Pour votre voyage, activez le Pass {offer_name}. Composez {cta}.",
            "Pour votre premier voyage connecté, Maroc Telecom vous offre 20% de bonus sur votre premier Pass Roaming. Composez {cta} avant de partir !",
            "Restez connecté avec vos proches même à l'étranger. Activez un Pass Roaming Maroc Telecom et partagez vos meilleurs moments. Composez {cta}."
        ],
        ("USAGE_Roaming","PROFIL_VoyageurComplet","*7"): [
            "Voyagez l'esprit tranquille avec Maroc Telecom. Le Pass Roaming Multiservices {cta} est votre compagnon idéal : {volume}, {minutes} et {sms}. Composez {cta}.",
            "Nouveau ! Le Pass Roaming {cta} de Maroc Telecom inclut maintenant les appels entrants gratuits dans {destinations}. Offre valable jusqu'au {deadline}. Composez {cta}.",
            "En tant que voyageur complet, Maroc Telecom vous offre 30 minutes d'appels supplémentaires sur votre prochaine activation du Pass {cta} de {price}. Composez {cta}.",
            "Data, appels, SMS : tout ce dont vous avez besoin à l'étranger. Simplifiez-vous la vie avec le Pass Multiservices de Maroc Telecom. Rechargez sur {link}",
            "Optimisez vos communications en roaming. Le Pass {cta} de Maroc Telecom vous offre le meilleur des trois mondes à un tarif préférentiel. Composez {cta}."
        ],
        ("USAGE_SMS","PROFIL_Sms","*1"): [
            "Nouveau ! Profitez de l'offre Pass SMS {cta} de Maroc Telecom : {sms_count} SMS vers tous les opérateurs pour seulement {price}. Composez {cta}.",
            "La communication par SMS reste essentielle. Jusqu'au {deadline}, Maroc Telecom double vos SMS avec le Pass {cta} de {price}. Composez {cta}.",
            "Votre fidélité à notre service SMS est récompensée ! Pour votre prochaine recharge {cta}, Maroc Telecom vous offre 100 SMS supplémentaires. Composez {cta}.",
            "Exclusif Maroc Telecom ! Rechargez {price} avec le Pass {cta} et envoyez {sms_count} SMS pendant {validity}. Composez {cta}.",
            "Besoin d'envoyer plus de messages ? Le Pass {offer_name} de Maroc Telecom vous offre {sms_count} SMS pour {price}. Composez {cta}."
        ],
        ("USAGE_Voix","PROFIL_CommunicantInternational","*4"): [
            "Gardez le contact avec le monde entier grâce à Maroc Telecom. Avec le Pass International {cta}, appelez l'{zone} à des tarifs imbattables. Composez {cta}.",
            "Spécial Europe ! Ce mois-ci, votre Pass International {cta} de Maroc Telecom vous donne {minutes} d'appels vers la France et l'Espagne. Composez {cta}.",
            "Vos appels vers le Canada et les USA à prix réduit avec Maroc Telecom. Activez le Pass International {cta} et bénéficiez de nos meilleurs tarifs. Composez {cta}.",
            "Nouveau ! Le Pass International {cta} de Maroc Telecom inclut 30 minutes vers le Moyen-Orient pour {price}. Plus d'infos sur {link}",
            "Ne perdez plus le contact avec vos proches à l'étranger. Le Pass {cta} de Maroc Telecom est la solution la plus économique. Composez {cta}."
        ],
        ("USAGE_Voix","PROFIL_FideleOnNet","*22"): [
            "Votre réseau est chez IAM, profitez-en ! Activez l'option illimitée Maroc Telecom et appelez tous les numéros IAM sans compter. Composez {cta}.",
            "Merci pour votre fidélité à notre réseau ! Maroc Telecom vous offre un bonus de 2H d'appels vers IAM sur votre prochaine recharge. Composez {cta}.",
            "Vos appels on-net sont les plus importants. C'est pourquoi le Pass {cta} de Maroc Telecom vous offre toujours le meilleur tarif. Composez {cta}.",
            "La force de notre réseau, c'est vous. Continuez à communiquer avec vos proches sur Maroc Telecom avec nos offres généreuses. Composez {cta}.",
            "Offre spéciale Fidélité Maroc Telecom ! Ce weekend, tous vos appels vers les numéros IAM sont à -50%. Composez {cta} pour en profiter."
        ],
        ("USAGE_Voix","PROFIL_VoixNational","*22"): [
            "Vos appels vers tous les opérateurs nationaux à prix réduit avec Maroc Telecom ! Avec le Pass {cta}, profitez de {minutes} pour {price}. Composez {cta}.",
            "Parlez plus longtemps. Le Pass National {cta} de Maroc Telecom vous offre maintenant {minutes} d'appels. Communiquez en toute sérénité. Composez {cta}.",
            "Parce que vous appelez beaucoup, Maroc Telecom vous offre un bonus de 30 minutes sur votre prochaine recharge Pass {cta} de {price}. Composez {cta}.",
            "Exclusif Maroc Telecom ! Cette semaine, votre Pass {cta} de {price} est valable {validity} . Composez {cta}.",
            "Ne vous souciez plus des minutes. Le Pass {offer_name} de Maroc Telecom vous offre {minutes} vers tous les réseaux pour {price}. Composez {cta}."
        ]
    }
    # Resolve best template list for the triplet
    key_exact = (famille, persona, brand_key if not cta else cta)
    key_persona_default = (famille, persona, "__DEFAULT__")
    key_family_brand = (famille, "__ALL__", brand_key if not cta else cta)
    key_family_default = (famille, "__ALL__", "__DEFAULT__")
    key_global_default = ("__ALL__", "__ALL__", "__DEFAULT__")

    templates = (
        TEMPLATES.get(key_exact)
        or TEMPLATES.get(key_persona_default)
        or TEMPLATES.get(key_family_brand)
        or TEMPLATES.get(key_family_default)
        or TEMPLATES.get(key_global_default)
    )

    if isinstance(templates, (tuple, set)):
        templates = list(templates)
    elif isinstance(templates, str):
        templates = [templates]

    # Deterministic rotation across variants via session state counter
    data = base_fields()
    triplet_key = f"template_idx_{famille}_{persona}_{cta or brand_key or 'default'}"
    if triplet_key not in st.session_state:
        st.session_state[triplet_key] = 0

    template_count = max(len(templates), 1)
    idx = st.session_state[triplet_key] % template_count
    st.session_state[triplet_key] = (idx + 1) % template_count
    text = templates[idx].format(**data)
    return " ".join(text.split())


def live_llm(
    llm_input: Dict,
    base_url: str,
    api_key: str,
    model_id: str = DEFAULT_LIVE_LLM_MODEL_ID,
) -> str:
    system_prompt = (
        "Tu es un rédacteur sms marketing pour Maroc Telecom (IAM). À partir de l’INPUT JSON fourni (profil_client, offer_context, promo_context, cta, deadline, links),  écris UN SEUL SMS promotionnel en français en respectant STRICTEMENT : 1) ≤ 480 caractères ; 2) n’utiliser QUE les chiffres/prix/volumes/durées/destinations présents dans l’input ; 3) Termine TOUJOURS par un appel à l'action clair (code USSD , lien) ;5) tonalité fluide , naturelle et percutante ; 6) Le message ne doit contenir AUCUNE NOTE ; 7)Réponds SEULEMENT en français ; 8) Ne mentionne pas le nom du persona"
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(llm_input, ensure_ascii=False)},
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 140,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def fetch_latest_chat_id(token: Optional[str]) -> Tuple[bool, str]:
    token = token or TELEGRAM_BOT_TOKEN
    if not token:
        return False, "Missing bot token."
    resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
    data = resp.json()
    if not data.get("ok"):
        return False, f"getUpdates failed: {data}"
    for update in reversed(data.get("result", [])):
        message = update.get("message") or update.get("channel_post")
        if message and message.get("chat", {}).get("id"):
            return True, str(message["chat"]["id"])
    return False, "No chat id found. Press /start in Telegram and try again."

def send_to_telegram(token: Optional[str], sms: str, chat_id: str) -> Tuple[bool, str]:
    token = token or TELEGRAM_BOT_TOKEN
    if not token:
        return False, "Missing bot token."
    if not chat_id:
        return False, "Missing chat id."

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": sms},
        timeout=10,
    )
    data = resp.json()
    if not data.get("ok"):
        return False, f"sendMessage failed: {data}"
    return True, chat_id

def extract_offer_metrics(offer_context: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    price = offer_context.get("prix_dh")
    volume = offer_context.get("volume")
    validity = offer_context.get("validite")
    if price is not None:
        try:
            price = f"{float(price):.0f} MAD"
        except Exception:
            price = f"{price} MAD"
    return price, volume, validity

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
st.markdown("<div class='layout-stack'>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>Maroc Telecom Marketing SMS Platform</h1>
        <p> Marketing SMS interface, powered by customer segmentation, RAG and finetuned LLM message generation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='content-container'>", unsafe_allow_html=True)

state_famille = st.session_state.get("famille_select")
state_persona = st.session_state.get("persona_select")

# Enterprise Grid Layout
st.markdown("<div class='enterprise-grid'>", unsafe_allow_html=True)

# Top Row: Audience Configuration and Contextual Insights
top_row = st.columns([1, 1], gap="large")

# Left: Audience Configuration
with top_row[0]:
    st.markdown("""
    <div class='premium-card'>
        <div class='card-header'>
            <div class='card-icon'>📡</div>
            <div>
                <h3 class='card-title'>Audience Configuration</h3>
                <p class='card-subtitle'>Choose a usage type, persona and CTA (offer code) or handset brand (for smartphone offers), to get a custom marketing sms.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    selection = dependent_dropdowns()
    st.markdown("</div>", unsafe_allow_html=True)  # Close premium-card
    
# Right: Contextual Insights
with top_row[1]:
    st.markdown("""
    <div class='premium-card'>
        <div class='card-header'>
            <div class='card-icon'>🧠</div>
            <div>
                <h3 class='card-title'>Contextual Insights</h3>
                <p class='card-subtitle'>Brief description to help you understand the targeted audience you chosed.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    selected_famille = selection.famille if selection else state_famille
    selected_persona = selection.persona if selection else state_persona
    render_contextual_insight(selected_famille, selected_persona)
    st.markdown("</div>", unsafe_allow_html=True)  # Close premium-card

## ---------------- Center section with LLM Mode & Animated Generate Button (no page reset) ----------------
st.markdown("<div style='margin: 2rem 0;'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div class='premium-card llm-mode-card'>
            <div class='llm-heading'>
                <span class='llm-icon'>🤖</span>
                <div>
                    <h3>LLM Mode</h3>
                    <p>Select the engine that will craft your custom SMS.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    <style>
    /* Center the whole radio group */
    div[role="radiogroup"]{
      display:flex !important;
      justify-content:center !important;
      align-items:center !important;
      gap:14px;               /* spacing between options */
      flex-wrap:nowrap !important;
    }
    /* Make each option render inline (not stacked) */
    div[role="radiogroup"] > label{
      display:inline-flex !important;
      align-items:center;
      padding:10px 16px;
      border-radius:9999px;
      border:1px solid rgba(61,165,255,0.18);
      background:rgba(15,20,40,0.95);
      cursor:pointer;
    }
    </style>
    """, unsafe_allow_html=True)
    
    llm_mode = st.radio(
        " ",
        ["Mock", "Live inference"],
        key="llm_mode",
        horizontal=True,
    )
    mode_details = {
        "Mock": {
            "key": "mock",
            "title": "Mock mode",
            "description": "Maroc Telecom marketing sms templates with deterministic rotation for fast 0$ generation.",
        },
        "Live inference": {
            "key": "live",
            "title": "Live inference",
            "description": "Remote hosted LLM (finetuned , 4bit  quantized) generating pine-point customised marketing sms for the customer audience you are targeting.",
        },
    }
    selected_detail = mode_details.get(llm_mode, mode_details["Mock"])
    st.markdown(
        f"""
        <div class='llm-mode-info {selected_detail["key"]}'>
            <div class='llm-info-body'>
                <span class='llm-dot {selected_detail["key"]}'></span>
                <div>
                    <strong>{selected_detail["title"]}</strong>
                    <p>{selected_detail["description"]}</p>
                </div>
            </div>
        </div>
        <p class='llm-caption'></p>
        </div>
        <style>
        .llm-mode-card {{
            text-align: left;
            display: flex;
            flex-direction: column;
            gap: 1.4rem;
            padding: 1.75rem 1.9rem 1.6rem;
            background: linear-gradient(170deg, rgba(32,42,65,0.88), rgba(15,16,24,0.78));
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 24px 48px rgba(9, 12, 20, 0.45);
        }}
        .llm-header {{display:flex; flex-direction:column; gap:1.05rem;}}
        .llm-chip {{
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.7);
            background: rgba(91, 115, 255, 0.16);
            border: 1px solid rgba(91,115,255,0.35);
            width: fit-content;
        }}
        .llm-heading {{display:flex; gap:1rem; align-items:flex-start;}}
        .llm-icon {{
            font-size: 1.85rem;
            display:flex;
            align-items:center;
            justify-content:center;
            width:3.1rem;
            height:3.1rem;
            border-radius: 20px;
            background: rgba(91,115,255,0.15);
            box-shadow: inset 0 0 0 1px rgba(91,115,255,0.35);
        }}
        .llm-heading h3 {{
            margin: 0;
            font-size: 1.25rem;
            color: var(--text-primary);
        }}
        .llm-heading p {{
            margin: 0.35rem 0 0;
            color: rgba(255,255,255,0.65);
            font-size: 0.92rem;
            line-height: 1.45;
        }}
        .llm-mode-wrapper {{
            width: 100%;
            display: flex;
            justify-content: center;
        }}
        .llm-mode-card .llm-mode-wrapper div[data-testid="stHorizontalBlock"] {{
            width: auto;
            padding: 0.35rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            display: flex;
            justify-content: center;
        }}
        .llm-mode-card .llm-mode-wrapper div[role="radiogroup"] {{
            display: flex;
            gap: 0.4rem;
            justify-content: center;
        }}
        .llm-mode-card .llm-mode-wrapper div[role="radio"] {{
            min-width: 150px;
            padding: 0.65rem 1.1rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.25s ease;
            cursor: pointer;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }}
        .llm-mode-card .llm-mode-wrapper div[role="radio"][aria-checked="true"] {{
            background: linear-gradient(135deg,#6366F1,#3B82F6);
            border-color: rgba(99,102,241,0.7);
            box-shadow: 0 12px 34px rgba(59,130,246,0.32);
        }}
        .llm-mode-card .llm-mode-wrapper div[role="radio"] p {{
            margin: 0;
            text-align: center;
            font-weight: 600;
            font-size: 0.95rem;
            color: rgba(255,255,255,0.82);
        }}
        .llm-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-top: 0.3rem;
            box-shadow: 0 0 0 4px rgba(255,255,255,0.05);
        }}
        .llm-dot.mock {{background: linear-gradient(135deg,#F97316,#FB923C);}} 
        .llm-dot.live {{background: linear-gradient(135deg,#22C55E,#0EA5E9);}} 
        .llm-mode-info {{
            margin-top: 1.1rem;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
        }}
        .llm-mode-info.mock {{box-shadow: inset 0 0 0 1px rgba(249,115,22,0.35);}} 
        .llm-mode-info.live {{box-shadow: inset 0 0 0 1px rgba(34,197,94,0.35);}} 
        .llm-info-body {{
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
        }}
        .llm-info-body strong {{
            display: block;
            font-size: 0.98rem;
            margin-bottom: 0.2rem;
            color: var(--text-primary);
        }}
        .llm-info-body p {{
            margin: 0;
            font-size: 0.86rem;
            color: rgba(255,255,255,0.65);
            line-height: 1.45;
        }}
        .llm-caption {{
            margin: 0.6rem 0 0;
            font-size: 0.82rem;
            color: rgba(255,255,255,0.55);
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if llm_mode == "Live inference":
        st.link_button(
            "🧪 Open Colab notebook (run all, get ngrok URL)",
            "https://colab.research.google.com/drive/156lcy9KyfdHSoz0U8-Wde_gDf_uC3Qkp?usp=sharing",
            use_container_width=True,
        )
        col_a, col_b = st.columns([2, 1])
        stored_url = st.session_state.get("LIVE_LLM_URL", "")
        stored_model = st.session_state.get("LIVE_LLM_MODEL_ID", DEFAULT_LIVE_LLM_MODEL_ID)
        with col_a:
            user_base_raw = st.text_input(
                "Paste vLLM URL (ngrok) you got from colab",
                value=stored_url,
                placeholder="https://xxxx-xx-xx-xx.ngrok-free.app",
            )
        with col_b:
            st.text_input(
                "Model id",
                value=stored_model,
                disabled=True,
                help="Finetuned marketing SMS expert model ",
            )
        model_id_ui = stored_model

        validate_clicked = st.button("Validate endpoint")
        if validate_clicked:
            normalized = (user_base_raw or "").strip()
            if not normalized:
                st.warning("Paste the public base URL before validating.")
            else:
                normalized = normalized.rstrip("/")
                parsed = urlparse(normalized)
                if parsed.scheme not in ("http", "https"):
                    st.warning("Only http(s) URLs are supported.")
                elif parsed.netloc and "ngrok" not in parsed.netloc.lower():
                    st.warning("The URL should be an ngrok public base.")
                else:
                    try:
                        health_resp = requests.get(f"{normalized}/health", timeout=10)
                        health_resp.raise_for_status()
                        models_resp = requests.get(f"{normalized}/v1/models", timeout=10)
                        models_resp.raise_for_status()
                        models_json = models_resp.json()
                        first_model = models_json.get("data", [{}])[0].get("id", "unknown")
                        session_model = (model_id_ui or DEFAULT_LIVE_LLM_MODEL_ID).strip() or DEFAULT_LIVE_LLM_MODEL_ID
                        st.session_state["LIVE_LLM_URL"] = normalized
                        st.session_state["LIVE_LLM_MODEL_ID"] = session_model
                        LIVE_LLM_URL = normalized
                        st.success(f"Connected to vLLM endpoint . LLM used : {first_model} ")
                    except Exception as exc:
                        st.error(f"Validation failed: {exc}")

    # Placeholder that we will mutate (ready -> fetching -> success -> ready) without rerun
    btn_placeholder = st.empty()

    def render_ready():
        btn_placeholder.markdown(
    """
    <button id='gen-btn' style='
        width:100%; padding:16px 24px; border:2px solid #ff2d55; border-radius:14px; cursor:pointer;
        background:transparent; color:#ff2d55; font-weight:600; font-size:15px;
        display:flex; align-items:center; justify-content:center; gap:10px; position:relative; overflow:hidden;
        transition:all .35s cubic-bezier(.4,0,.2,1);
    '>
        <span style='font-size:18px'>🚀</span>
        <span>Personalized SMS</span>
    </button>
    <script>
    const btn = window.parent.document.getElementById('gen-btn');
    if(btn){
      btn.onclick = function(){
        window.parent.postMessage({type:'gen_click'}, '*');
      }
    }
    </script>
    """,
    unsafe_allow_html=True,
)


    def render_fetching():
        btn_placeholder.markdown(
            """
            <div style='width:100%; padding:16px 24px; border:none; border-radius:14px;
                background:linear-gradient(135deg,#1a70ff,#4a94ff); color:#fff; font-weight:600; font-size:15px;
                box-shadow:0 8px 32px rgba(26,112,255,.45); position:relative; overflow:hidden;'
                id='fetching-tile'>
                <div style='display:flex; align-items:center; justify-content:center; gap:12px;'>
                    <div style='width:22px;height:22px;border:3px solid rgba(255,255,255,0.35);border-top:3px solid #fff;border-radius:50%;animation:spin 1s linear infinite'></div>
                    <span>🔍 Qdrant RAG fetching Maroc Telecom catalogs...</span>
                </div>
            </div>
            <style>
              #fetching-tile{animation:fetchingPulse 2s ease-in-out infinite}
              @keyframes fetchingPulse{0%,100%{transform:scale(1.00)}50%{transform:scale(1.03)}}
              @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_llm_generating():
        btn_placeholder.markdown(
            """
            <div style='width:100%; padding:18px 26px; border:none; border-radius:14px;
                background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; font-weight:600; font-size:15px;
                box-shadow:0 12px 36px rgba(99,102,241,.45); position:relative; overflow:hidden;'
                id='llm-tile'>
                <div style='display:flex; align-items:center; justify-content:center; gap:14px;'>
                    <div style='position:relative;width:24px;height:24px;'>
                        <div style='width:24px;height:24px;border-radius:50%;border:3px solid rgba(255,255,255,0.35);'></div>
                        <div style='position:absolute;top:-6px;left:-6px;width:36px;height:36px;border-radius:50%;border:3px solid rgba(255,255,255,0.18);animation:llmPulse 1.8s ease-in-out infinite;'></div>
                    </div>
                    <span>✨ Mistral finetuned LLM is generating your custom marketing SMS...</span>
                </div>
            </div>
            <style>
              #llm-tile{animation:llmGlow 2.4s ease-in-out infinite}
              @keyframes llmGlow{0%,100%{box-shadow:0 12px 32px rgba(99,102,241,.35);}50%{box-shadow:0 18px 42px rgba(139,92,246,.55);}}
              @keyframes llmPulse{0%,100%{transform:scale(0.9);opacity:0.55;}50%{transform:scale(1.05);opacity:1;}}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_success():
        btn_placeholder.markdown(
            """
            <div style='width:100%; padding:16px 24px; border:none; border-radius:14px;
                background:linear-gradient(135deg,#10b981,#059669); color:#fff; font-weight:600; font-size:15px;
                box-shadow:0 8px 32px rgba(16,185,129,.45); position:relative; overflow:hidden;'
                id='success-tile'>
                <div style='display:flex; align-items:center; justify-content:center; gap:12px;'>
                    <span style='font-size:20px'>✨</span>
                    <span>SMS Generated Successfully!</span>
                    <span style='font-size:20px'>🎉</span>
                </div>
            </div>
            <style>
              #success-tile{animation:successBounce .6s cubic-bezier(.68,-.55,.265,1.55)}
              @keyframes successBounce{0%{transform:scale(.85);opacity:0}55%{transform:scale(1.08);opacity:1}100%{transform:scale(1);opacity:1}}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Initialise ui state
    if 'gen_ui_state' not in st.session_state:
        st.session_state['gen_ui_state'] = 'ready'

    # Render current state
    if st.session_state['gen_ui_state'] == 'ready':
        render_ready()
    elif st.session_state['gen_ui_state'] == 'fetching':
        render_fetching()
    elif st.session_state['gen_ui_state'] == 'llm_generating':
        render_llm_generating()
    elif st.session_state['gen_ui_state'] == 'success':
        render_success()

    # Lightweight message listener (uses streamlit's built-in rerun on state change simulation via empty updates)
    gen_event = st.session_state.get('gen_event_trigger')  # internal debug

    # Handle click event coming from JS (Streamlit cannot directly capture window.postMessage; we simulate via st.button fallback)
    # Provide fallback server-side button for accessibility (small & centered)
    fallback_col_spacer_left, fallback_col_center, fallback_col_spacer_right = st.columns([1,2,1])
    with fallback_col_center:
        fallback_clicked = st.button(
            "Click here to generate your custom SMS !",  # user requested label (kept exact spelling)
            use_container_width=True,
        )
        # Inject minimal CSS to shrink its visual footprint
        st.markdown(
            """
            <style>
            button[kind="secondary"]#generate_sms_btn_fallback, 
            div[data-testid="stButton"] button#generate_sms_btn_fallback {
                font-size: 12px !important;
                padding: 10px 20px !important;
                border-radius: 10px !important;
                background: linear-gradient(135deg,#444,#222) !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
            }
            button#generate_sms_btn_fallback:focus {outline:2px solid var(--primary,#ff2d55)}
            </style>
            """,
            unsafe_allow_html=True,
        )
    if fallback_clicked and st.session_state['gen_ui_state'] == 'ready':
        st.session_state['gen_ui_state'] = 'fetching'
        render_fetching()
        import time as _t
        try:
            if not selection:
                st.warning("⚠️ Select a usage type and persona before generating.")
                st.session_state['gen_ui_state'] = 'ready'
                render_ready()
            else:
                rag_start = _t.time()
                compose_response = call_compose(selection)
                rag_elapsed = _t.time() - rag_start
                if rag_elapsed < 1.0:
                    _t.sleep(1.0 - rag_elapsed)
                payload_raw = compose_response.get("llm_input_json")
                if not payload_raw:
                    raise ValueError("Compose payload missing llm_input_json")
                payload = json.loads(payload_raw)
                if hasattr(selection, 'is_equipment') and selection.is_equipment:
                    brand_value = selection.value
                    payload.setdefault("brand", brand_value)
                    payload.setdefault("hset_brand", brand_value)
                    offer_ctx = payload.setdefault("offer_context", {})
                    offer_ctx.setdefault("marque", brand_value)
                st.session_state['llm_input'] = payload
                live_url = (st.session_state.get("LIVE_LLM_URL") or LIVE_LLM_URL or "").strip()
                live_model_id = st.session_state.get("LIVE_LLM_MODEL_ID") or DEFAULT_LIVE_LLM_MODEL_ID
                if llm_mode == 'Live inference' and live_url:
                    st.session_state['gen_ui_state'] = 'llm_generating'
                    render_llm_generating()
                    _t.sleep(0.3)
                    sms_text = live_llm(
                        payload,
                        live_url,
                        os.getenv('LLM_API_KEY') or LIVE_LLM_API_KEY,
                        live_model_id,
                    )
                else:
                    sms_text = mock_llm(payload)
                st.session_state['sms_text'] = sms_text
                st.session_state['gen_ui_state'] = 'success'
                render_success()
                _t.sleep(1.0)
                st.session_state['gen_ui_state'] = 'ready'
                render_ready()
        except Exception as exc:
            st.error(f"SMS Generation failed: {exc}")
            st.session_state['gen_ui_state'] = 'ready'
            render_ready()

    st.markdown("</div>", unsafe_allow_html=True)  # Close premium-card
st.markdown("</div>", unsafe_allow_html=True)  # Close center section wrapper

# Bottom row
bottom_row = st.columns(2, gap="large")

# Personalized SMS
with bottom_row[0]:
    st.markdown("""
    <div class='premium-card'>
        <div class='card-header'>
            <div class='card-icon'>📱</div>
            <div>
                <h3 class='card-title'>Personalized SMS</h3>
                <p class='card-subtitle'>AI-powered SMS generation with mobile preview</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    preview_container = st.container()
    st.markdown("</div>", unsafe_allow_html=True)

# Telegram Delivery
with bottom_row[1]:
    st.markdown("""
    <div class='premium-card'>
        <div class='card-header'>
            <div class='card-icon'>📤</div>
            <div>
                <h3 class='card-title'>Telegram Delivery</h3>
                <p class='card-subtitle'>Instant delivery to @IAMistralbot</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    telegram_container = st.container()
    st.markdown("</div>", unsafe_allow_html=True)

# SMS generation is now handled inline within the button click

# Close enterprise grid
st.markdown("</div>", unsafe_allow_html=True)

sms_text = st.session_state.get("sms_text", "")
llm_input = st.session_state.get("llm_input")

with preview_container:
    # Show success notification when SMS is generated
    if st.session_state.get("show_success") and sms_text:
        st.markdown("""
        <div style='background:linear-gradient(135deg,var(--success),#0d9765); color:#fff; padding:12px 16px; 
                    border-radius:12px; margin-bottom:16px; text-align:center; 
                    animation:success .6s cubic-bezier(0.68,-0.55,0.265,1.55) both;
                    box-shadow:0 4px 16px rgba(16,185,129,.3)'>
            <span style='font-size:18px; margin-right:8px'>✨</span>
            <strong>SMS Generated Successfully!</strong>
            <span style='font-size:18px; margin-left:8px'>🎉</span>
        </div>
        """, unsafe_allow_html=True)
        # Clear success state after showing
        st.session_state["show_success"] = False
    
    if sms_text and llm_input:
        st.markdown(build_sms_preview_html(sms_text, llm_input), unsafe_allow_html=True)
        with st.expander("View JSON payload"):
            st.json(llm_input)
    else:
        st.markdown("""
            <div style='text-align:center; padding:32px 20px; background:rgba(255,255,255,0.02); 
                        border-radius:16px; border:1px dashed var(--border-subtle)'>
                <div style='font-size:48px; margin-bottom:16px; opacity:0.6'>📱</div>
                <h4 style='margin:0 0 12px; color:var(--text-soft); font-weight:600'>SMS Preview Ready</h4>
                <p class='card-caption' style='margin:0 0 20px'>
                    Generate a personalized SMS to preview the copy and product highlights in real time
                </p>
                <div style='display:flex; justify-content:center; gap:10px; font-size:22px; opacity:0.4'>
                    <span style='background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:12px'>🚀</span>
                    <span style='background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:12px'>💬</span>
                    <span style='background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:12px'>📊</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

with telegram_container:
    if sms_text and llm_input:
        st.markdown("<div class='card telegram-card'>", unsafe_allow_html=True)
        st.markdown(
            f"",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='card-caption'>Click 'Launch Bot' to start the bot , then click 'Send Now' to receive your personalized sms via Telegram</p>",
            unsafe_allow_html=True,
        )
        
        st.link_button(
            "🚀 Launch Bot",
            f"https://t.me/{TELEGRAM_BOT_USERNAME}?start=iam_campaign",
            type="primary",
            use_container_width=True
        )
        
        if st.button("📤 Send Now", type="secondary", use_container_width=True):
            ok, chat_id = fetch_latest_chat_id(TELEGRAM_BOT_TOKEN)
            if not ok:
                st.error(chat_id)
            else:
                sent, info = send_to_telegram(TELEGRAM_BOT_TOKEN, sms_text, chat_id)
                if sent:
                    st.success("✅ Message delivered")
                else:
                    st.error(info)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='text-align:center; padding:32px 20px; background:rgba(255,255,255,0.02); 
                        border-radius:16px; border:1px dashed var(--border-subtle)'>
                <div style='font-size:48px; margin-bottom:16px; opacity:0.6'>➤</div>
                <h4 style='margin:0 0 12px; color:var(--text-soft); font-weight:600'>Receive your SMS via Telegram</h4>
                <p class='card-caption' style='margin:0 0 20px'>
                    Maroc Telecom Telegram Bot sends you the custom marketing SMS instantly  
                </p>
                <div style='display:flex; justify-content:center; gap:10px; font-size:22px; opacity:0.4'>
                    <span style='background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:12px'>💬</span>
                    <span style='background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:12px'>➡️</span>
                    <span style='background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:12px'>📡</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


st.markdown(
    """
<footer>
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px;">
        <span>Maroc Telecom • Marketing SMS Campaign Platform </span>
        <span style="opacity: 0.6;">•</span>
        <span style="font-size: 0.8rem; opacity: 0.8;">Powered by customer segmentation, RAG, LLM SMS generation</span>
    </div>
</footer>
""",
    unsafe_allow_html=True,
)

# Close the layout containers
st.markdown("</div>", unsafe_allow_html=True)  # Close layout-stack
st.markdown("</div>", unsafe_allow_html=True)  # Close page-wrapper

