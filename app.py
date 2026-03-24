import os
import random

import gradio as gr
import requests as _requests_lib
from dotenv import load_dotenv

from src.agent.workflow import run_draft_to_ready
from src.schemas.models import DraftRequest
from src.llm.ollama_client import get_ollama_client
from src.llm.openrouter_client import get_openrouter_client
from src.llm.mock_client import MockLLMClient

load_dotenv()


# ---------------------------------------------------------------------------
# Design System CSS — Zinc/Slate dark theme with Indigo accent
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
html, :root { color-scheme: dark !important; }
@media (prefers-color-scheme: light) {
    body { background: var(--bg0) !important; color: var(--text) !important; }
    .gradio-container { background: transparent !important; color: var(--text) !important; }
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --bg0: #09090b;
  --bg1: #18181b;
  --bg2: #27272a;
  --card: rgba(255,255,255,0.035);
  --card-hover: rgba(255,255,255,0.055);
  --card-border: rgba(255,255,255,0.07);
  --card-border-hover: rgba(255,255,255,0.12);
  --text: #fafafa;
  --text-secondary: #d4d4d8;
  --muted: #a1a1aa;
  --subtle: #71717a;
  --accent: #6366f1;
  --accent-hover: #818cf8;
  --accent-glow: rgba(99,102,241,0.25);
  --accent2: #22c55e;
  --danger: #ef4444;
  --border: rgba(255,255,255,0.07);
  --radius-sm: 10px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 16px rgba(0,0,0,0.35);
  --shadow-lg: 0 8px 32px rgba(0,0,0,0.4);
  --transition: 0.2s cubic-bezier(.4,0,.2,1);
}

*{ box-sizing: border-box; }

body{
  background: var(--bg0);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ---- Gradio overrides ---- */
.gradio-container{
  background: transparent !important;
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
}
footer.svelte-1rjryqp { display: none !important; }
.wrap, .container{ background: transparent; }

/* ---- Typography ---- */
h1, h2, h3{ color: var(--text) !important; font-family: 'Inter', system-ui, sans-serif !important; }
h2{
  font-size: 13px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  color: var(--muted) !important;
  margin-top: 0 !important;
  margin-bottom: 16px !important;
}
label, .gr-label span{
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
  color: var(--muted) !important;
}
.gr-markdown, .markdown{ color: var(--text) !important; }

/* ---- Card sections ---- */
.card-section{
  background: var(--card) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 24px 28px 24px !important;
  margin-bottom: 16px !important;
  box-shadow: var(--shadow-sm);
  overflow: visible !important;
  transition: border-color var(--transition), box-shadow var(--transition);
}
.card-section:hover{
  border-color: var(--card-border-hover) !important;
  box-shadow: var(--shadow-md);
}

/* ---- Buttons ---- */
button, .gr-button{
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 20px !important;
  transition: all var(--transition) !important;
  cursor: pointer;
}
button:hover{
  transform: translateY(-1px);
}
button:active{
  transform: translateY(0);
}
button:focus-visible{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

/* Primary */
.gradio-container button.primary,
.gradio-container .primary button{
  background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  color: white !important;
  font-size: 14px !important;
  padding: 12px 24px !important;
  box-shadow: 0 0 20px var(--accent-glow);
}
.gradio-container button.primary:hover,
.gradio-container .primary button:hover{
  box-shadow: 0 0 0 3px var(--accent-glow), 0 8px 24px rgba(99,102,241,0.2);
  filter: brightness(1.05);
}

/* Secondary */
.gradio-container button.secondary,
.gradio-container .secondary button{
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent-hover) !important;
}
.gradio-container button.secondary:hover,
.gradio-container .secondary button:hover{
  background: rgba(99,102,241,0.08) !important;
  box-shadow: 0 0 0 3px var(--accent-glow);
}

/* Stop / tertiary */
.gradio-container button.stop,
.gradio-container .stop button{
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
}
.gradio-container button.stop:hover,
.gradio-container .stop button:hover{
  background: rgba(255,255,255,0.04) !important;
  border-color: var(--muted) !important;
}

/* ---- Inputs ---- */
textarea, input[type="text"], input[type="number"]{
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.025) !important;
  color: var(--text) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 14px !important;
  padding: 12px 16px !important;
  transition: border-color var(--transition), box-shadow var(--transition) !important;
}
textarea:focus, input:focus{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
  outline: none !important;
}
textarea::placeholder, input::placeholder{
  color: var(--subtle) !important;
  font-size: 13px !important;
}

/* ---- Dropdowns ---- */
.gr-dropdown, select{
  font-family: 'Inter', system-ui, sans-serif !important;
}

/* ---- Accordion ---- */
.gr-accordion, .accordion, details{
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  border-radius: var(--radius-md) !important;
  transition: border-color var(--transition);
}

/* ---- Output boxes ---- */
.output-box, .gr-textbox, .gr-json{
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.025) !important;
  border-radius: var(--radius-md) !important;
}

/* ---- Generation settings bar ---- */
.gen-settings-bar{
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  margin-bottom: 16px !important;
}
.gen-settings-bar summary,
.gen-settings-bar .label-wrap{
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  color: var(--muted) !important;
}

/* ---- Step indicator ---- */
.step-indicator{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin: 12px 0 20px;
}
.step-indicator .step{
  display: flex;
  align-items: center;
  gap: 8px;
}
.step-indicator .step-circle{
  width: 32px; height: 32px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700;
  transition: all 0.3s ease;
  flex-shrink: 0;
  font-family: 'Inter', sans-serif;
}
.step-indicator .step-circle.active{
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: #fff;
  box-shadow: 0 0 16px var(--accent-glow);
}
.step-indicator .step-circle.done{
  background: var(--accent2);
  color: #fff;
}
.step-indicator .step-circle.pending{
  background: var(--bg2);
  color: var(--subtle);
  border: 1px solid var(--border);
}
.step-indicator .step-label{
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-right: 6px;
  font-family: 'Inter', sans-serif;
}
.step-indicator .step-label.active{ color: var(--text); }
.step-indicator .step-label.done{ color: var(--accent2); }
.step-indicator .step-label.pending{ color: var(--subtle); }
.step-indicator .step-line{
  width: 40px; height: 2px;
  margin: 0 6px;
  border-radius: 1px;
}
.step-indicator .step-line.done{ background: var(--accent2); }
.step-indicator .step-line.pending{ background: var(--border); }

/* ---- Questions HTML ---- */
.questions-list{ list-style: none; padding: 0; margin: 0; }
.questions-list li{
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 16px;
  margin-bottom: 8px;
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 14px;
  line-height: 1.5;
  color: var(--text-secondary);
}
.questions-list .q-num{
  width: 26px; height: 26px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: #fff;
  font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  margin-top: 1px;
}

/* ---- Rubric HTML ---- */
.rubric-grid{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}
.rubric-item{
  padding: 12px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.025);
  display: flex;
  align-items: center;
  gap: 10px;
}
.rubric-badge{
  width: 28px; height: 28px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 700;
  flex-shrink: 0;
}
.rubric-badge.pass{
  background: rgba(34,197,94,0.15);
  color: var(--accent2);
}
.rubric-badge.fail{
  background: rgba(239,68,68,0.15);
  color: var(--danger);
}
.rubric-label{
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.4;
}

/* ---- Landing page CTA button ---- */
.get-started-btn{
  display: flex !important;
  justify-content: center !important;
  margin-top: -20px !important;
  margin-bottom: 80px !important;
}
.get-started-btn button{
  font-size: 16px !important;
  font-weight: 700 !important;
  padding: 14px 44px !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: white !important;
  box-shadow: 0 0 40px var(--accent-glow), var(--shadow-md) !important;
  letter-spacing: -0.01em !important;
}
.get-started-btn button:hover{
  box-shadow: 0 0 60px rgba(99,102,241,0.4), 0 8px 32px rgba(99,102,241,0.25) !important;
  transform: translateY(-2px) !important;
  filter: brightness(1.08) !important;
}

/* ---- Back button ---- */
.back-btn{
  max-width: fit-content !important;
}
.back-btn button{
  font-size: 13px !important;
  padding: 6px 14px !important;
  color: var(--subtle) !important;
  background: transparent !important;
  border: 1px solid var(--border) !important;
}
.back-btn button:hover{
  color: var(--text) !important;
  border-color: var(--card-border-hover) !important;
  background: rgba(255,255,255,0.03) !important;
}

/* ---- Kill Gradio's blue flash/blink ---- */
.progress-bar,
.generating,
.eta-bar,
.wrap.generating,
.translucent,
.progress-text{
  background: transparent !important;
  background-color: transparent !important;
}
/* Override Gradio 6 loading overlay that causes blue blink */
.wrap.default{
  background-color: transparent !important;
}
.pending{
  opacity: 1 !important;
  animation: none !important;
}
/* Gradio spinner: restyle to match our theme */
.loading{
  border-color: var(--border) !important;
  border-top-color: var(--accent) !important;
}
/* The small progress line at top of components */
.progress-bar > .progress{
  background: linear-gradient(90deg, var(--accent), var(--accent-hover)) !important;
}
/* Disable any built-in Gradio pulse/flash animations on containers */
.gradio-container *{
  --loader-color: var(--accent) !important;
}
.meta-text, .meta-text-center{
  color: var(--subtle) !important;
}

/* ---- Subtle ambient animation: slow-drifting gradient orb ---- */
@keyframes ambientDrift{
  0%  { transform: translate(0, 0) scale(1); opacity: 0.07; }
  25% { transform: translate(30px, -20px) scale(1.05); opacity: 0.09; }
  50% { transform: translate(-20px, 15px) scale(0.98); opacity: 0.06; }
  75% { transform: translate(15px, 25px) scale(1.03); opacity: 0.08; }
  100%{ transform: translate(0, 0) scale(1); opacity: 0.07; }
}
@keyframes ambientDrift2{
  0%  { transform: translate(0, 0) scale(1); opacity: 0.05; }
  33% { transform: translate(-25px, 20px) scale(1.06); opacity: 0.07; }
  66% { transform: translate(20px, -15px) scale(0.97); opacity: 0.04; }
  100%{ transform: translate(0, 0) scale(1); opacity: 0.05; }
}

/* App page ambient background glow */
#app-page{
  position: relative;
  overflow: hidden;
}
#app-page::before,
#app-page::after{
  content: '';
  position: fixed;
  border-radius: 50%;
  pointer-events: none;
  z-index: 0;
  filter: blur(80px);
}
#app-page::before{
  width: 500px;
  height: 500px;
  top: -100px;
  left: 5%;
  background: radial-gradient(circle, rgba(99,102,241,0.12), transparent 70%);
  animation: ambientDrift 20s ease-in-out infinite;
}
#app-page::after{
  width: 400px;
  height: 400px;
  bottom: 10%;
  right: 5%;
  background: radial-gradient(circle, rgba(129,140,248,0.08), transparent 70%);
  animation: ambientDrift2 25s ease-in-out infinite;
}
/* Ensure content sits above the ambient glow */
#app-page > *{
  position: relative;
  z-index: 1;
}

/* ---- Responsive ---- */
@media (max-width: 900px){
  .gradio-container{ padding: 8px !important; }
  .card-section{ padding: 18px 18px 14px !important; }
  .step-indicator .step-line{ width: 20px; }
}
@media (max-width: 480px){
  h2{ font-size: 12px !important; }
  button{ width: 100% !important; }
  .rubric-grid{ grid-template-columns: 1fr; }
}

/* ---- Animations ---- */
@keyframes fadeInUp{
  from{ opacity:0; transform:translateY(12px); }
  to{ opacity:1; transform:translateY(0); }
}
.fade-in{ animation: fadeInUp 0.4s ease forwards; }

/* Force landing page centering (Gradio 6 overrides) */
/* Force landing page centering (Gradio 6 overrides) */
.lp-hero, .lp-hero h1, .lp-hero p, .lp-hero-sub { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }
.lp-section-label, .lp-section-title { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }
.lp { width: 100% !important; max-width: 100% !important; }
#landing-page { max-width: 100% !important; padding: 0 !important; }
#landing-page > div { max-width: 100% !important; }
.lp-hero { display: flex !important; flex-direction: column !important; align-items: center !important; width: 100% !important; }
.lp-hero h1 { width: 100% !important; max-width: 780px !important; }
.lp-hero-sub { width: 100% !important; max-width: 520px !important; }
.lp-features-grid { justify-items: center; }
.lp-steps { justify-content: center !important; }
""".strip()


# ---------------------------------------------------------------------------
# Landing Page HTML — Premium SaaS marketing page
# ---------------------------------------------------------------------------
LANDING_PAGE_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
  .lp{ font-family:'Inter',system-ui,-apple-system,sans-serif; color:#fafafa; max-width:100%; overflow-x:hidden; }
  .lp *{ box-sizing:border-box; }

  /* ---- Hero ---- */
  .lp-hero{
    position: relative;
    text-align: center;
    padding: 140px 32px 80px;
    overflow: hidden;
  }
  @keyframes heroGlow{
    0%  { transform: translateX(-50%) scale(1); opacity: 0.8; }
    50% { transform: translateX(-48%) scale(1.08); opacity: 1; }
    100%{ transform: translateX(-50%) scale(1); opacity: 0.8; }
  }
  @keyframes heroGlow2{
    0%  { transform: scale(1) translate(0,0); opacity: 0.6; }
    50% { transform: scale(1.12) translate(-15px, 10px); opacity: 0.9; }
    100%{ transform: scale(1) translate(0,0); opacity: 0.6; }
  }
  .lp-hero-glow{
    position: absolute;
    top: -180px;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 600px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.18) 0%, rgba(99,102,241,0.05) 40%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: heroGlow 8s ease-in-out infinite;
  }
  .lp-hero-glow-2{
    position: absolute;
    top: 40px;
    right: 15%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(129,140,248,0.08), transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: heroGlow2 12s ease-in-out infinite;
  }
  .lp-hero h1{
    position: relative;
    z-index: 1;
    font-size: 60px;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.08;
    margin: 0 auto;
    max-width: 780px;
    color: #fafafa;
  }
  .lp-gradient-text{
    background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .lp-hero-sub{
    position: relative;
    z-index: 1;
    font-size: 18px;
    color: #a1a1aa;
    max-width: 520px;
    margin: 28px auto 0;
    line-height: 1.6;
    font-weight: 400;
  }

  /* ---- Section label ---- */
  .lp-section-label{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6366f1;
    text-align: center;
    margin-bottom: 16px;
  }
  .lp-section-title{
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    text-align: center;
    color: #fafafa;
    margin: 0 0 48px;
  }

  /* ---- Features ---- */
  .lp-features{
    padding: 80px 32px;
    max-width: 1060px;
    margin: 0 auto;
  }
  .lp-features-grid{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }
  .lp-feature-card{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 36px 28px 32px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .lp-feature-card:hover{
    border-color: rgba(255,255,255,0.12);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }
  .lp-feature-icon{
    width: 44px; height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(129,140,248,0.1));
    border: 1px solid rgba(99,102,241,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 20px;
  }
  .lp-feature-card h3{
    font-size: 17px;
    font-weight: 700;
    color: #fafafa;
    margin: 0 0 10px;
    letter-spacing: -0.01em;
  }
  .lp-feature-card p{
    font-size: 14px;
    color: #a1a1aa;
    line-height: 1.6;
    margin: 0;
  }

  /* ---- How it works ---- */
  .lp-how{
    padding: 80px 32px 100px;
    max-width: 900px;
    margin: 0 auto;
  }
  .lp-steps{
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0;
  }
  .lp-step{
    text-align: center;
    flex: 1;
    max-width: 240px;
  }
  .lp-step-num{
    width: 52px; height: 52px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: #fff;
    font-size: 22px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 18px;
    box-shadow: 0 0 24px rgba(99,102,241,0.25);
  }
  .lp-step h3{
    font-size: 16px;
    font-weight: 700;
    color: #fafafa;
    margin: 0 0 8px;
  }
  .lp-step p{
    font-size: 14px;
    color: #71717a;
    margin: 0;
    line-height: 1.5;
  }
  .lp-step-arrow{
    color: #3f3f46;
    font-size: 28px;
    margin-top: 12px;
    padding: 0 8px;
    flex-shrink: 0;
  }

  /* ---- Footer ---- */
  .lp-footer{
    padding: 40px 32px;
    text-align: center;
    color: #3f3f46;
    font-size: 13px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-weight: 500;
    letter-spacing: 0.01em;
  }

  /* ---- Deep dive ---- */
  .lp-deep{ padding: 80px 32px; max-width: 900px; margin: 0 auto; }
  .lp-deep h3{ font-size: 18px; font-weight: 700; color: #fafafa; margin: 32px 0 10px; }
  .lp-deep h3:first-child{ margin-top: 0; }
  .lp-deep p, .lp-deep li{ font-size: 14px; color: #a1a1aa; line-height: 1.7; margin: 0 0 8px; }
  .lp-deep ul{ padding-left: 20px; margin: 0 0 16px; }
  .lp-deep code{
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 5px; padding: 1px 7px; font-size: 13px; color: #818cf8;
    font-family: 'SF Mono', Consolas, monospace;
  }
  .lp-deep .formula{
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 16px 20px; margin: 12px 0 16px;
    font-family: 'SF Mono', Consolas, monospace; font-size: 13px; color: #d4d4d8;
    line-height: 1.8; overflow-x: auto;
  }
  .lp-param-grid{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 14px; margin: 16px 0;
  }
  .lp-param-card{
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 20px;
  }
  .lp-param-card h4{ font-size: 14px; font-weight: 700; color: #fafafa; margin: 0 0 6px; }
  .lp-param-card p{ font-size: 13px; margin: 0; }
  .lp-param-card .range{ font-size: 12px; color: #6366f1; margin-top: 4px; }

  /* ---- Responsive ---- */
  @media (max-width: 768px){
    .lp-hero h1{ font-size: 36px; }
    .lp-hero-sub{ font-size: 16px; }
    .lp-hero{ padding: 100px 20px 60px; }
    .lp-features-grid{ grid-template-columns: 1fr; gap: 14px; }
    .lp-steps{ flex-direction: column; align-items: center; gap: 24px; }
    .lp-step-arrow{ transform: rotate(90deg); font-size: 22px; padding: 0; }
    .lp-section-title{ font-size: 24px; }
    .lp-param-grid{ grid-template-columns: 1fr; }
  }
</style>

<div class="lp">
  <!-- HERO -->
  <section class="lp-hero">
    <div class="lp-hero-glow"></div>
    <div class="lp-hero-glow-2"></div>
    <h1>
      Turn messy notes into<br>
      <span class="lp-gradient-text">polished, ready-to-send</span><br>
      messages
    </h1>
    <p class="lp-hero-sub">
      AI-powered writing agent that drafts, self-checks, and finalizes
      your communications across any channel &mdash; in seconds.
    </p>
  </section>

  <!-- FEATURES -->
  <section class="lp-features">
    <div class="lp-section-label">Capabilities</div>
    <div class="lp-section-title">Everything you need to write better</div>
    <div class="lp-features-grid">
      <div class="lp-feature-card">
        <div class="lp-feature-icon">&#9993;</div>
        <h3>Multi-Channel Drafts</h3>
        <p>Email, WhatsApp, or Teams &mdash; each draft is formatted for the platform you need.</p>
      </div>
      <div class="lp-feature-card">
        <div class="lp-feature-icon">&#10004;</div>
        <h3>Self-Check Rubric</h3>
        <p>13 quality checks &mdash; greeting, closing, tone, word count, channel rules, faithfulness, and hallucination detection.</p>
      </div>
      <div class="lp-feature-card">
        <div class="lp-feature-icon">&#9881;</div>
        <h3>Style Presets</h3>
        <p>Professional, Friendly, Persuasive, or Creative &mdash; each preset tunes temperature, penalties, and phrase matching.</p>
      </div>
    </div>
  </section>

  <!-- HOW IT WORKS -->
  <section class="lp-how">
    <div class="lp-section-label">How It Works</div>
    <div class="lp-section-title">Five-stage agent pipeline</div>
    <div class="lp-steps">
      <div class="lp-step">
        <div class="lp-step-num">1</div>
        <h3>Clarify</h3>
        <p>Heuristic + LLM ensemble detects missing dates, amounts, or names and asks up to 6 targeted questions.</p>
      </div>
      <div class="lp-step-arrow">&#8594;</div>
      <div class="lp-step">
        <div class="lp-step-num">2</div>
        <h3>Draft &amp; Score</h3>
        <p>Generates up to 5 variants, scores each on 9 weighted components, and picks the best one.</p>
      </div>
      <div class="lp-step-arrow">&#8594;</div>
      <div class="lp-step">
        <div class="lp-step-num">3</div>
        <h3>Check &amp; Finalize</h3>
        <p>Runs 13 quality checks, detects hallucinations, then optionally rewrites to fix rubric issues.</p>
      </div>
    </div>
  </section>

  <!-- DEEP DIVE: HOW SCORING WORKS -->
  <section class="lp-deep">
    <div class="lp-section-label">Under the Hood</div>
    <div class="lp-section-title">How the scoring engine works</div>

    <h3>The Scoring Formula</h3>
    <p>Every draft variant is scored by a weighted sum of 9 components. The variant with the highest total score is selected.</p>
    <div class="formula">
score = w_closing &times; closing<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_next_step &times; next_step<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_short &times; min_length<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_subject &times; subject_line<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_word_size &times; word_count_fit<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_intent &times; intent_coverage<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_hallucination &times; halluc_penalty<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_tone &times; tone_match<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; + w_faithfulness &times; faithfulness
    </div>

    <h3>What Each Score Means</h3>
    <ul>
      <li><strong>Closing</strong> (+1.0 / -0.5) &mdash; Does the draft end with a sign-off? Looks for: <code>sincerely</code>, <code>regards</code>, <code>thanks</code>, <code>best</code>, <code>cheers</code>, etc.</li>
      <li><strong>Next Step</strong> (-0.5 to +1.5) &mdash; Does the draft include actionable language matching your style preset? Each preset has a database of weighted phrases (e.g., Professional: "please let me know" = 1.0, "kindly" = 0.6). The matched weight is converted to a score: <code>delta = -0.5 + 2.0 &times; (matched_weight / total_weight)</code>.</li>
      <li><strong>Min Length</strong> (+0.8 / -1.0) &mdash; Is the draft at least 40 words? Short drafts are heavily penalized.</li>
      <li><strong>Subject Line</strong> (+0.7 / -0.8 / -1.0) &mdash; Email channel rewards a "Subject:" line when requested (+0.7), penalizes its absence (-0.8). Non-email channels penalize a subject line (-1.0).</li>
      <li><strong>Word Count Fit</strong> (+1.0 or penalty) &mdash; Is the word count in the target range? Small: 60-90, Medium: 110-160, Large: 180-260. Penalty is proportional to distance: <code>-min(1.5, distance / target)</code>.</li>
      <li><strong>Intent Coverage</strong> (0.0 to 1.0) &mdash; If the user provided specific dates, amounts, or university names, does the draft include them? <code>score = details_found / details_required</code>.</li>
      <li><strong>Hallucination Penalty</strong> (0 or -1.0 each) &mdash; If the user did NOT provide a date but the draft invents one, that is -1.0. Same for fabricated amounts.</li>
      <li><strong>Tone Match</strong> (0.0 to 1.0) &mdash; Fraction of style-preset markers found. Professional looks for: "kindly", "at your earliest convenience", "i look forward", etc. Friendly looks for: "when you get a chance", "thanks so much", etc.</li>
      <li><strong>Faithfulness</strong> (0.0 to 1.0) &mdash; How closely does the draft reflect your original notes? Uses semantic similarity (sentence-transformers) when available, or word-overlap as fallback. Calculated as: for each source sentence, find the best-matching draft sentence by cosine similarity, then average all matches.</li>
    </ul>

    <h3>Faithfulness Scoring</h3>
    <p>When <code>sentence-transformers</code> is installed, faithfulness is measured using the <strong>all-MiniLM-L6-v2</strong> model:</p>
    <ul>
      <li>Both your notes and the draft are split into sentences.</li>
      <li>Each sentence is converted to a 384-dimensional embedding vector.</li>
      <li>For each source sentence, the system finds the draft sentence with the highest cosine similarity.</li>
      <li>The faithfulness score is the average of these best-match similarities (0% = no overlap, 100% = perfect match).</li>
    </ul>
    <p>Without sentence-transformers, a <strong>word-overlap fallback</strong> is used: it extracts meaningful words (3+ characters, excluding 56 common stop words) from both texts and computes <code>overlap / source_words</code>.</p>

    <h3>Hallucination Detection</h3>
    <p>When <code>sentence-transformers</code> is installed, an NLI (Natural Language Inference) model (<strong>cross-encoder/nli-deberta-v3-xsmall</strong>) classifies each draft sentence against your original notes:</p>
    <ul>
      <li><strong>Entailed</strong> &mdash; the sentence is supported by your notes (good).</li>
      <li><strong>Neutral</strong> &mdash; the sentence is neither supported nor contradicted. Flagged if confidence &ge; 70%.</li>
      <li><strong>Contradiction</strong> &mdash; the sentence contradicts your notes (always flagged).</li>
    </ul>
    <p>The hallucination score = fraction of flagged sentences. If it exceeds 30% (configurable), a warning is shown.</p>

    <h3>Quality Checks (13 total)</h3>
    <ul>
      <li><strong>Has Greeting</strong> &mdash; starts with Hi, Hello, Dear, etc.</li>
      <li><strong>Has Closing</strong> &mdash; ends with Sincerely, Regards, Thanks, Best, etc.</li>
      <li><strong>Next Step Language</strong> &mdash; includes phrases like "please", "let me know", "feel free".</li>
      <li><strong>Minimum Length</strong> &mdash; at least 40 words.</li>
      <li><strong>Has Paragraphs</strong> &mdash; 3+ lines (not a wall of text).</li>
      <li><strong>Tone Match</strong> &mdash; at least 1 style-preset marker found.</li>
      <li><strong>Word Count Target</strong> &mdash; word count within the Small/Medium/Large range.</li>
      <li><strong>Email: Subject Line</strong> &mdash; present when requested.</li>
      <li><strong>Email: Greeting</strong> &mdash; greeting present for email channel.</li>
      <li><strong>WhatsApp: No Subject</strong> &mdash; no "Subject:" line in WhatsApp messages.</li>
      <li><strong>WhatsApp: Concise</strong> &mdash; under 120 words for WhatsApp.</li>
      <li><strong>Teams: No Subject</strong> &mdash; no "Subject:" line for Teams.</li>
      <li><strong>Teams: Professional</strong> &mdash; has both greeting and closing.</li>
    </ul>

    <h3>Clarification Engine</h3>
    <p>Before drafting, the agent checks if critical information is missing. It uses an <strong>ensemble of heuristics + LLM</strong>:</p>
    <ul>
      <li><strong>Heuristic questions</strong> &mdash; pattern-matches for missing dates (regex: MM/DD, "March 12", weekday names, "5pm"), missing amounts ($, EUR, "deposit"), or unspecified university names.</li>
      <li><strong>LLM questions</strong> &mdash; the LLM analyzes your notes at low temperature (0.3) and suggests additional questions.</li>
      <li><strong>Deduplication</strong> &mdash; questions from both sources are merged, near-duplicates removed (cosine similarity &gt; 0.85, or substring match), and capped at 6.</li>
      <li><strong>Conservative proceed</strong> &mdash; drafting begins only when BOTH heuristic and LLM agree all critical info is present.</li>
    </ul>
  </section>

  <!-- GENERATION PARAMETERS -->
  <section class="lp-deep">
    <div class="lp-section-label">Generation Controls</div>
    <div class="lp-section-title">What the parameters do</div>

    <div class="lp-param-grid">
      <div class="lp-param-card">
        <h4>Temperature</h4>
        <p>Controls randomness. Lower = more predictable and focused. Higher = more creative and varied.</p>
        <div class="range">Range: 0.0 &ndash; 2.0 &bull; Professional default: 0.4 &ndash; 0.9</div>
      </div>
      <div class="lp-param-card">
        <h4>Top-P (Nucleus Sampling)</h4>
        <p>Only considers tokens whose cumulative probability reaches this threshold. Lower = fewer choices = more focused output.</p>
        <div class="range">Range: 0.0 &ndash; 1.0 &bull; Default: 0.9</div>
      </div>
      <div class="lp-param-card">
        <h4>Top-K</h4>
        <p>Limits the model to the K most likely next tokens at each step. Lower = more conservative.</p>
        <div class="range">Range: 0 &ndash; 200 &bull; Professional default: 20 &ndash; 45</div>
      </div>
      <div class="lp-param-card">
        <h4>Repeat Penalty</h4>
        <p>Penalizes tokens that already appeared in the output. Higher = less repetition.</p>
        <div class="range">Range: 0.0 &ndash; 2.5 &bull; Default: 1.1</div>
      </div>
      <div class="lp-param-card">
        <h4>Presence Penalty</h4>
        <p>Encourages the model to talk about new topics. Positive values push for variety; negative values allow repetition.</p>
        <div class="range">Range: -2.0 &ndash; 2.5 &bull; Default: 0.0</div>
      </div>
      <div class="lp-param-card">
        <h4>Frequency Penalty</h4>
        <p>Reduces the chance of repeating the same word proportional to how often it already appeared.</p>
        <div class="range">Range: -2.0 &ndash; 2.5 &bull; Default: 0.0</div>
      </div>
    </div>

    <h3>Style Presets</h3>
    <p>Each preset auto-tunes the parameters above based on your creativity intensity slider (0&ndash;100):</p>
    <ul>
      <li><strong>Professional</strong> &mdash; Low temperature (0.4&ndash;0.9), tight top-k (20&ndash;45). Produces formal, predictable output.</li>
      <li><strong>Friendly</strong> &mdash; Medium temperature (0.6&ndash;1.3), wider top-k (30&ndash;75). Warmer, more natural tone.</li>
      <li><strong>Persuasive</strong> &mdash; Higher temperature (0.75&ndash;1.5), wide top-k (40&ndash;110). Bolder word choices, stronger arguments.</li>
      <li><strong>Creative</strong> &mdash; Highest temperature (1.05&ndash;2.0), widest top-k (80&ndash;160). Most varied and inventive output.</li>
    </ul>

    <h3>Scoring Weights</h3>
    <p>The default weights used for the scoring formula (overridable via <code>evals/scoring_weights.json</code>):</p>
    <div class="formula">
w_closing: 1.0 &nbsp;&bull;&nbsp; w_next_step: 1.0 &nbsp;&bull;&nbsp; w_short: 1.0<br>
w_subject: 1.0 &nbsp;&bull;&nbsp; w_word_size: 1.0 &nbsp;&bull;&nbsp; w_intent: 1.0<br>
w_hallucination: 1.0 &nbsp;&bull;&nbsp; w_tone: 0.8 &nbsp;&bull;&nbsp; w_faithfulness: 0.8
    </div>
  </section>

  <!-- FOOTER -->
  <footer class="lp-footer">
    Draft-to-Ready Writing Agent &middot; Powered by Gradio + Ollama
  </footer>
</div>
"""


# ---------------------------------------------------------------------------
# App page header HTML — slim, compact
# ---------------------------------------------------------------------------
APP_HEADER_HTML = """
<div style="display:flex;align-items:center;gap:14px;padding:14px 24px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#6366f1,#818cf8);display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">
    &#9997;&#65039;
  </div>
  <div>
    <div style="font-size:18px;font-weight:700;letter-spacing:-0.02em;color:#fafafa;line-height:1.2;">
      Draft-to-Ready
    </div>
    <div style="font-size:12px;color:#71717a;font-weight:500;margin-top:1px;">
      Writing Agent
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Helper: Step indicator HTML
# ---------------------------------------------------------------------------
def _build_step_indicator(step: int = 0) -> str:
    """Return HTML for a 3-step progress bar. step: 0=start, 1=clarify, 2=draft, 3=final."""
    labels = ["Clarify", "Draft", "Final"]
    parts = []
    for i, label in enumerate(labels):
        idx = i + 1
        if idx < step:
            cls_circle = "done"
            cls_label = "done"
            symbol = "&#10003;"
        elif idx == step:
            cls_circle = "active"
            cls_label = "active"
            symbol = str(idx)
        else:
            cls_circle = "pending"
            cls_label = "pending"
            symbol = str(idx)
        parts.append(
            f'<div class="step">'
            f'<span class="step-circle {cls_circle}">{symbol}</span>'
            f'<span class="step-label {cls_label}">{label}</span>'
            f'</div>'
        )
        if i < len(labels) - 1:
            line_cls = "done" if idx < step else "pending"
            parts.append(f'<div class="step-line {line_cls}"></div>')

    return f'<div class="step-indicator">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Helper: Render questions as styled HTML
# ---------------------------------------------------------------------------
def _render_questions_html(questions: list[str]) -> str:
    if not questions:
        return ""
    items = "".join(
        f'<li><span class="q-num">{i + 1}</span><span>{q}</span></li>'
        for i, q in enumerate(questions)
    )
    return f'<ol class="questions-list">{items}</ol>'


# ---------------------------------------------------------------------------
# Helper: Render rubric as styled HTML
# ---------------------------------------------------------------------------
def _section_label(text: str, color: str = "var(--muted)") -> str:
    return (
        f'<div style="margin-top:16px;margin-bottom:6px;font-size:11px;font-weight:700;'
        f'color:{color};text-transform:uppercase;letter-spacing:0.06em;">{text}</div>'
    )


def _badge(value: str, cls: str = "pass", size: str = "14px") -> str:
    return f'<span class="rubric-badge {cls}" style="font-size:{size};">{value}</span>'


def _item(badge_html: str, label: str, extra_style: str = "") -> str:
    return (
        f'<div class="rubric-item" style="{extra_style}">'
        f'{badge_html}<span class="rubric-label">{label}</span>'
        f'</div>'
    )


def _pct_badge(score, good_threshold: float = 0.4, invert: bool = False) -> str:
    """Render a percentage badge. invert=True means higher is worse (hallucination)."""
    if score is None:
        return _badge("-", "pass", "11px")
    pct = int(round(float(score) * 100))
    if invert:
        cls = "fail" if pct > int(good_threshold * 100) else "pass"
    else:
        cls = "pass" if pct >= int(good_threshold * 100) else "fail"
    return _badge(f"{pct}%", cls, "12px")


def _render_rubric_html(rubric: dict) -> str:
    if not rubric:
        return ""

    P = []  # html parts

    # ================================================================
    # SECTION 1: Key Metrics (score badges row)
    # ================================================================
    badges = []
    faith = rubric.get("faithfulness_score")
    if faith is not None:
        badges.append(_item(_pct_badge(faith, 0.4), "Faithfulness"))
    halluc = rubric.get("hallucination_score")
    if halluc is not None:
        badges.append(_item(_pct_badge(halluc, 0.3, invert=True), "Hallucination"))
    sel_score = rubric.get("selected_variant_score")
    if sel_score is not None:
        badges.append(_item(_badge(str(sel_score), "pass", "11px"), "Overall Score"))
    variants_req = rubric.get("draft_variants_requested")
    sel_idx = rubric.get("selected_variant_index")
    if variants_req is not None and sel_idx is not None:
        badges.append(_item(_badge(f"#{sel_idx + 1}", "pass", "11px"), f"Best of {variants_req}"))
    finalized = rubric.get("finalized_with")
    if finalized:
        badges.append(_item(_badge("&#9998;", "pass", "10px"), finalized))
    if badges:
        P.append(f'<div class="rubric-grid">{"".join(badges)}</div>')

    # ================================================================
    # SECTION 2: Quality Checks (all boolean rubric items)
    # ================================================================
    check_defs = [
        ("has_closing", "Has Closing", "Draft ends with a sign-off (Sincerely, Regards, Thanks, etc.)"),
        ("has_greeting", "Has Greeting", "Draft starts with a greeting (Hi, Hello, Dear, etc.)"),
        ("mentions_next_step", "Next Step Language", "Draft includes actionable phrases (please, let me know, etc.)"),
        ("not_too_short", "Minimum Length", "Draft has at least 40 words"),
        ("has_paragraphs", "Has Paragraphs", "Draft has 3+ lines (not a wall of text)"),
        ("tone_match", "Tone Match", "At least one tone marker for the selected style preset was found"),
        ("word_count_in_range", "Word Count Target", None),
        ("email_has_subject", "Email: Subject Line", "Email drafts should include a Subject: line when requested"),
        ("email_has_greeting", "Email: Greeting", "Email drafts should include a greeting"),
        ("whatsapp_no_subject", "WhatsApp: No Subject", "WhatsApp messages should not have a Subject: line"),
        ("whatsapp_concise", "WhatsApp: Concise", "WhatsApp messages should be under 120 words"),
        ("teams_no_subject", "Teams: No Subject", "Teams messages should not have a Subject: line"),
        ("teams_professional", "Teams: Professional", "Teams messages should have both greeting and closing"),
    ]

    checks = []
    for key, display, tooltip in check_defs:
        val = rubric.get(key)
        if val is None:
            continue
        is_pass = bool(val)
        symbol = "&#10003;" if is_pass else "&#10007;"
        cls = "pass" if is_pass else "fail"
        label = display
        # Enrich word count target with actual numbers
        if key == "word_count_in_range":
            wc = rubric.get("word_count", "?")
            target = rubric.get("word_count_target", "?")
            label = f"Word Count: {wc} (target {target})"
        if tooltip and not is_pass:
            label += f' <span style="color:var(--subtle);font-size:11px;"> — {tooltip}</span>'
        checks.append(_item(_badge(symbol, cls), label))

    if rubric.get("hallucination_warning"):
        checks.append(_item(
            _badge("!", "fail"),
            '<span style="color:var(--danger);">Hallucination Warning — draft may contain unsupported claims</span>',
            "border-color:rgba(239,68,68,0.3);",
        ))

    # Tone markers detail
    tone_hits = rubric.get("tone_markers_found")
    tone_total = rubric.get("tone_markers_total")
    if tone_hits is not None and tone_total is not None:
        cls = "pass" if tone_hits >= 1 else "fail"
        checks.append(_item(_badge(f"{tone_hits}/{tone_total}", cls, "10px"), "Tone Markers Found"))

    if checks:
        P.append(_section_label("Quality Checks"))
        P.append(f'<div class="rubric-grid">{"".join(checks)}</div>')

    # ================================================================
    # SECTION 3: Flagged Sentences
    # ================================================================
    flagged = rubric.get("flagged_sentences", [])
    if flagged:
        items = []
        for sent in flagged[:5]:
            trunc = sent[:100] + ("..." if len(sent) > 100 else "")
            items.append(_item(_badge("&#10007;", "fail"), f'<span style="font-size:12px;">Flagged: {trunc}</span>', "border-color:rgba(239,68,68,0.2);"))
        P.append(_section_label("Flagged Sentences", "var(--danger)"))
        P.append(f'<div class="rubric-grid">{"".join(items)}</div>')

    # ================================================================
    # SECTION 4: Advanced Scoring Parameters (collapsible)
    # ================================================================
    scored = rubric.get("draft_variants_scored", [])
    selected = rubric.get("selected_variant_index")
    sel_variant = None
    for v in scored:
        if v.get("index") == selected:
            sel_variant = v
            break

    adv_parts = []
    sv = sel_variant or {}

    _adv_hdr = (
        '<div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;'
        'letter-spacing:0.06em;margin:18px 0 8px;">{}</div>'
    )
    _adv_pill_pass = (
        '<span style="display:inline-block;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);'
        'border-radius:6px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px;color:#22c55e;">{}</span>'
    )
    _adv_pill_fail = (
        '<span style="display:inline-block;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);'
        'border-radius:6px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px;color:#ef4444;">{}</span>'
    )
    _adv_tip = (
        '<div style="margin:4px 0 8px;padding:8px 12px;background:rgba(99,102,241,0.06);'
        'border:1px solid rgba(99,102,241,0.12);border-radius:8px;font-size:12px;'
        'color:var(--text-secondary,#d4d4d8);line-height:1.5;">'
        '<strong style="color:var(--accent-hover,#818cf8);">Tip:</strong> {}</div>'
    )

    def _adv_row(label, val, weight, rule, reason, tip):
        if val is None:
            return ""
        vf = float(val)
        wv = vf * weight
        c = "var(--accent2)" if vf > 0 else "var(--danger)" if vf < 0 else "var(--muted)"
        s = "+" if vf > 0 else ""
        icon = "&#10003;" if vf > 0 else "&#10007;" if vf < 0 else "&#9679;"
        tip_html = _adv_tip.format(tip) if vf <= 0 and tip else ""
        return (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<td style="padding:6px 10px;font-weight:600;color:var(--text);">{label}</td>'
            f'<td style="padding:6px 10px;text-align:center;"><span style="color:{c};font-weight:600;">{s}{vf:.3f}</span></td>'
            f'<td style="padding:6px 10px;text-align:center;color:var(--subtle);font-size:12px;">&times;{weight:.1f}</td>'
            f'<td style="padding:6px 10px;text-align:center;"><span style="color:{c};font-weight:600;">{s}{wv:.3f}</span></td>'
            f'<td style="padding:6px 10px;font-size:11px;color:var(--subtle);">{rule}</td></tr>'
            f'<tr><td colspan="5" style="padding:0 10px 2px;">'
            f'<div style="font-size:11px;color:{c};margin-bottom:1px;">{icon} {reason}</div>'
            f'{tip_html}</td></tr>'
        )

    comp = sv.get("component", {})
    if comp:
        # Gather context for explanations
        faith_exp = sv.get("faithfulness_explanation", {})
        fw = faith_exp.get("found_in_draft", [])
        mw = faith_exp.get("missing_from_draft", [])
        tf = sv.get("tone_markers_found", [])
        tm = sv.get("tone_markers_missing", [])
        preset = sv.get("style_preset", "Professional")
        np_list = sv.get("next_step_matched_phrases", [])
        hn = sv.get("halluc_notes", [])
        wc_a = sv.get("word_count", "?")
        wc_t = rubric.get("word_count_target", "?")

        fv = float(comp.get("faithfulness_contrib", 0))
        f_reason = (f"Good coverage. Found: {', '.join(fw[:5])}" if fv > 0.3
                    else f"Partial. Found: {', '.join(fw[:3])}. Missing: {', '.join(mw[:3])}" if fv > 0
                    else f"Very low overlap. Missing: {', '.join(mw[:5])}" if mw else "No overlap.")
        f_tip = f"Include these words: {', '.join(mw[:5])}" if mw else ""

        hv = float(comp.get("hallucination_contrib", 0))
        h_reason = f"Fabricated: {', '.join(hn)}" if hv < 0 else "No fabricated details."
        h_tip = "Remove invented dates/amounts not in your notes." if hv < 0 else ""

        iv = float(comp.get("intent_contrib", 0))
        i_reason = "All required details present." if iv >= 0.8 else ("Partial details." if iv > 0 else "No specific details required.")
        i_tip = "Add specific dates, amounts, names to notes." if iv < 0.5 else ""

        tv = float(comp.get("tone_contrib", 0))
        t_reason = (f"Strong {preset}. Found: {', '.join(tf[:3])}" if tv >= 0.5
                    else f"Partial {preset}. Found: {', '.join(tf[:2])}. Missing: {', '.join(tm[:2])}" if tv > 0
                    else f"No {preset} markers. Expected: {', '.join(tm[:3])}" if tm else "No markers defined.")
        t_tip = f"Use phrases like: {', '.join(tm[:3])}" if tm else ""

        nv = float(comp.get("next_step_contrib", 0))
        pn = [p.get("phrase", p) if isinstance(p, dict) else p for p in np_list[:3]]
        n_reason = f"Found: {', '.join(pn)}" if nv > 0 else "No actionable phrases detected."
        n_tip = "Add 'please let me know' or similar." if nv <= 0 else ""

        sbv = float(comp.get("subject_contrib", 0))
        s_reason = "Subject line correct." if sbv >= 0 else "Subject line missing or wrongly present."
        s_tip = "Add 'Subject:' for email, remove for WhatsApp/Teams." if sbv < 0 else ""

        wv = float(comp.get("word_size_contrib", 0))
        w_reason = f"Word count ({wc_a}) in target ({wc_t})." if wv >= 0.8 else f"Word count ({wc_a}) outside target ({wc_t})."
        w_tip = f"Aim for {wc_t} words." if wv < 0.5 else ""

        cv = float(comp.get("closing_contrib", 0))
        c_reason = "Sign-off found." if cv > 0 else "No sign-off detected."
        c_tip = "Add Sincerely, Best regards, or Thanks." if cv <= 0 else ""

        mv = float(comp.get("short_contrib", 0))
        m_reason = f"Meets minimum ({wc_a} words)." if mv > 0 else f"Too short ({wc_a} words, need 40+)."
        m_tip = "Add more context to reach 40+ words." if mv <= 0 else ""

        rows_data = [
            ("Faithfulness", comp.get("faithfulness_contrib"), 1.5, "Overlap with your notes", f_reason, f_tip),
            ("Hallucination", comp.get("hallucination_contrib"), 1.4, "-1.0 per fabrication", h_reason, h_tip),
            ("Intent", comp.get("intent_contrib"), 1.2, "Required details present", i_reason, i_tip),
            ("Tone", comp.get("tone_contrib"), 1.1, f"{preset} markers", t_reason, t_tip),
            ("Next Step", comp.get("next_step_contrib"), 0.9, "Actionable phrases", n_reason, n_tip),
            ("Subject", comp.get("subject_contrib"), 0.8, "Channel compliance", s_reason, s_tip),
            ("Word Count", comp.get("word_size_contrib"), 0.6, f"Target: {wc_t}", w_reason, w_tip),
            ("Closing", comp.get("closing_contrib"), 0.5, "Sign-off presence", c_reason, c_tip),
            ("Min Length", comp.get("short_contrib"), 0.4, "40+ words", m_reason, m_tip),
        ]
        srows = [_adv_row(*r) for r in rows_data if r[1] is not None]
        if srows:
            adv_parts.append(
                f'{_adv_hdr.format("Score Breakdown (sorted by importance)")}'
                f'<table style="width:100%;border-collapse:collapse;font-size:13px;color:var(--text-secondary,#d4d4d8);">'
                f'<thead><tr style="border-bottom:1px solid var(--border);color:var(--subtle);">'
                f'<th style="padding:5px 10px;text-align:left;font-size:10px;text-transform:uppercase;">Component</th>'
                f'<th style="padding:5px 10px;text-align:center;font-size:10px;text-transform:uppercase;">Raw</th>'
                f'<th style="padding:5px 10px;text-align:center;font-size:10px;text-transform:uppercase;">Weight</th>'
                f'<th style="padding:5px 10px;text-align:center;font-size:10px;text-transform:uppercase;">Weighted</th>'
                f'<th style="padding:5px 10px;text-align:left;font-size:10px;text-transform:uppercase;">Rule</th>'
                f'</tr></thead><tbody>{"".join(srows)}</tbody></table>'
            )

        # Faithfulness deep dive
        if fw or mw:
            method = faith_exp.get("method", "word_overlap")
            ml = "Sentence Embeddings (all-MiniLM-L6-v2)" if method == "sentence_embeddings" else "Word Overlap (fallback)"
            adv_parts.append(
                f'{_adv_hdr.format("Faithfulness Detail")}'
                f'<div style="font-size:12px;color:var(--subtle);margin-bottom:8px;">Method: {ml}</div>'
                f'<div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px;">Found in draft:</div>'
                f'<div style="margin-bottom:8px;">{"".join(_adv_pill_pass.format(w) for w in fw[:12]) or "<em style=color:var(--subtle)>none</em>"}</div>'
                f'<div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px;">Missing from draft:</div>'
                f'<div style="margin-bottom:8px;">{"".join(_adv_pill_fail.format(w) for w in mw[:12]) or "<em style=color:var(--subtle)>none</em>"}</div>'
            )

        # Tone deep dive
        if tf or tm:
            adv_parts.append(
                f'{_adv_hdr.format(f"Tone Detail ({preset} preset)")}'
                f'<div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px;">Found:</div>'
                f'<div style="margin-bottom:8px;">{"".join(_adv_pill_pass.format(m) for m in tf) or "<em style=color:var(--subtle)>none</em>"}</div>'
                f'<div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px;">Try using these:</div>'
                f'<div style="margin-bottom:8px;">{"".join(_adv_pill_fail.format(m) for m in tm) or "<em style=color:var(--subtle)>all found!</em>"}</div>'
            )

        # Hallucination notes
        if hn:
            adv_parts.append(
                f'{_adv_hdr.format("Hallucination Notes")}'
                + "".join(f'<div style="padding:4px 0;font-size:12px;color:var(--danger);">&#9888; {n}</div>' for n in hn)
            )

    # Per-variant comparison table
    if scored:
        rows = []
        for v in scored:
            idx = v.get("index", 0)
            sc = v.get("score", "?")
            wc = v.get("word_count", "?")
            f_score = v.get("faithfulness_score")
            f_str = f"{int(round(float(f_score)*100))}%" if f_score is not None else "-"
            tone = v.get("tone_score")
            tone_str = f"{int(round(float(tone)*100))}%" if tone is not None else "-"
            subj = "Yes" if v.get("has_subject") else "No"
            sel = " &#9733;" if idx == selected else ""
            bg = "background:rgba(99,102,241,0.06);" if idx == selected else ""
            rows.append(
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);{bg}">'
                f'<td style="padding:5px 10px;font-weight:600;">#{idx+1}{sel}</td>'
                f'<td style="padding:5px 10px;">{sc}</td>'
                f'<td style="padding:5px 10px;">{wc}</td>'
                f'<td style="padding:5px 10px;">{f_str}</td>'
                f'<td style="padding:5px 10px;">{tone_str}</td>'
                f'<td style="padding:5px 10px;">{subj}</td>'
                f'</tr>'
            )
        adv_parts.append(
            f'<div style="margin-bottom:14px;">'
            f'<div style="font-size:12px;font-weight:600;color:var(--muted);margin-bottom:6px;">ALL VARIANTS</div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:12px;color:var(--text-secondary,#d4d4d8);">'
            f'<thead><tr style="border-bottom:1px solid var(--border);color:var(--subtle);">'
            f'<th style="padding:5px 10px;text-align:left;font-size:10px;">VARIANT</th>'
            f'<th style="padding:5px 10px;text-align:left;font-size:10px;">SCORE</th>'
            f'<th style="padding:5px 10px;text-align:left;font-size:10px;">WORDS</th>'
            f'<th style="padding:5px 10px;text-align:left;font-size:10px;">FAITH.</th>'
            f'<th style="padding:5px 10px;text-align:left;font-size:10px;">TONE</th>'
            f'<th style="padding:5px 10px;text-align:left;font-size:10px;">SUBJECT</th>'
            f'</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            f'</div>'
        )

    # Wrap advanced section in a collapsible details/summary
    if adv_parts:
        P.append(
            f'<details style="margin-top:16px;border:1px solid var(--border,rgba(255,255,255,0.07));'
            f'border-radius:12px;padding:0;overflow:hidden;">'
            f'<summary style="padding:12px 16px;cursor:pointer;font-size:12px;font-weight:700;'
            f'color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;'
            f'background:rgba(255,255,255,0.02);user-select:none;">'
            f'&#9881; Advanced Scoring Parameters'
            f'</summary>'
            f'<div style="padding:16px 18px 12px;">{"".join(adv_parts)}</div>'
            f'</details>'
        )

    return "\n".join(P) if P else ""


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------
def _extract_answers_part(text: str) -> str:
    if not text:
        return ""
    marker = "Your answers:"
    lower = text.lower()
    if marker.lower() in lower:
        start = lower.index(marker.lower()) + len(marker)
        return text[start:].strip()
    return text.strip()


def _format_questions_for_answer_box(questions: list[str], existing_answers: str) -> str:
    if not questions:
        return (existing_answers or "").strip()
    q_lines = "\n".join([f"- {q}" for q in questions])
    return f"Questions to answer:\n{q_lines}\n\nYour answers:\n{(existing_answers or '').strip()}"


def build_request(
    raw_notes: str,
    user_answers: str,
    purpose: str,
    audience: str,
    tone: str,
    channel: str,
    include_subject: bool,
    word_size: str,
    style_preset: str,
    draft_variants: int,
    finalize_requested: bool,
    seed: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
) -> DraftRequest:
    extracted_answers = user_answers.strip()
    marker = "Your answers:"
    lower = extracted_answers.lower()
    if marker.lower() in lower:
        start = lower.index(marker.lower()) + len(marker)
        extracted_answers = extracted_answers[start:].strip()

    return DraftRequest(
        raw_notes=raw_notes.strip(),
        user_answers=extracted_answers,
        purpose=purpose.strip(),
        audience=audience.strip(),
        tone=tone.strip(),
        channel=channel.strip(),
        include_subject=(include_subject and channel.strip().lower() == "email"),
        word_size=word_size,
        style_preset=style_preset,
        draft_variants=draft_variants,
        finalize_requested=finalize_requested,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )


def _preset_params(style_preset: str, creativity_intensity: float) -> dict:
    """Convert preset + intensity (0-100) into Ollama sampling parameters."""
    t = max(0.0, min(100.0, float(creativity_intensity))) / 100.0
    preset = style_preset.strip().lower()

    if preset == "professional":
        temperature = 0.4 + 0.5 * t
        top_p = 0.80 + 0.12 * t
        top_k = int(20 + 25 * t)
        repeat_penalty = 1.12 + 0.10 * t
        presence_penalty = 0.0 + 0.15 * t
        frequency_penalty = 0.0 + 0.10 * t
    elif preset == "friendly":
        temperature = 0.6 + 0.7 * t
        top_p = 0.88 + 0.10 * t
        top_k = int(30 + 45 * t)
        repeat_penalty = 1.08 + 0.14 * t
        presence_penalty = 0.1 + 0.35 * t
        frequency_penalty = 0.0 + 0.18 * t
    elif preset == "persuasive":
        temperature = 0.75 + 0.75 * t
        top_p = 0.90 + 0.09 * t
        top_k = int(40 + 70 * t)
        repeat_penalty = 1.10 + 0.18 * t
        presence_penalty = 0.15 + 0.60 * t
        frequency_penalty = 0.05 + 0.30 * t
    else:
        temperature = 1.05 + 0.95 * t
        top_p = 0.92 + 0.08 * t
        top_k = int(80 + 80 * t)
        repeat_penalty = 1.04 + 0.16 * t
        presence_penalty = 0.35 + 0.90 * t
        frequency_penalty = 0.10 + 0.60 * t

    return {
        "temperature": max(0.0, min(2.0, float(temperature))),
        "top_p": max(0.0, min(1.0, float(top_p))),
        "top_k": max(0, min(200, int(top_k))),
        "repeat_penalty": max(0.0, min(2.5, float(repeat_penalty))),
        "presence_penalty": max(-2.0, min(2.5, float(presence_penalty))),
        "frequency_penalty": max(-2.0, min(2.5, float(frequency_penalty))),
    }


def _empty_draft_output():
    return "", "", "", "", "", _build_step_indicator(0)


def generate_draft(
    raw_notes: str,
    user_answers: str,
    purpose: str,
    audience: str,
    tone: str,
    channel: str,
    include_subject: bool,
    word_size: str,
    draft_variants: int,
    finalize_requested: bool,
    generation_mode: str,
    style_preset: str,
    creativity_intensity: float,
    randomize_seed: bool,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    llm_provider: str = "Mock (demo)",
    openrouter_model: str = "mistralai/mistral-7b-instruct",
):
    # -- Input validation --
    if not raw_notes or not raw_notes.strip():
        gr.Warning("Please enter your raw notes before generating a draft.")
        return _empty_draft_output()
    if len(raw_notes.strip()) < 10:
        gr.Warning("Your notes are very short. Consider adding more detail for a better draft.")
    if llm_provider and llm_provider.strip().lower().startswith("openrouter") and not os.getenv("OPENROUTER_API_KEY"):
        gr.Warning("OpenRouter API key not set. Add OPENROUTER_API_KEY to your .env file or switch to Mock.")
        return _empty_draft_output()

    if generation_mode.strip().lower().startswith("preset"):
        params = _preset_params(style_preset=style_preset, creativity_intensity=creativity_intensity)
        temperature = params["temperature"]
        top_p = params["top_p"]
        top_k = params["top_k"]
        repeat_penalty = params["repeat_penalty"]
        presence_penalty = params["presence_penalty"]
        frequency_penalty = params["frequency_penalty"]

    seed_value = random.randint(0, 2**31 - 1) if randomize_seed else seed

    req = build_request(
        raw_notes=raw_notes,
        user_answers=user_answers,
        purpose=purpose,
        audience=audience,
        tone=tone,
        channel=channel,
        include_subject=include_subject,
        word_size=word_size,
        style_preset=style_preset,
        draft_variants=draft_variants,
        finalize_requested=finalize_requested,
        seed=seed_value,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    provider = (llm_provider or "").strip().lower()
    try:
        if provider.startswith("openrouter"):
            llm_client = get_openrouter_client(model_name=openrouter_model)
        elif provider.startswith("ollama"):
            model_name = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
            llm_client = get_ollama_client(model_name=model_name)
        else:
            llm_client = MockLLMClient()
        resp = run_draft_to_ready(req, llm_client=llm_client)
    except _requests_lib.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            gr.Warning("Invalid OpenRouter API key. Check your .env file or switch to Mock mode.")
        elif e.response is not None and e.response.status_code == 429:
            gr.Warning("Rate limit reached. Wait a moment and try again.")
        else:
            gr.Warning(f"API error: {getattr(e.response, 'status_code', 'unknown')}")
        return _empty_draft_output()
    except _requests_lib.exceptions.ConnectionError:
        if provider.startswith("ollama"):
            gr.Warning("Ollama is not running. Start it with 'ollama serve' or switch to OpenRouter/Mock.")
        else:
            gr.Warning("Connection failed. Check your network or switch providers.")
        return _empty_draft_output()
    except ValueError as e:
        gr.Warning(str(e))
        return _empty_draft_output()
    except Exception as e:
        gr.Warning(f"Unexpected error: {str(e)[:200]}")
        return _empty_draft_output()

    existing_answers = _extract_answers_part(user_answers or "")

    # Determine workflow step for indicator
    if resp.final:
        step = 3
        user_answers_out = ""
    elif resp.draft:
        step = 2
        user_answers_out = _format_questions_for_answer_box(resp.questions, existing_answers)
    elif resp.questions:
        step = 1
        user_answers_out = _format_questions_for_answer_box(resp.questions, existing_answers)
    else:
        step = 0
        user_answers_out = _format_questions_for_answer_box(resp.questions, existing_answers)

    step_html = _build_step_indicator(step)
    questions_html = _render_questions_html(resp.questions)
    rubric_html = _render_rubric_html(resp.rubric_check)

    return questions_html, resp.draft, rubric_html, resp.final, user_answers_out, step_html


# ---------------------------------------------------------------------------
# Force dark mode JS
# ---------------------------------------------------------------------------
FORCE_DARK_JS = "() => { document.documentElement.classList.add('dark'); }"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Draft-to-Ready Writing Agent", js=FORCE_DARK_JS) as demo:

    # ===================================================================
    # PAGE 1: LANDING PAGE
    # ===================================================================
    with gr.Column(visible=True, elem_id="landing-page") as landing_page:
        gr.HTML(LANDING_PAGE_HTML)
        btn_get_started = gr.Button(
            "Get Started",
            variant="primary",
            elem_classes=["get-started-btn"],
        )

    # ===================================================================
    # PAGE 2: APP PAGE
    # ===================================================================
    with gr.Column(visible=False, elem_id="app-page") as app_page:

        # -- Slim header with back button --
        with gr.Row():
            gr.HTML(APP_HEADER_HTML)
            btn_back = gr.Button("Back", variant="stop", elem_classes=["back-btn"])

        # -- Main 30/70 layout --
        with gr.Row():

            # ==================== LEFT SIDEBAR (30%) ====================
            with gr.Column(scale=3, min_width=320):

                # Card: Content
                with gr.Group(elem_classes=["card-section"]):
                    gr.Markdown("## Content")
                    raw_notes = gr.Textbox(
                        label="Your raw notes / bullets",
                        lines=8,
                        placeholder="I need to email my professor Dr. Smith about extending the deadline for my CS101 assignment. Reason: I was sick last week with the flu and couldn't complete it. Original deadline was March 20.",
                    )
                    user_answers = gr.Textbox(
                        label="Your answers to questions",
                        lines=4,
                        placeholder="Type your answers to the agent's questions here, then click Generate again.",
                    )
                    gr.Markdown('<p style="color:var(--subtle);font-size:12px;margin:4px 0 8px;">After clicking Generate, the agent may ask clarifying questions. Answer them above and click Generate again.</p>')

                # Card: LLM Provider
                with gr.Group(elem_classes=["card-section"]):
                    gr.Markdown("## LLM Provider")
                    _has_key = bool(os.getenv("OPENROUTER_API_KEY"))
                    llm_provider = gr.Dropdown(
                        choices=["OpenRouter (cloud)", "Ollama (local)", "Mock (demo)"],
                        value="OpenRouter (cloud)" if _has_key else "Mock (demo)",
                        label="Provider",
                    )
                    openrouter_model = gr.Dropdown(
                        choices=[
                            "mistralai/mistral-7b-instruct",
                            "meta-llama/llama-3-8b-instruct",
                            "google/gemma-2-9b-it",
                            "qwen/qwen-2.5-7b-instruct",
                        ],
                        value=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
                        label="OpenRouter model",
                        visible=_has_key,
                    )

                # Card: Context
                with gr.Group(elem_classes=["card-section"]):
                    gr.Markdown("## Context")
                    purpose = gr.Textbox(
                        label="Purpose",
                        value="Email reply",
                        placeholder="e.g., deadline extension, apology letter, meeting request, complaint",
                    )
                    audience = gr.Textbox(
                        label="Audience",
                        value="Teacher",
                        placeholder="e.g., Professor Smith, HR department, landlord, customer support",
                    )
                    with gr.Row():
                        tone = gr.Dropdown(
                            choices=["Formal", "Friendly", "Firm", "Humble"],
                            value="Formal",
                            label="Tone",
                        )
                        channel = gr.Dropdown(
                            choices=["Email", "WhatsApp", "Microsoft Teams"],
                            value="Email",
                            label="Channel",
                        )
                    include_subject = gr.Checkbox(
                        label="Include subject line",
                        value=True,
                    )

                # Helper text above generate button
                gr.Markdown('<p style="color:var(--subtle);font-size:12px;text-align:center;margin-bottom:4px;">The agent will ask clarifying questions → generate multiple draft variants → score and select the best one</p>')

                # Generate button
                btn_draft = gr.Button("Generate Draft", variant="primary")

            # ==================== RIGHT MAIN AREA (70%) ====================
            with gr.Column(scale=7, min_width=480):

                # Generation settings (collapsed bar)
                with gr.Accordion(
                    "Generation Settings",
                    open=False,
                    elem_classes=["gen-settings-bar"],
                ):
                    with gr.Row():
                        word_size = gr.Dropdown(
                            choices=["Small", "Medium", "Large"],
                            value="Medium",
                            label="Message size",
                        )
                        draft_variants = gr.Slider(
                            label="Draft variants",
                            minimum=1, maximum=5, step=1, value=3,
                        )
                        generation_mode = gr.Dropdown(
                            choices=["Preset (recommended)", "Custom (advanced)"],
                            value="Preset (recommended)",
                            label="Generation mode",
                        )
                        style_preset = gr.Dropdown(
                            choices=["Professional", "Friendly", "Persuasive", "Creative"],
                            value="Professional",
                            label="Style preset",
                        )
                    with gr.Row():
                        creativity_intensity = gr.Slider(
                            label="Creativity intensity",
                            minimum=0, maximum=100, step=1, value=35,
                        )
                        randomize_seed = gr.Checkbox(label="Auto-randomize seed", value=True)
                        seed = gr.Number(label="Seed", value=42, precision=0)

                    with gr.Accordion(
                        "Advanced settings (Custom mode only)",
                        open=False,
                        visible=False,
                    ) as advanced_accordion:
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0, maximum=2.0, step=0.05, value=0.7,
                        )
                        with gr.Row():
                            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                            top_k = gr.Slider(label="top_k", minimum=0, maximum=200, step=1, value=40)
                        with gr.Row():
                            repeat_penalty = gr.Slider(label="repeat_penalty", minimum=0.0, maximum=2.5, step=0.05, value=1.1)
                            presence_penalty = gr.Slider(label="presence_penalty", minimum=-2.0, maximum=2.5, step=0.05, value=0.0)
                        frequency_penalty = gr.Slider(label="frequency_penalty", minimum=-2.0, maximum=2.5, step=0.05, value=0.0)

                # Step indicator
                step_indicator = gr.HTML(value=_build_step_indicator(0))

                # Card: Draft Stage
                with gr.Group(elem_classes=["card-section"]):
                    gr.Markdown("## Draft Stage")
                    gr.Markdown('<p style="color:var(--subtle);font-size:12px;margin:4px 0 8px;">Multiple variants generated and scored. The highest-scoring draft is shown based on faithfulness, tone, and formatting.</p>')
                    questions_out = gr.HTML(label="Clarifying questions")
                    draft_out = gr.Textbox(label="Polished draft", lines=12)

                # Card: Finalization
                with gr.Group(elem_classes=["card-section"]):
                    gr.Markdown("## Finalization")
                    gr.Markdown('<p style="color:var(--subtle);font-size:12px;margin:4px 0 8px;">Click Finalize Draft to run an editing pass that addresses issues found by the self-check rubric.</p>')
                    rubric_out = gr.HTML(label="Self-check rubric")
                    final_out = gr.Textbox(label="Final version", lines=12)

                # Action buttons
                with gr.Row():
                    btn_finalize = gr.Button("Finalize Draft", variant="secondary")
                    btn_new = gr.Button("New Draft", variant="stop")

    # ===================================================================
    # Event Handlers
    # ===================================================================

    # -- Page navigation --
    def _go_to_app():
        return gr.update(visible=False), gr.update(visible=True)

    def _go_to_landing():
        return gr.update(visible=True), gr.update(visible=False)

    btn_get_started.click(
        fn=_go_to_app,
        inputs=[],
        outputs=[landing_page, app_page],
    )

    btn_back.click(
        fn=_go_to_landing,
        inputs=[],
        outputs=[landing_page, app_page],
    )

    # -- Toggle OpenRouter model dropdown --
    def _toggle_openrouter_model(provider: str):
        return gr.update(visible=provider.startswith("OpenRouter"))

    llm_provider.change(
        fn=_toggle_openrouter_model,
        inputs=[llm_provider],
        outputs=[openrouter_model],
    )

    # -- Toggle advanced settings --
    def _toggle_advanced_settings(mode: str):
        mode_norm = (mode or "").strip().lower()
        is_custom = not mode_norm.startswith("preset")
        return gr.update(visible=is_custom)

    generation_mode.change(
        fn=_toggle_advanced_settings,
        inputs=[generation_mode],
        outputs=[advanced_accordion],
    )

    # -- Draft generation handlers --
    all_inputs = [
        raw_notes, user_answers, purpose, audience, tone,
        channel, include_subject, word_size, draft_variants,
        generation_mode, style_preset, creativity_intensity, randomize_seed, seed,
        temperature, top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty,
        llm_provider, openrouter_model,
    ]
    all_outputs = [questions_out, draft_out, rubric_out, final_out, user_answers, step_indicator]

    def _generate_draft_only(
        raw_notes_: str, user_answers_: str, purpose_: str, audience_: str, tone_: str,
        channel_: str, include_subject_: bool, word_size_: str,
        draft_variants_: int, generation_mode_: str, style_preset_: str,
        creativity_intensity_: float, randomize_seed_: bool, seed_: int,
        temperature_: float, top_p_: float, top_k_: int,
        repeat_penalty_: float, presence_penalty_: float, frequency_penalty_: float,
        llm_provider_: str, openrouter_model_: str,
    ):
        return generate_draft(
            raw_notes=raw_notes_, user_answers=user_answers_, purpose=purpose_,
            audience=audience_, tone=tone_, channel=channel_,
            include_subject=include_subject_, word_size=word_size_,
            draft_variants=draft_variants_, finalize_requested=False,
            generation_mode=generation_mode_, style_preset=style_preset_,
            creativity_intensity=creativity_intensity_, randomize_seed=randomize_seed_,
            seed=seed_, temperature=temperature_, top_p=top_p_, top_k=top_k_,
            repeat_penalty=repeat_penalty_, presence_penalty=presence_penalty_,
            frequency_penalty=frequency_penalty_,
            llm_provider=llm_provider_, openrouter_model=openrouter_model_,
        )

    def _finalize_draft_only(
        raw_notes_: str, user_answers_: str, purpose_: str, audience_: str, tone_: str,
        channel_: str, include_subject_: bool, word_size_: str,
        draft_variants_: int, generation_mode_: str, style_preset_: str,
        creativity_intensity_: float, randomize_seed_: bool, seed_: int,
        temperature_: float, top_p_: float, top_k_: int,
        repeat_penalty_: float, presence_penalty_: float, frequency_penalty_: float,
        llm_provider_: str, openrouter_model_: str,
    ):
        return generate_draft(
            raw_notes=raw_notes_, user_answers=user_answers_, purpose=purpose_,
            audience=audience_, tone=tone_, channel=channel_,
            include_subject=include_subject_, word_size=word_size_,
            draft_variants=draft_variants_, finalize_requested=True,
            generation_mode=generation_mode_, style_preset=style_preset_,
            creativity_intensity=creativity_intensity_, randomize_seed=randomize_seed_,
            seed=seed_, temperature=temperature_, top_p=top_p_, top_k=top_k_,
            repeat_penalty=repeat_penalty_, presence_penalty=presence_penalty_,
            frequency_penalty=frequency_penalty_,
            llm_provider=llm_provider_, openrouter_model=openrouter_model_,
        )

    def _new_draft():
        return "", "", "", "", "", "", _build_step_indicator(0)

    btn_draft.click(
        fn=_generate_draft_only,
        inputs=all_inputs,
        outputs=all_outputs,
    )

    btn_finalize.click(
        fn=_finalize_draft_only,
        inputs=all_inputs,
        outputs=all_outputs,
    )

    btn_new.click(
        fn=_new_draft,
        inputs=[],
        outputs=[raw_notes, user_answers, questions_out, draft_out, rubric_out, final_out, step_indicator],
    )


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, server_name="0.0.0.0")
