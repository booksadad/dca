import json
import streamlit as st
import google.generativeai as genai

def run_institutional_audit(api_key, port_state_str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.1-flash-lite', generation_config={"response_mime_type": "application/json"})
    
    prompt_cro = f"""
    You are a strict Risk Audit system for an institutional quant fund. Do not roleplay.
    Objective: Audit the 'Portfolio State' for constraint violations. 
    CRITICAL INSTRUCTION: The 'audit_explanation' MUST be written in THAI language. Provide a detailed, professional explanation (3-4 sentences).
    Portfolio State: {port_state_str}
    Return ONLY JSON:
    {{
      "risk_level": "LOW", "MEDIUM", or "HIGH",
      "liquidity_concern": boolean,
      "concentration_anomaly_detected": boolean,
      "audit_explanation": "string (in THAI)"
    }}
    """
    
    prompt_pm = f"""
    You are an Alpha Validation system for an institutional quant fund. Do not roleplay.
    Objective: Verify if 'proposed_buys' aligns with high 'Alpha_Score' candidates.
    CRITICAL INSTRUCTION: The 'audit_explanation' MUST be written in THAI language. Provide a detailed, professional explanation (3-4 sentences) on alignment score.
    Portfolio State: {port_state_str}
    Return ONLY JSON:
    {{
      "alpha_alignment_score": float (0.0 to 1.0),
      "missed_opportunities": ["Ticker1"],
      "audit_explanation": "string (in THAI)"
    }}
    """
    
    res_cro = model.generate_content(prompt_cro).text
    res_pm = model.generate_content(prompt_pm).text
    
    return json.loads(res_cro), json.loads(res_pm)
