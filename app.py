import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained Naive Bayes model safely
MODEL_PATH = "naive_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Could not find {MODEL_PATH} in the root directory.")

# HTML Template with Tailwind CSS for a premium, modern dashboard UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Predictive Analytics Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between selection:bg-indigo-500 selection:text-white">

    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-9 w-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                </div>
                <span class="font-bold text-lg tracking-tight bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">Model Intelligence Hub</span>
            </div>
            <span class="text-xs bg-slate-800 text-slate-400 px-3 py-1.5 rounded-full font-medium border border-slate-700/50">v1.6.1 Active</span>
        </div>
    </header>

    <main class="max-w-5xl w-full mx-auto px-4 py-12 flex-grow">
        <div class="text-center max-w-2xl mx-auto mb-10">
            <h1 class="text-3xl font-extrabold tracking-tight sm:text-4xl bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">Predictive Inference Engine</h1>
            <p class="mt-3 text-slate-400 text-sm sm:text-base">Fill out the customer metrics panel below to execute real-time Gaussian Naive Bayes classification.</p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
            <div class="lg:col-span-2 bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 sm:p-8 backdrop-blur shadow-xl">
                <h2 class="text-lg font-semibold text-slate-200 mb-6 flex items-center gap-2">
                    <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
                    Feature Configuration Matrix
                </h2>
                
                <form action="/predict" method="POST" class="space-y-6">
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        
                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Age</label>
                            <input type="number" name="age" required value="{{ inputs.get('age', 34) }}" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                        </div>
                        
                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Tenure Months</label>
                            <input type="number" name="tenure_months" required value="{{ inputs.get('tenure_months', 12) }}" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Avg Order Value ($)</label>
                            <input type="number" step="0.01" name="avg_order_value" required value="{{ inputs.get('avg_order_value', 85.50) }}" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Total Orders</label>
                            <input type="number" name="total_orders" required value="{{ inputs.get('total_orders', 5) }}" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Last Purchase (Days Ago)</label>
                            <input type="number" name="last_purchase_days_ago" required value="{{ inputs.get('last_purchase_days_ago', 4) }}" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Support Tickets Raised</label>
                            <input type="number" name="support_tickets" required value="{{ inputs.get('support_tickets', 0) }}" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Gender</label>
                            <select name="gender" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors text-slate-300">
                                <option value="0" {% if inputs.get('gender') == '0' %}selected{% endif %}>Female (0)</option>
                                <option value="1" {% if inputs.get('gender') == '1' %}selected{% endif %}>Male (1)</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">City Profile Index</label>
                            <select name="city" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors text-slate-300">
                                <option value="0" {% if inputs.get('city') == '0' %}selected{% endif %}>Tier 1 City (0)</option>
                                <option value="1" {% if inputs.get('city') == '1' %}selected{% endif %}>Tier 2 City (1)</option>
                                <option value="2" {% if inputs.get('city') == '2' %}selected{% endif %}>Tier 3 City (2)</option>
                            </select>
                        </div>

                        <div class="sm:col-span-2">
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Subscription Tier</label>
                            <select name="subscription_type" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition-colors text-slate-300">
                                <option value="0" {% if inputs.get('subscription_type') == '0' %}selected{% endif %}>Basic (0)</option>
                                <option value="1" {% if inputs.get('subscription_type') == '1' %}selected{% endif %}>Standard (1)</option>
                                <option value="2" {% if inputs.get('subscription_type') == '2' %}selected{% endif %}>Premium (2)</option>
                            </select>
                        </div>

                    </div>

                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm rounded-xl py-3 shadow-lg shadow-indigo-600/20 hover:shadow-indigo-500/30 transition-all transform hover:-translate-y-0.5">
                        Compute Classification Prediction
                    </button>
                </form>
            </div>

            <div class="space-y-6">
                <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 backdrop-blur shadow-xl min-h-[280px] flex flex-col justify-between">
                    <div>
                        <h2 class="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Inference Output</h2>
                        
                        {% if prediction is not none %}
                            {% if prediction == 1 %}
                            <div class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-xl p-4 mb-4">
                                <div class="flex items-center space-x-2">
                                    <span class="flex h-3 w-3 relative">
                                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                        <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                                    </span>
