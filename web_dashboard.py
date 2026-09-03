"""
=============================================================================
MetriGuard — Full-Stack Interactive Web Dashboard (Flask + Tailwind CSS)
Fast, Responsive, and User-Friendly Legal Metrology Portal.
=============================================================================
"""

import os
import io
import re
import json
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template_string

# Import our CV and Legal Rules Modules
from cv_pipeline import ImagePreprocessor, PackageExtractor
from legal_rules import LegalMetrologyRulesEngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MetriGuard — Legal Metrology Compliance Portal</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #F8FAFC; color: #0F172A; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .badge-pass { background-color: #DEF7EC; color: #03543F; border: 1px solid #BCF0DA; }
        .badge-minor { background-color: #FEF08A; color: #854D0E; border: 1px solid #FDE047; }
        .badge-severe { background-color: #FDE8E8; color: #9B1C1C; border: 1px solid #FBD5D5; }
        .spinner { border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: #fff; width: 16px; height: 16px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Top Navigation Header -->
    <header class="bg-slate-900 text-white shadow-md border-b border-slate-800">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">⚖️</span>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                        MetriGuard
                        <span class="text-[11px] bg-amber-500 text-slate-950 font-bold px-2 py-0.5 rounded">SIH26034 Prototype</span>
                    </h1>
                    <p class="text-xs text-slate-400">Legal Metrology (Packaged Commodities) Rules, 2011 + 2026 Amendments</p>
                </div>
            </div>
            <div class="flex items-center gap-3 text-xs font-medium text-slate-300">
                <span class="bg-slate-800 px-3 py-1.5 rounded-md border border-slate-700">YOLO11 Active Pipeline</span>
                <span class="bg-emerald-950 text-emerald-300 px-3 py-1.5 rounded-md border border-emerald-800 flex items-center gap-1.5">
                    <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span> Engine Online
                </span>
            </div>
        </div>
    </header>

    <!-- Statutory Disclaimer -->
    <div class="bg-amber-50 border-b border-amber-200 px-4 py-2 text-center text-xs font-medium text-amber-900">
        ⚠️ <strong>DECISION-SUPPORT NOTICE:</strong> Preliminary compliance analysis under the Legal Metrology Act, 2009. Automated findings require authorised officer sign-off before regulatory action.
    </div>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-grow grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- Left Column: Controls & Upload (4 cols) -->
        <div class="lg:col-span-4 space-y-6">

            <!-- Package Configuration Card -->
            <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
                <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <span>📐</span> Principal Display Panel (PDP)
                </h2>
                
                <div class="space-y-3 text-xs">
                    <div>
                        <label class="block font-semibold text-slate-700 mb-1">Package Shape</label>
                        <select id="pdpShape" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 font-medium focus:ring-2 focus:ring-blue-500">
                            <option value="rectangular">📦 Rectangular Box / Carton (H × W)</option>
                            <option value="cylindrical">🧴 Cylindrical Can / Bottle (40% × H × C)</option>
                            <option value="irregular">🍿 Irregular Snack Pouch (40% × Surface Area)</option>
                        </select>
                    </div>

                    <div class="grid grid-cols-2 gap-2">
                        <div>
                            <label class="block font-semibold text-slate-700 mb-1">Height (cm)</label>
                            <input type="number" id="pdpHeight" value="15.0" step="0.5" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 font-medium">
                        </div>
                        <div>
                            <label class="block font-semibold text-slate-700 mb-1">Width (cm)</label>
                            <input type="number" id="pdpWidth" value="10.0" step="0.5" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 font-medium">
                        </div>
                    </div>

                    <div class="pt-2">
                        <label class="flex items-center gap-2 cursor-pointer font-medium text-slate-700">
                            <input type="checkbox" id="isFood" checked class="rounded text-blue-600 focus:ring-blue-500">
                            Food Commodity (Enable FSSAI & Veg/Non-Veg Checks)
                        </label>
                    </div>

                    <div>
                        <label class="flex items-center gap-2 cursor-pointer font-medium text-slate-700">
                            <input type="checkbox" id="enableBgMask" checked class="rounded text-blue-600 focus:ring-blue-500">
                            Enable Packaging Background Removal & Masking
                        </label>
                    </div>
                </div>
            </div>

            <!-- Upload Card -->
            <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
                <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <span>📷</span> Ingest Package Image
                </h2>

                <div id="dropZone" class="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-5 text-center cursor-pointer transition-colors bg-slate-50">
                    <svg class="mx-auto h-9 w-9 text-slate-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    <p class="mt-2 text-xs font-semibold text-slate-700">Click or drag package photo here</p>
                    <p class="text-[10px] text-slate-500">JPEG, PNG up to 30MB</p>
                    <input type="file" id="fileInput" accept="image/*" class="hidden">
                </div>
                
                <!-- OR Load Sample -->
                <div class="mt-4 pt-3 border-t border-slate-100">
                    <label class="block font-semibold text-slate-700 mb-1 text-xs">...or quickly load a generated test sample</label>
                    <select id="sampleSelect" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 text-sm font-medium focus:ring-2 focus:ring-blue-500">
                        <option value="">-- Choose a test sample --</option>
                        <option value="sample_1_instant_masala_noodles.png">Sample 1: Instant Masala Noodles (Compliant)</option>
                        <option value="sample_2_potato_chips_cream_&_onion.png">Sample 2: Potato Chips (Non-Compliant)</option>
                        <option value="sample_3_cold_pressed_mustard_oil.png">Sample 3: Mustard Oil (Compliant)</option>
                        <option value="sample_4_premium_full_cream_milk.png">Sample 4: Full Cream Milk (Compliant)</option>
                        <option value="sample_5_mixed_fruit_jam.png">Sample 5: Mixed Fruit Jam (Compliant)</option>
                        <option value="sample_6_spicy_chicken_sausage.png">Sample 6: Chicken Sausage (Non-Veg)</option>
                        <option value="sample_7_tomato_ketchup.png">Sample 7: Tomato Ketchup</option>
                        <option value="sample_8_rich_chocolate_chip_cookies.png">Sample 8: Choco Cookies</option>
                        <option value="sample_9_carbonated_cola_beverage.png">Sample 9: Cola Beverage</option>
                        <option value="sample_10_garam_masala_powder.png">Sample 10: Garam Masala</option>
                    </select>
                </div>

                <div class="mt-4 flex gap-2">
                    <button id="sampleBtn" class="bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-300 text-xs font-bold py-2.5 px-3 rounded-lg transition-colors flex items-center justify-center gap-1.5 shadow-sm">
                        <span>🍪</span> Legacy Biscuit
                    </button>
                    <button id="analyzeBtn" class="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold py-2.5 px-3 rounded-lg transition-colors flex items-center justify-center gap-1.5 shadow-sm">
                        <span id="btnSpinner" class="spinner hidden"></span>
                        <span id="btnText">🚀 Run Compliance Analysis</span>
                    </button>
                </div>
            </div>

            <!-- YOLO11 Training Status Card (Live) -->
            <div class="bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-xl shadow-sm p-5 border border-slate-700">
                <div class="flex items-center justify-between mb-2">
                    <h2 class="text-xs font-bold uppercase tracking-wider text-amber-400">🧠 YOLO11m — Full 81k Training</h2>
                    <span id="trainStatusBadge" class="text-[10px] bg-blue-900 text-blue-300 font-semibold px-2 py-0.5 rounded border border-blue-700">Loading…</span>
                </div>
                <!-- Extraction progress bar -->
                <div class="mb-3">
                    <div class="flex justify-between text-[10px] text-slate-400 mb-1">
                        <span>Dataset Extraction</span>
                        <span id="extractPct">—</span>
                    </div>
                    <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div id="extractBar" class="h-full bg-amber-400 rounded-full transition-all duration-500" style="width:0%"></div>
                    </div>
                    <div class="text-[10px] text-slate-500 mt-1" id="extractDetail">Initializing…</div>
                </div>
                <!-- Training progress -->
                <div class="bg-slate-950 p-2.5 rounded text-[11px] font-mono text-slate-300 mb-3 border border-slate-800" id="trainInfo">
                    Waiting for training to begin…
                </div>
                <div class="text-[11px] text-slate-400 flex items-center justify-between">
                    <span>Weights: <code class="font-mono text-slate-300" id="weightsPath">runs/.../best.pt</code></span>
                    <span id="weightsReady" class="font-semibold text-slate-500">Pending</span>
                </div>
            </div>
            <script>
                async function pollTrainingStatus() {
                    try {
                        const res = await fetch('/api/training-status');
                        const d = await res.json();
                        const ds = d.dataset_extraction;
                        const fullRun = d.full_81k_training;
                        const prevRun = d.previous_30ep_run;

                        // Extraction bar
                        const pct = ds.pct_done || 0;
                        document.getElementById('extractPct').innerText = pct.toFixed(1) + '%';
                        document.getElementById('extractBar').style.width = pct + '%';
                        document.getElementById('extractDetail').innerText =
                            `${ds.total_extracted.toLocaleString()} / ${ds.target_total.toLocaleString()} images extracted (Train: ${ds.train_extracted}, Val: ${ds.val_extracted})`;

                        // Training info
                        const activeRun = fullRun || prevRun;
                        if (activeRun && activeRun.epochs_completed > 0) {
                            document.getElementById('trainInfo').innerText =
                                `Epochs: ${activeRun.epochs_completed} | Best mAP@50: ${(activeRun.best_map50 * 100).toFixed(1)}%`;
                            document.getElementById('trainStatusBadge').innerText = '🟢 Training';
                            document.getElementById('trainStatusBadge').className = 'text-[10px] bg-emerald-900 text-emerald-300 font-semibold px-2 py-0.5 rounded border border-emerald-700';
                        } else if (pct > 0) {
                            document.getElementById('trainInfo').innerText = 'Extracting dataset — training starts after extraction…';
                            document.getElementById('trainStatusBadge').innerText = '⏳ Extracting';
                        }
                        if (activeRun && activeRun.weights_ready) {
                            document.getElementById('weightsReady').innerText = '✅ Ready';
                            document.getElementById('weightsReady').className = 'text-emerald-400 font-semibold';
                            document.getElementById('weightsPath').innerText = 'best.pt available';
                        }
                    } catch(e) {}
                    setTimeout(pollTrainingStatus, 8000);
                }
                pollTrainingStatus();
            </script>

        </div>

        <!-- Right Column: Results & Inspection Findings (8 cols) -->
        <div class="lg:col-span-8 space-y-6">

            <!-- Visual Preview Row (Original vs Background-Masked) -->
            <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
                <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4 flex items-center justify-between">
                    <span>🖼️ Package Segmentation & Visual Detection</span>
                    <span id="qualityBadge" class="text-xs font-semibold px-2.5 py-0.5 rounded bg-slate-100 text-slate-600">Awaiting Image</span>
                </h2>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="border border-slate-200 rounded-lg p-2.5 bg-slate-50 text-center">
                        <p class="text-xs font-bold text-slate-700 mb-2">Original Packaging Ingest</p>
                        <div class="h-60 flex items-center justify-center overflow-hidden rounded bg-white border border-slate-200 p-1">
                            <img id="origImgPreview" src="" alt="Original Preview" class="max-h-full max-w-full object-contain hidden">
                            <span id="origPlaceholder" class="text-xs text-slate-400">Click "Load Biscuit Sample" to start</span>
                        </div>
                    </div>
                    <div class="border border-slate-200 rounded-lg p-2.5 bg-slate-50 text-center">
                        <p class="text-xs font-bold text-slate-700 mb-2">Masked & Label-Isolated (with Detections)</p>
                        <div class="h-60 flex items-center justify-center overflow-hidden rounded bg-white border border-slate-200 p-1">
                            <img id="maskedImgPreview" src="" alt="Masked Preview" class="max-h-full max-w-full object-contain hidden">
                            <span id="maskedPlaceholder" class="text-xs text-slate-400">Isolated Label Preview</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Overall Verdict Banner -->
            <div id="verdictBanner" class="hidden rounded-xl p-5 border shadow-sm">
                <div class="flex items-start gap-4">
                    <span id="verdictIcon" class="text-3xl"></span>
                    <div class="flex-1">
                        <h3 id="verdictTitle" class="text-base font-bold"></h3>
                        <p id="verdictDesc" class="text-xs mt-1 text-slate-700 font-medium"></p>
                        <!-- Compliance stats row -->
                        <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
                            <div class="bg-white/90 border border-slate-300 rounded-lg px-3 py-1.5 text-center">
                                <div class="text-[10px] text-slate-500 uppercase">Panel</div>
                                <div id="panelTypeStat" class="text-xs font-bold text-slate-800 truncate">—</div>
                            </div>
                            <div class="bg-white/90 border border-slate-300 rounded-lg px-3 py-1.5 text-center">
                                <div class="text-[10px] text-slate-500 uppercase">PDP Area</div>
                                <div id="pdpStat" class="text-xs font-bold text-slate-800">—</div>
                            </div>
                            <div class="bg-red-50 border border-red-200 rounded-lg px-3 py-1.5 text-center">
                                <div class="text-[10px] text-red-500 uppercase">Severe</div>
                                <div id="severeStat" class="text-xs font-bold text-red-700">—</div>
                            </div>
                            <div class="bg-yellow-50 border border-yellow-200 rounded-lg px-3 py-1.5 text-center">
                                <div class="text-[10px] text-yellow-600 uppercase">Minor</div>
                                <div id="minorStat" class="text-xs font-bold text-yellow-700">—</div>
                            </div>
                        </div>
                        <div id="minFontStat" class="mt-2 text-[10px] text-slate-500"></div>
                    </div>
                </div>
            </div>

            <!-- Detailed Checklist -->
            <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                        <span>📋</span> Statutory Compliance Checklist (Rule 6 to Schedule II)
                        <span id="ruleCountBadge" class="text-[10px] font-semibold bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full"></span>
                    </h2>
                    <button id="downloadBtn" class="hidden text-xs bg-slate-900 hover:bg-black text-white font-semibold py-1.5 px-3 rounded-lg transition-colors shadow-sm">
                        📥 Download Audit Report
                    </button>
                </div>

                <div id="checklistContainer" class="space-y-3">
                    <div class="p-8 text-center text-slate-400 text-xs">
                        Click <strong class="text-slate-600">"Load Biscuit Sample"</strong> or upload an image to view the complete statutory analysis.
                    </div>
                </div>
            </div>

            <!-- OCR Tokens Viewer -->
            <div id="ocrPanel" class="hidden bg-white rounded-xl shadow-sm border border-slate-200 p-5">
                <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-3 flex items-center gap-2">
                    🔤 OCR Text Extracted from Image
                    <span id="ocrTokenCount" class="text-[10px] font-semibold bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full"></span>
                </h2>
                <div id="ocrTokensTable" class="overflow-x-auto rounded-lg border border-slate-200">
                    <table class="w-full text-xs">
                        <thead class="bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th class="text-left px-3 py-2 font-semibold text-slate-600">#</th>
                                <th class="text-left px-3 py-2 font-semibold text-slate-600">Extracted Text</th>
                                <th class="text-left px-3 py-2 font-semibold text-slate-600">Confidence</th>
                                <th class="text-left px-3 py-2 font-semibold text-slate-600">BBox</th>
                            </tr>
                        </thead>
                        <tbody id="ocrTableBody" class="divide-y divide-slate-100"></tbody>
                    </table>
                </div>
            </div>

        </div>

    </main>

    <footer class="bg-white border-t border-slate-200 py-3 text-center text-xs text-slate-500">
        MetriGuard Legal Metrology Assistant • Smart India Hackathon (SIH26034)
    </footer>

    <!-- Client JavaScript -->
    <script>
        let currentImageBase64 = null;
        let lastAuditReport = null;

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500', 'bg-blue-50'); };
        dropZone.ondragleave = () => { dropZone.classList.remove('border-blue-500', 'bg-blue-50'); };
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500', 'bg-blue-50');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        };

        fileInput.onchange = () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        };

        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageBase64 = e.target.result;
                document.getElementById('origImgPreview').src = currentImageBase64;
                document.getElementById('origImgPreview').classList.remove('hidden');
                document.getElementById('origPlaceholder').classList.add('hidden');
                runAnalysis();
            };
            reader.readAsDataURL(file);
        }

        document.getElementById('sampleSelect').onchange = (e) => {
            if (!e.target.value) return;
            setLoading(true);
            fetch('/api/test-sample/' + e.target.value)
                .then(res => res.json())
                .then(data => {
                    setLoading(false);
                    if(data.error) {
                        alert(data.error);
                        return;
                    }
                    currentImageBase64 = data.image;
                    document.getElementById('origImgPreview').src = currentImageBase64;
                    document.getElementById('origImgPreview').classList.remove('hidden');
                    document.getElementById('origPlaceholder').classList.add('hidden');
                })
                .catch(err => {
                    setLoading(false);
                    alert('Error loading test sample: ' + err.message);
                });
        };

        document.getElementById('sampleBtn').onclick = () => {
            setLoading(true);
            fetch('/api/sample')
                .then(res => res.json())
                .then(data => {
                    currentImageBase64 = data.image;
                    document.getElementById('origImgPreview').src = currentImageBase64;
                    document.getElementById('origImgPreview').classList.remove('hidden');
                    document.getElementById('origPlaceholder').classList.add('hidden');
                    runAnalysis();
                })
                .catch(err => {
                    setLoading(false);
                    alert('Error loading sample: ' + err.message);
                });
        };

        document.getElementById('analyzeBtn').onclick = () => runAnalysis();

        function setLoading(isLoading) {
            const spinner = document.getElementById('btnSpinner');
            const btnText = document.getElementById('btnText');
            const btn = document.getElementById('analyzeBtn');
            const sampleBtn = document.getElementById('sampleBtn');

            if (isLoading) {
                spinner.classList.remove('hidden');
                btnText.innerText = 'Analyzing...';
                btn.disabled = true;
                sampleBtn.disabled = true;
            } else {
                spinner.classList.add('hidden');
                btnText.innerText = '🚀 Run Analysis';
                btn.disabled = false;
                sampleBtn.disabled = false;
            }
        }

        function runAnalysis() {
            if (!currentImageBase64) {
                alert('Please upload or select an image first!');
                return;
            }

            setLoading(true);
            const container = document.getElementById('checklistContainer');
            container.innerHTML = '<div class="p-8 text-center text-blue-600 font-semibold text-xs animate-pulse">Running Neural Segmentation, OCR & Deterministic Rules Engine...</div>';

            const payload = {
                image: currentImageBase64,
                shape: document.getElementById('pdpShape').value,
                height_cm: parseFloat(document.getElementById('pdpHeight').value),
                width_cm: parseFloat(document.getElementById('pdpWidth').value),
                is_food: document.getElementById('isFood').checked,
                enable_masking: document.getElementById('enableBgMask').checked
            };

            fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => {
                if (!res.ok) throw new Error('HTTP ' + res.status + ': Server Error');
                return res.json();
            })
            .then(data => {
                setLoading(false);
                lastAuditReport = data;
                renderResults(data);
            })
            .catch(err => {
                setLoading(false);
                container.innerHTML = `<div class="p-6 text-center text-red-600 text-xs font-semibold bg-red-50 rounded-lg border border-red-200">Analysis Exception: ${err.message}. Check backend console.</div>`;
            });
        }

        function renderResults(data) {
            // Masked Image
            if (data.masked_image) {
                document.getElementById('maskedImgPreview').src = data.masked_image;
                document.getElementById('maskedImgPreview').classList.remove('hidden');
                document.getElementById('maskedPlaceholder').classList.add('hidden');
            }

            // Quality Badge
            const q = data.quality;
            document.getElementById('qualityBadge').innerText = `Sharpness: ${q.blur_score} (${q.quality_grade})`;

            // Verdict Banner
            const vBanner = document.getElementById('verdictBanner');
            const vIcon = document.getElementById('verdictIcon');
            const vTitle = document.getElementById('verdictTitle');
            const vDesc = document.getElementById('verdictDesc');

            vBanner.classList.remove('hidden', 'bg-emerald-50', 'border-emerald-300', 'bg-yellow-50', 'border-yellow-300', 'bg-red-50', 'border-red-300');

            if (data.overall_verdict === 'COMPLIANT') {
                vBanner.classList.add('bg-emerald-50', 'border-emerald-300');
                vIcon.innerText = '✅';
                vTitle.className = 'text-base font-bold text-emerald-950';
            } else if (data.overall_verdict === 'IMPROVEMENT_NOTICE') {
                vBanner.classList.add('bg-yellow-50', 'border-yellow-300');
                vIcon.innerText = '🟡';
                vTitle.className = 'text-base font-bold text-yellow-950';
            } else if (data.overall_verdict === 'PANEL_SCAN_ONLY') {
                vBanner.classList.add('bg-blue-50', 'border-blue-300');
                vIcon.innerText = '🔎';
                vTitle.className = 'text-base font-bold text-blue-950';
            } else {
                vBanner.classList.add('bg-red-50', 'border-red-300');
                vIcon.innerText = '🔴';
                vTitle.className = 'text-base font-bold text-red-950';
            }
            if (data.panel_type && data.panel_type !== 'FRONT_PDP') {
                const panelBadge = `<div class="mt-2.5 text-xs font-bold bg-blue-100 text-blue-900 border border-blue-300 px-3 py-1.5 rounded-lg inline-block">📋 Detected Panel: <em>${data.panel_description || data.panel_type}</em> — Upload front face for full audit</div>`;
                document.getElementById('verdictDesc').insertAdjacentHTML('afterend', panelBadge);
            }

            vTitle.innerText = data.action_headline;
            vDesc.innerText = data.action_description;

            // Populate stats bar
            const panelLabel = (data.panel_type || 'FRONT_PDP').replace(/_/g, ' ');
            document.getElementById('panelTypeStat').innerText = panelLabel;
            document.getElementById('pdpStat').innerText = `${data.pdp_area_cm2} cm²`;
            document.getElementById('severeStat').innerText = `${data.severe_violations_count || 0} violations`;
            document.getElementById('minorStat').innerText = `${data.minor_infractions_count || 0} infractions`;
            document.getElementById('minFontStat').innerText = `Min required font: ${data.min_font_size_mm} mm for this PDP area | Quality: ${data.quality?.quality_grade || '—'}`;

            // OCR Tokens table
            if (data.ocr_tokens && data.ocr_tokens.length > 0) {
                document.getElementById('ocrPanel').classList.remove('hidden');
                document.getElementById('ocrTokenCount').innerText = `${data.ocr_tokens.length} tokens`;
                const tbody = document.getElementById('ocrTableBody');
                tbody.innerHTML = '';
                data.ocr_tokens.forEach((t, i) => {
                    const conf = (t.confidence * 100).toFixed(0);
                    const confColor = conf >= 90 ? 'text-emerald-600' : conf >= 70 ? 'text-yellow-600' : 'text-red-500';
                    const bbox = t.bbox ? `[${t.bbox.join(', ')}]` : '—';
                    tbody.innerHTML += `<tr class="hover:bg-slate-50">
                        <td class="px-3 py-1.5 text-slate-400">${i+1}</td>
                        <td class="px-3 py-1.5 font-medium text-slate-800 max-w-xs truncate" title="${t.text}">${t.text}</td>
                        <td class="px-3 py-1.5 font-mono ${confColor}">${conf}%</td>
                        <td class="px-3 py-1.5 font-mono text-slate-400 text-[10px]">${bbox}</td>
                    </tr>`;
                });
            }

            // Checklist Cards
            const container = document.getElementById('checklistContainer');
            container.innerHTML = '';

            const totalRules = data.rule_checks.length;
            const severeCount = data.rule_checks.filter(r => r.status === 'SEVERE_VIOLATION').length;
            const minorCount = data.rule_checks.filter(r => r.status === 'MINOR_INFRACTION').length;
            document.getElementById('ruleCountBadge').innerText = `${totalRules} checks`;

            data.rule_checks.forEach(r => {

                let badgeClass = 'badge-pass';
                let badgeText = '🟢 COMPLIANT';
                if (r.status === 'MINOR_INFRACTION') {
                    badgeClass = 'badge-minor';
                    badgeText = '🟡 IMPROVEMENT NOTICE';
                } else if (r.status === 'SEVERE_VIOLATION') {
                    badgeClass = 'badge-severe';
                    badgeText = '🔴 SEVERE VIOLATION';
                } else if (r.status === 'NOT_APPLICABLE') {
                    badgeClass = 'bg-slate-200 text-slate-600';
                    badgeText = '⚪ NOT APPLICABLE';
                } else if (r.status === 'INFORMATIONAL') {
                    badgeClass = 'bg-blue-100 text-blue-700 border border-blue-300';
                    badgeText = '🔵 PANEL DETECTED';
                }


                const card = document.createElement('div');
                const borderColor = r.status === 'SEVERE_VIOLATION' ? 'border-red-300 bg-red-50' :
                                    r.status === 'MINOR_INFRACTION' ? 'border-yellow-300 bg-yellow-50' :
                                    r.status === 'COMPLIANT' ? 'border-emerald-200 bg-emerald-50/30' :
                                    r.status === 'INFORMATIONAL' ? 'border-blue-200 bg-blue-50/40' :
                                    'border-slate-200 bg-slate-50';
                card.className = `border rounded-lg p-3.5 text-xs space-y-2 shadow-sm ${borderColor}`;
                card.innerHTML = `
                    <div class="flex items-start justify-between gap-2">
                        <span class="font-bold text-slate-900 leading-snug">${r.rule_title}</span>
                        <span class="shrink-0 px-2.5 py-0.5 rounded-full font-bold text-[10px] ${badgeClass}">${badgeText}</span>
                    </div>
                    <div class="text-slate-500 font-mono text-[10px]">📖 ${r.statutory_ref}</div>

                    ${r.what_was_checked ? `<div class="bg-white/80 border border-slate-200 rounded p-2 text-[11px] text-slate-600"><strong class="text-slate-700">🔍 What Was Checked:</strong> ${r.what_was_checked}</div>` : ''}

                    <div class="grid grid-cols-2 gap-2">
                        <div class="bg-white border border-slate-200 rounded p-2">
                            <div class="text-[9px] font-bold text-slate-400 uppercase mb-0.5">Extracted from Label</div>
                            <div class="font-medium text-slate-800 break-words">${r.extracted_text || '—'}</div>
                            ${r.extracted_value && r.extracted_value !== r.extracted_text ? `<div class="text-[10px] text-slate-500 mt-0.5">Value: <em>${r.extracted_value}</em></div>` : ''}
                        </div>
                        <div class="bg-white border border-slate-200 rounded p-2">
                            <div class="text-[9px] font-bold text-slate-400 uppercase mb-0.5">Regulatory Requirement</div>
                            <div class="font-medium text-slate-700 break-words">${r.required_value || 'See explanation'}</div>
                        </div>
                    </div>

                    <p class="text-slate-700 leading-relaxed">${r.explanation}</p>

                    ${r.remedy_notice ? `<div class="text-amber-900 bg-amber-100/80 border border-amber-300 p-2.5 rounded-lg text-[11px] flex gap-1.5"><span>⚠️</span><span><strong>Jan Vishwas Remedy Notice:</strong> ${r.remedy_notice}</span></div>` : ''}
                    ${r.penalty_ref ? `<div class="text-red-900 bg-red-100/80 border border-red-300 p-2.5 rounded-lg text-[11px] flex gap-1.5"><span>⚖️</span><span><strong>Statutory Penalty:</strong> ${r.penalty_ref}</span></div>` : ''}
                `;
                container.appendChild(card);
            });


            document.getElementById('downloadBtn').classList.remove('hidden');
        }

        document.getElementById('downloadBtn').onclick = () => {
            if (!lastAuditReport) return;
            const blob = new Blob([JSON.stringify(lastAuditReport, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `MetriGuard_Audit_${Date.now()}.json`;
            a.click();
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/test-sample/<filename>', methods=['GET'])
def get_test_sample(filename):
    """Serves the generated test samples."""
    sample_dir = os.path.join(os.path.dirname(__file__), "test_samples")
    file_path = os.path.join(sample_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Sample not found"}), 404
        
    with open(file_path, "rb") as f:
        b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return jsonify({"image": b64})

@app.route('/api/sample', methods=['GET'])
def get_sample():
    """Generates a high-contrast Indian FMCG biscuit packaging label."""
    img = Image.new("RGB", (650, 520), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Outer Package Border
    draw.rectangle([10, 10, 640, 510], outline="#1E293B", width=4)
    
    # Brand & Title
    draw.text((30, 30), "ROYAL BISCUITS — RICH BUTTER DELIGHT", fill="#0F172A")
    
    # Statutory Declarations
    draw.text((30, 80), "Manufactured & Packed by: Royal Agro Foods Pvt Ltd, Plot 14, Okhla, New Delhi 110020", fill="#1E293B")
    draw.text((30, 125), "Generic Name: Butter Biscuits", fill="#1E293B")
    draw.text((30, 170), "Net Quantity: 250 g", fill="#1E293B")
    draw.text((30, 215), "MRP Rs. 75.00 (inclusive of all taxes)", fill="#0F172A")
    draw.text((30, 260), "Unit Sale Price: Rs. 0.30 per g", fill="#1E293B")
    draw.text((30, 300), "Date of Mfg: 08/2026", fill="#1E293B")
    draw.text((30, 345), "Consumer Care: feedback@royalagro.com | Toll-Free: 1800200100", fill="#1E293B")
    draw.text((30, 390), "FSSAI Lic. No. 10018011000142", fill="#15803D")
    
    # Veg Symbol
    draw.rectangle([530, 25, 585, 80], outline="#16A34A", width=3)
    draw.ellipse([543, 38, 572, 67], fill="#16A34A")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return jsonify({"image": b64})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    img_b64 = data.get("image", "")
    shape = data.get("shape", "rectangular")
    h_cm = float(data.get("height_cm", 15.0))
    w_cm = float(data.get("width_cm", 10.0))
    is_food = bool(data.get("is_food", True))
    enable_masking = bool(data.get("enable_masking", True))

    if not img_b64:
        return jsonify({"error": "No image payload provided"}), 400

    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 1. Quality Assessment
    quality = ImagePreprocessor.assess_quality(pil_img)

    # 2. Background Masking / Segmentation
    if enable_masking:
        masked_pil = ImagePreprocessor.remove_background(pil_img)
    else:
        masked_pil = pil_img

    # 3. Spatial Calibration (ArUco Detection)
    ratio, marker_box = ImagePreprocessor.detect_aruco_calibration(pil_img)

    # 4. OCR Extraction
    tokens = PackageExtractor.run_ocr(masked_pil)

    # 5. Deterministic Legal Metrology Rules Evaluation
    audit = LegalMetrologyRulesEngine.evaluate_package(
        ocr_tokens=tokens,
        pdp_shape=shape,
        pdp_height_cm=h_cm,
        pdp_width_cm=w_cm,
        px_to_mm_ratio=ratio,
        is_food_commodity=is_food
    )

    # 6. Draw Bounding Boxes on the Masked Image
    draw_img = masked_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    for t in tokens:
        bbox = t.get("bbox")
        if bbox and len(bbox) == 4:
            draw.rectangle(bbox, outline="#2563EB", width=2)

    buf = io.BytesIO()
    draw_img.save(buf, format="PNG")
    masked_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    rule_checks_list = []
    for r in audit["rule_checks"]:
        rule_checks_list.append({
            "rule_id": r.rule_id,
            "rule_title": r.rule_title,
            "statutory_ref": r.statutory_ref,
            "status": r.status,
            "extracted_text": r.extracted_text,
            "what_was_checked": getattr(r, "what_was_checked", ""),
            "extracted_value": getattr(r, "extracted_value", ""),
            "required_value": getattr(r, "required_value", ""),
            "explanation": r.explanation,
            "remedy_notice": r.remedy_notice,
            "penalty_ref": r.penalty_ref,
            "confidence": getattr(r, "confidence", 1.0)
        })

    response = {
        "overall_verdict": audit["overall_verdict"],
        "action_headline": audit["action_headline"],
        "action_description": audit["action_description"],
        "panel_type": audit.get("panel_type", "FRONT_PDP"),
        "panel_description": audit.get("panel_description", "Principal Display Panel"),
        "pdp_area_cm2": audit["pdp_area_cm2"],
        "min_font_size_mm": audit["min_font_size_mm"],
        "quality": quality,
        "masked_image": masked_b64,
        "ocr_tokens": [{"text": t["text"], "bbox": t["bbox"], "confidence": t["confidence"]} for t in tokens],
        "rule_checks": rule_checks_list
    }

    return jsonify(response)


@app.route('/api/training-status', methods=['GET'])
def training_status():
    """Returns current YOLO11 training progress stats."""
    import os, glob

    runs_base = r"C:\Users\ajtan\runs\detect"
    full_run = os.path.join(runs_base, "metriguard_yolo_runs_full", "full_81k_training")
    prev_run = os.path.join(runs_base, "metriguard_yolo_runs", "sih26034_yolo11")

    def get_run_info(run_dir):
        if not os.path.exists(run_dir):
            return None
        results_csv = os.path.join(run_dir, "results.csv")
        best_pt = os.path.join(run_dir, "weights", "best.pt")
        epoch = 0
        map50 = 0.0
        if os.path.exists(results_csv):
            with open(results_csv) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            epoch = len(lines) - 1
            if len(lines) > 1:
                cols = [c.strip() for c in lines[0].split(",")]
                vals = [v.strip() for v in lines[-1].split(",")]
                try:
                    map_idx = next(i for i, c in enumerate(cols) if "mAP50" in c and "95" not in c)
                    map50 = round(float(vals[map_idx]), 4)
                except Exception:
                    pass
        return {
            "run_dir": run_dir,
            "epochs_completed": epoch,
            "best_map50": map50,
            "weights_ready": os.path.exists(best_pt),
            "best_weights_path": best_pt if os.path.exists(best_pt) else None
        }

    # Count extracted dataset images
    full_ds = r"C:\Users\ajtan\.gemini\antigravity\scratch\metriguard_backend\dataset_yolo_full"
    train_count = 0
    val_count = 0
    if os.path.exists(os.path.join(full_ds, "images", "train")):
        train_count = len(os.listdir(os.path.join(full_ds, "images", "train")))
    if os.path.exists(os.path.join(full_ds, "images", "val")):
        val_count = len(os.listdir(os.path.join(full_ds, "images", "val")))

    total_actual = train_count + val_count
    return jsonify({
        "full_81k_training": get_run_info(full_run),
        "previous_30ep_run": get_run_info(prev_run),
        "dataset_extraction": {
            "target_total": total_actual,
            "train_extracted": train_count,
            "val_extracted": val_count,
            "total_extracted": total_actual,
            "pct_done": 100.0 if total_actual > 0 else 0.0
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict_yolo():
    """Runs YOLO11 inference on uploaded image to detect label regions."""
    import os
    from ultralytics import YOLO

    best_weights = r"C:\Users\ajtan\runs\detect\metriguard_yolo_runs\sih26034_yolo11\weights\best.pt"
    if not os.path.exists(best_weights):
        return jsonify({"error": "Model weights not yet available. Training still in progress."}), 503

    data = request.get_json() or {}
    img_b64 = data.get("image", "")
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    model = YOLO(best_weights)
    results = model.predict(source=np.array(pil_img), conf=0.25, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })

    return jsonify({"detections": detections, "count": len(detections)})


if __name__ == '__main__':
    print("[*] Launching MetriGuard Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)

