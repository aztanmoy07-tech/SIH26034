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

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metri Guard</title>
    <script src="https://cdn.tailwindcss.com">
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        if(dropZone) {
            dropZone.onclick = () => fileInput.click();
            dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500', 'bg-blue-50'); };
            dropZone.ondragleave = () => { dropZone.classList.remove('border-blue-500', 'bg-blue-50'); };
            dropZone.ondrop = (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-blue-500', 'bg-blue-50');
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            };
        }

        if(fileInput) {
            fileInput.onchange = () => {
                if (fileInput.files.length) handleFile(fileInput.files[0]);
            };
        }

        let currentImageBase64 = null;
        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageBase64 = e.target.result;
                const preview = document.getElementById('origImgPreview');
                if(preview) {
                    preview.src = currentImageBase64;
                    preview.classList.remove('hidden');
                    document.getElementById('origPlaceholder').classList.add('hidden');
                    document.getElementById('btnExtract').disabled = false;
                }
            };
            reader.readAsDataURL(file);
        }

        async function runAnalysis() {
            if (!currentImageBase64) return;
            goToStep(2);
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image: currentImageBase64, ruleset: 'lm_2011' })
                });
                
                const data = await response.json();
                goToStep(3);
                
                if(data.error) {
                    alert(data.error);
                    return;
                }
                
                // Update verdicts
                const resVerdict = document.getElementById('resVerdict');
                if (data.preliminary_status === "SEVERE_VIOLATION") {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-red-600 text-white';
                } else if (data.preliminary_status === "COMPLIANT") {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-600 text-white';
                } else {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-500 text-white';
                }
                resVerdict.innerText = data.preliminary_status;
                
                // Build findings container
                const container = document.getElementById('findingsContainer');
                container.innerHTML = '';
                
                data.findings.forEach(f => {
                    const statusColor = f.status === 'SEVERE_VIOLATION' ? 'text-red-700 bg-red-50 border-red-200' : 
                                      f.status === 'COMPLIANT' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' :
                                      'text-amber-700 bg-amber-50 border-amber-200';
                    
                    container.innerHTML += 
                        <div class="border border-slate-200 rounded-xl p-4 hover:border-blue-300 transition-colors">
                            <h3 class="font-bold text-slate-800 mb-2">Rule: \</h3>
                            <p class="text-sm text-slate-500 mb-2">Detected from OCR:</p>
                            <div class="bg-slate-50 rounded-lg p-3 border border-slate-200 mb-4 font-mono text-sm">
                                                            </div>
                            <div class="flex items-start gap-3 p-3 rounded-lg text-sm font-bold border ">
                                <p>\: \</p>
                            </div>
                        </div>
                    ;
                });
                
            } catch (err) {
                console.error(err);
                alert("Error connecting to backend");
                goToStep(1);
            }
        }

    </script>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800;900&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Nunito', sans-serif; 
            background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 50%, #f1f8e9 100%);
            color: #0F172A; 
            min-height: 100vh;
        }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .badge-pass { background-color: #DEF7EC; color: #03543F; border: 1px solid #BCF0DA; }
        .badge-minor { background-color: #FEF08A; color: #854D0E; border: 1px solid #FDE047; }
        .badge-severe { background-color: #FDE8E8; color: #9B1C1C; border: 1px solid #FBD5D5; }
        .spinner { border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: #fff; width: 16px; height: 16px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        /* Glassmorphism cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        }
        @keyframes float3D {
            0% { transform: scale(0.5) perspective(800px) rotateX(15deg) rotateY(-15deg); opacity: 0; filter: drop-shadow(0 20px 10px rgba(0,0,0,0.3)); }
            100% { transform: scale(1.1) perspective(800px) rotateX(0deg) rotateY(0deg); opacity: 1; filter: drop-shadow(0 30px 20px rgba(0,0,0,0.15)); }
        }
        .splash-active {
            animation: float3D 1s cubic-bezier(0.1, 0.8, 0.2, 1) forwards;
        }
        #splashScreen {
            transition: background-color 1s ease-in-out, opacity 1s ease-in-out;
        }
        #splashLogo {
            transition: transform 1.2s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.8s ease-in-out;
            transform-origin: center center;
        }
    </style>
</head>
<body class="flex flex-col">

    <div id="splashScreen" class="fixed inset-0 z-[9999] bg-slate-50 flex items-center justify-center">
        <img id="splashLogo" src="/static/images/logo.jpg" alt="MetriGuard Splash" class="w-72 h-auto splash-active rounded-3xl mix-blend-multiply">
    </div>

    <header class="bg-white/60 backdrop-blur-md border-b border-white/50 sticky top-0 z-50">
        <div class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2 flex items-center justify-between min-h-[5.5rem]">
            <div class="flex items-center space-x-3 cursor-pointer transform transition hover:scale-105 z-20">
                <img id="navLogo" src="/static/images/logo.jpg" alt="Metri Guard Logo" class="h-20 w-auto object-contain rounded-xl shadow-md opacity-0 transition-opacity duration-700">
            </div>
            
            <div class="absolute inset-0 flex items-center justify-center pointer-events-none z-10 hidden sm:flex">
                <h1 class="text-4xl md:text-5xl font-black tracking-wider drop-shadow-sm" style="font-family: 'Montserrat', sans-serif;">
                    <span class="text-slate-900">Metri</span> <span class="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-800">Guard</span>
                </h1>
            </div>

            <div id="authHeaderBlock" class="flex items-center gap-4 z-20">
                <button onclick="document.getElementById('loginModal').classList.remove('hidden')" class="text-sm font-bold text-slate-700 hover:text-slate-900 transition-colors px-2">Log in</button>
                <button onclick="document.getElementById('loginModal').classList.remove('hidden')" class="bg-slate-900 hover:bg-slate-800 text-white px-5 py-2 rounded-full font-bold text-sm shadow-md transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:scale-105 active:scale-95">Sign up</button>
            </div>
            <div id="userHeaderBlock" class="hidden flex items-center gap-4 z-20">
                <span class="text-sm font-bold text-slate-700">Welcome, <span id="userNameDisplay" class="text-emerald-700"></span>!</span>
                <button onclick="logout()" class="text-xs text-red-600 font-bold hover:underline px-2">Log out</button>
            </div>
        </div>
    </header>

    <div id="loginModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm transition-opacity">
        <div class="bg-white rounded-3xl shadow-2xl p-8 max-w-sm w-full mx-4 transform transition-all">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-black text-slate-900">Authentication</h2>
                <button onclick="document.getElementById('loginModal').classList.add('hidden')" class="text-slate-400 hover:text-slate-700 transition-colors">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
            </div>
            <div class="space-y-4">
                <div>
                    <label class="block text-xs font-bold text-slate-500 mb-1">Email Address</label>
                    <input type="email" id="authEmail" class="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-sm font-semibold focus:ring-2 focus:ring-emerald-400 outline-none transition-all">
                </div>
                <div>
                    <label class="block text-xs font-bold text-slate-500 mb-1">Password</label>
                    <input type="password" id="authPassword" class="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-sm font-semibold focus:ring-2 focus:ring-emerald-400 outline-none transition-all">
                </div>
                
                <div id="authError" class="hidden bg-red-50 text-red-600 text-xs font-bold p-2.5 rounded-lg text-center border border-red-200"></div>
                
                <div class="pt-2 flex flex-col gap-3">
                    <button onclick="handleAuth('login')" class="w-full flex justify-center items-center bg-slate-900 hover:bg-slate-800 text-white font-bold py-3 px-4 rounded-xl shadow-md transform transition-all duration-300 hover:-translate-y-1 hover:shadow-lg active:scale-95">
                        Secure Log In
                    </button>
                    <button onclick="handleAuth('signup')" class="w-full flex justify-center items-center bg-emerald-100 hover:bg-emerald-200 text-emerald-800 font-bold py-3 px-4 rounded-xl transition-colors shadow-sm transform transition-all duration-300 hover:-translate-y-1 hover:shadow-md active:scale-95">
                        Create New Account
                    </button>
                </div>
            </div>
            <div class="mt-6 text-center text-xs text-slate-500 font-bold">
                By continuing, you agree to our <button onclick="openContentModal('terms', 'Terms of Service')" class="text-emerald-600 hover:underline">Terms of Service</button> and <button onclick="openContentModal('privacy', 'Privacy Policy')" class="text-emerald-600 hover:underline">Privacy Policy</button>.
            </div>
        </div>
    </div>

    <div class="bg-amber-50 border-b border-amber-200 px-4 py-2 text-center text-xs font-medium text-amber-900">
        ⚠️ <strong>DECISION-SUPPORT NOTICE:</strong> Preliminary compliance analysis under the Legal Metrology Act, 2009. Automated findings require authorised officer sign-off before regulatory action.
    </div>

                </div>

                <div class="mt-4 flex gap-2">
                    <button id="sampleBtn" class="bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-300 text-xs font-bold py-2.5 px-3 rounded-lg flex items-center justify-center gap-1.5 shadow-sm transform transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:scale-105 active:scale-95">
                        <span>🍪</span> Legacy Biscuit
                    </button>
                    <button id="analyzeBtn" class="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold py-2.5 px-3 rounded-lg flex items-center justify-center gap-1.5 shadow-sm transform transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:scale-[1.02] active:scale-95">
                        <span id="btnSpinner" class="spinner hidden"></span>
                        <span id="btnText">🚀 Run Compliance Analysis</span>
                    </button>
                </div>
            </div>

            <div class="glass-card rounded-2xl p-6 transform transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:border-emerald-300/50">
                <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <span class="bg-white p-1.5 rounded-lg shadow-sm border border-slate-100">📊</span> Data Processing Status
                </h2>
                
                <div class="mb-4">
                    <div class="flex justify-between text-xs text-slate-500 mb-1.5 font-bold">
                        <span>Training Data Extracted</span>
                        <span id="miniExtractPct" class="text-emerald-600">—</span>
                    </div>
                    <div class="h-2 bg-slate-100 rounded-full overflow-hidden shadow-inner border border-slate-200">
                        <div id="miniExtractBar" class="h-full bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full transition-all duration-500" style="width:0%"></div>
                    </div>
                    <div class="text-[10px] text-slate-400 mt-1.5 font-bold text-center" id="miniExtractDetail">Initializing…</div>
                </div>

                <div id="miniTrainInfo" class="bg-slate-50 rounded-xl p-3 text-xs font-bold text-slate-500 border border-slate-200 text-center shadow-sm">
                    Connecting to engine...
                </div>
            </div>
        </div>

        <div class="lg:col-span-8 space-y-6">

            <div class="glass-card rounded-2xl p-6 transform transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:border-emerald-300/50">
                <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4 flex items-center justify-between">
                    <div class="flex items-center gap-2"><span class="bg-white p-1.5 rounded-lg shadow-sm border border-slate-100">🖼️</span> Package Segmentation & Visual Detection</div>
                    <span id="qualityBadge" class="text-xs font-bold px-3 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-200 shadow-sm">Awaiting Image</span>
                </h2>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="border border-slate-200/60 rounded-xl p-3 bg-white/50 backdrop-blur text-center shadow-sm">
                        <p class="text-xs font-bold text-slate-700 mb-2">Original Packaging Ingest</p>
                        <div class="h-60 flex items-center justify-center overflow-hidden rounded-lg bg-slate-50 border border-slate-200/60 p-1">
                            <img id="origImgPreview" src="" alt="Original Preview" class="max-h-full max-w-full object-contain hidden transition-opacity duration-300">
                            <span id="origPlaceholder" class="text-xs font-semibold text-slate-400">Click "Load Sample" to start</span>
                        </div>
                    </div>
                    <div class="border border-slate-200/60 rounded-xl p-3 bg-white/50 backdrop-blur text-center shadow-sm">
                        <p class="text-xs font-bold text-slate-700 mb-2">Masked & Label-Isolated</p>
                        <div class="h-60 flex items-center justify-center overflow-hidden rounded-lg bg-slate-50 border border-slate-200/60 p-1">
                            <img id="maskedImgPreview" src="" alt="Masked Preview" class="max-h-full max-w-full object-contain hidden transition-opacity duration-300">
                            <span id="maskedPlaceholder" class="text-xs font-semibold text-slate-400">Isolated Label Preview</span>
                        </div>
                    </div>
                </div>
            </div>

            <div id="verdictBanner" class="hidden rounded-2xl p-6 border shadow-lg backdrop-blur bg-white/90 transform transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl">
                <div class="flex items-start gap-4">
                    <span id="verdictIcon" class="text-4xl animate-bounce"></span>
                    <div class="flex-1">
                        <h3 id="verdictTitle" class="text-2xl font-black"></h3>
                        <p id="verdictDesc" class="text-sm mt-1 text-slate-700 font-bold"></p>
                        
                        <div class="mt-5 grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div class="bg-gradient-to-br from-white to-slate-50 border border-slate-200 rounded-2xl px-4 py-4 text-center shadow-md transform transition duration-300 hover:-translate-y-1 hover:shadow-xl">
                                <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Detected Panel</div>
                                <div id="panelTypeStat" class="text-base font-black text-slate-800 truncate">—</div>
                            </div>
                            <div class="bg-gradient-to-br from-white to-slate-50 border border-slate-200 rounded-2xl px-4 py-4 text-center shadow-md transform transition duration-300 hover:-translate-y-1 hover:shadow-xl">
                                <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">PDP Surface Area</div>
                                <div id="pdpStat" class="text-base font-black text-slate-800">—</div>
                            </div>
                            <div class="bg-gradient-to-br from-red-50 to-red-100 border border-red-200 rounded-2xl px-4 py-4 text-center shadow-md transform transition duration-300 hover:-translate-y-1 hover:shadow-xl">
                                <div class="text-[10px] font-bold text-red-500 uppercase tracking-wider mb-1">Severe Violations</div>
                                <div id="severeStat" class="text-2xl font-black text-red-700 drop-shadow-sm">—</div>
                            </div>
                            <div class="bg-gradient-to-br from-amber-50 to-amber-100 border border-amber-200 rounded-2xl px-4 py-4 text-center shadow-md transform transition duration-300 hover:-translate-y-1 hover:shadow-xl">
                                <div class="text-[10px] font-bold text-amber-600 uppercase tracking-wider mb-1">Minor Infractions</div>
                                <div id="minorStat" class="text-2xl font-black text-amber-700 drop-shadow-sm">—</div>
                            </div>
                        </div>
                        
                        <div id="minFontStat" class="mt-4 inline-block px-4 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs font-bold text-slate-600 shadow-sm"></div>
                    </div>
                </div>
            </div>

            <div class="glass-card rounded-2xl p-4 transform transition-all duration-300 hover:-translate-y-1 hover:shadow-xl">
                <div class="flex items-center gap-2 mb-3">
                    <span class="bg-white p-1.5 rounded-lg shadow-sm border border-slate-100 text-sm">🔍</span>
                    <h2 class="text-xs font-bold text-slate-700 uppercase tracking-wider">OCR Text Extraction — What Was Read From Your Label</h2>
                </div>
                <div id="ocrEngineStatus" class="text-xs text-slate-500 mb-2 font-medium">Run analysis to see OCR status</div>
                <div class="ocr-raw-panel hidden">
                    <div class="bg-slate-900 rounded-xl p-3 max-h-40 overflow-y-auto">
                        <pre id="ocrRawText" class="text-xs text-emerald-300 font-mono whitespace-pre-wrap leading-relaxed"></pre>
                    </div>
                    <p class="text-[10px] text-slate-400 mt-1 font-medium">⬆️ This is the actual text the AI read from your image. Violations are based purely on what is present or absent here.</p>
                </div>
            </div>

            <div class="glass-card rounded-2xl p-6 transform transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:border-emerald-300/50">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                        <span>📋</span> Statutory Compliance Checklist (Rule 6 to Schedule II)
                        <span id="ruleCountBadge" class="text-[10px] font-semibold bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full"></span>
                    </h2>
                    <button id="downloadBtn" class="hidden text-xs bg-slate-900 hover:bg-black text-white font-semibold py-1.5 px-3 rounded-lg shadow-sm transform transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:scale-105 active:scale-95">
                        📥 Download Audit Report
                    </button>
                </div>

                <div id="checklistContainer" class="space-y-3">
                    <div class="p-8 text-center text-slate-400 text-xs">
                        Click <strong class="text-slate-600">"Load Biscuit Sample"</strong> or upload an image to view the complete statutory analysis.
                    </div>
                </div>
            </div>

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

    
    <script>
        let uploadedPanels = {};
        
        function updateProgress(step) {
            const progress = ((step - 1) / 3) * 100;
            document.getElementById('progressLine').style.width = progress + '%';
            
            document.querySelectorAll('.step-indicator').forEach(el => {
                const s = parseInt(el.getAttribute('data-step'));
                const icon = document.getElementById('stepIcon' + s);
                if(s < step) {
                    icon.className = 'w-10 h-10 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold shadow-md transition-colors border-4 border-slate-50';
                    icon.innerHTML = '?';
                } else if(s === step) {
                    icon.className = 'w-10 h-10 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold shadow-md transition-colors border-4 border-emerald-100';
                    icon.innerHTML = s;
                } else {
                    icon.className = 'w-10 h-10 rounded-full bg-slate-200 text-slate-500 flex items-center justify-center font-bold shadow-md transition-colors border-4 border-slate-50';
                    icon.innerHTML = s;
                }
            });
        }

        function goToStep(step) {
            document.querySelectorAll('.step-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.step-content').forEach(el => el.classList.remove('block'));
            
            if(step === 1) document.getElementById('step1Capture').classList.remove('hidden');
            if(step === 2) {
                document.getElementById('step2Extract').classList.remove('hidden');
                document.getElementById('step2Extract').classList.add('flex');
                setTimeout(() => goToStep(3), 2000); // Mock AI extraction for now
            }
            if(step === 3) document.getElementById('step3Verify').classList.remove('hidden');
            if(step === 4) document.getElementById('step4Report').classList.remove('hidden');
            
            updateProgress(step);
        }

        function handlePanelUpload(input, checkId) {
            if(input.files && input.files[0]) {
                document.getElementById(checkId).classList.remove('hidden');
                const reader = new FileReader();
                reader.onload = (e) => {
                    uploadedPanels[checkId] = e.target.result;
                    const count = Object.keys(uploadedPanels).length;
                    document.getElementById('panelsUploadedCount').innerText = count;
                    if(count > 0) {
                        document.getElementById('btnExtract').disabled = false;
                    }
                };
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        function loadDemoSample() {
            // Pre-fill the UI for the SIH demo
            document.getElementById('checkFront').classList.remove('hidden');
            document.getElementById('checkBack').classList.remove('hidden');
            document.getElementById('checkSide').classList.remove('hidden');
            document.getElementById('panelsUploadedCount').innerText = '3';
            document.getElementById('btnExtract').disabled = false;
        }

        function resetCase() {
            uploadedPanels = {};
            document.getElementById('panelsUploadedCount').innerText = '0';
            ['checkFront', 'checkBack', 'checkSide', 'checkQR'].forEach(id => {
                document.getElementById(id).classList.add('hidden');
            });
            goToStep(1);
        }
 // Minimum browser-allowed threshold for near-ns realtime polling
        }
        setTimeout(pollTrainingStatus, 50);
    
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        if(dropZone) {
            dropZone.onclick = () => fileInput.click();
            dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500', 'bg-blue-50'); };
            dropZone.ondragleave = () => { dropZone.classList.remove('border-blue-500', 'bg-blue-50'); };
            dropZone.ondrop = (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-blue-500', 'bg-blue-50');
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            };
        }

        if(fileInput) {
            fileInput.onchange = () => {
                if (fileInput.files.length) handleFile(fileInput.files[0]);
            };
        }

        let currentImageBase64 = null;
        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageBase64 = e.target.result;
                const preview = document.getElementById('origImgPreview');
                if(preview) {
                    preview.src = currentImageBase64;
                    preview.classList.remove('hidden');
                    document.getElementById('origPlaceholder').classList.add('hidden');
                    document.getElementById('btnExtract').disabled = false;
                }
            };
            reader.readAsDataURL(file);
        }

        async function runAnalysis() {
            if (!currentImageBase64) return;
            goToStep(2);
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image: currentImageBase64, ruleset: 'lm_2011' })
                });
                
                const data = await response.json();
                goToStep(3);
                
                if(data.error) {
                    alert(data.error);
                    return;
                }
                
                // Update verdicts
                const resVerdict = document.getElementById('resVerdict');
                if (data.preliminary_status === "SEVERE_VIOLATION") {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-red-600 text-white';
                } else if (data.preliminary_status === "COMPLIANT") {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-600 text-white';
                } else {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-500 text-white';
                }
                resVerdict.innerText = data.preliminary_status;
                
                // Build findings container
                const container = document.getElementById('findingsContainer');
                container.innerHTML = '';
                
                data.findings.forEach(f => {
                    const statusColor = f.status === 'SEVERE_VIOLATION' ? 'text-red-700 bg-red-50 border-red-200' : 
                                      f.status === 'COMPLIANT' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' :
                                      'text-amber-700 bg-amber-50 border-amber-200';
                    
                    container.innerHTML += 
                        <div class="border border-slate-200 rounded-xl p-4 hover:border-blue-300 transition-colors">
                            <h3 class="font-bold text-slate-800 mb-2">Rule: \</h3>
                            <p class="text-sm text-slate-500 mb-2">Detected from OCR:</p>
                            <div class="bg-slate-50 rounded-lg p-3 border border-slate-200 mb-4 font-mono text-sm">
                                                            </div>
                            <div class="flex items-start gap-3 p-3 rounded-lg text-sm font-bold border ">
                                <p>\: \</p>
                            </div>
                        </div>
                    ;
                });
                
            } catch (err) {
                console.error(err);
                alert("Error connecting to backend");
                goToStep(1);
            }
        }

    </script>
    <footer class="mt-12 border-t border-slate-200/60 bg-white/40 backdrop-blur">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
            <div class="flex items-center gap-2 text-slate-500 font-bold text-sm">
                <div class="bg-gradient-to-br from-emerald-400 to-teal-500 text-white p-1.5 rounded-lg shadow-sm">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                </div>
                &copy; 2026 Metri Guard. All rights reserved.
            </div>
            <div class="flex items-center gap-6 text-sm font-bold text-slate-500">
                <button onclick="openContentModal('privacy', 'Privacy Policy')" class="hover:text-slate-900 transition-colors transform hover:-translate-y-0.5">Privacy Policy</button>
                <button onclick="openContentModal('terms', 'Terms of Service')" class="hover:text-slate-900 transition-colors transform hover:-translate-y-0.5">Terms of Service</button>
                <button onclick="openContentModal('contact', 'Contact Support')" class="hover:text-slate-900 transition-colors transform hover:-translate-y-0.5">Contact</button>
            </div>
        </div>
    </footer>

    <div id="contentModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm transition-opacity">
        <div class="bg-white rounded-3xl shadow-2xl p-8 max-w-2xl w-full mx-4 transform transition-all flex flex-col max-h-[80vh]">
            <div class="flex justify-between items-center mb-4">
                <h2 id="contentModalTitle" class="text-2xl font-black text-slate-900">Title</h2>
                <div class="flex items-center gap-3">
                    <button id="adminEditBtn" onclick="toggleEditMode()" class="text-xs font-bold bg-amber-100 text-amber-700 px-3 py-1.5 rounded-lg hover:bg-amber-200 transition-colors flex items-center gap-1 shadow-sm transform transition hover:-translate-y-0.5"><span class="animate-pulse">🛡️</span> Admin Edit</button>
                    <button onclick="closeContentModal()" class="text-slate-400 hover:text-slate-700 transition-colors">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto mb-4 bg-slate-50 rounded-xl p-4 border border-slate-100">
                <div id="contentDisplay" class="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed font-medium">Loading...</div>
                <textarea id="contentEditor" class="hidden w-full h-64 text-sm p-3 border border-emerald-300 rounded-lg focus:ring-2 focus:ring-emerald-400 outline-none bg-white font-mono shadow-inner resize-none"></textarea>
            </div>
            <div id="contentSaveContainer" class="hidden flex justify-end gap-3">
                <button onclick="toggleEditMode()" class="px-4 py-2 font-bold text-slate-500 hover:text-slate-700">Cancel</button>
                <button onclick="saveContent()" class="bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-6 rounded-xl shadow-md transform transition hover:-translate-y-0.5">Save Changes</button>
            </div>
        </div>
    </div>

    <script>
        let currentContentPage = '';
        
        async function openContentModal(pageId, title) {
            currentContentPage = pageId;
            document.getElementById('contentModalTitle').innerText = title;
            document.getElementById('contentModal').classList.remove('hidden');
            document.getElementById('contentDisplay').innerText = 'Loading...';
            document.getElementById('contentDisplay').classList.remove('hidden');
            document.getElementById('contentEditor').classList.add('hidden');
            document.getElementById('contentSaveContainer').classList.add('hidden');
            document.getElementById('contentSaveContainer').classList.remove('flex');
            
            try {
                const res = await fetch(`/api/content/${pageId}`);
                const data = await res.json();
                document.getElementById('contentDisplay').innerText = data.text;
                document.getElementById('contentEditor').value = data.text;
            } catch (e) {
                document.getElementById('contentDisplay').innerText = 'Error loading content.';
            }
        }
        
        function closeContentModal() {
            document.getElementById('contentModal').classList.add('hidden');
        }
        
        function toggleEditMode() {
            const isEditing = !document.getElementById('contentEditor').classList.contains('hidden');
            if (isEditing) {
                document.getElementById('contentEditor').classList.add('hidden');
                document.getElementById('contentDisplay').classList.remove('hidden');
                document.getElementById('contentSaveContainer').classList.add('hidden');
                document.getElementById('contentSaveContainer').classList.remove('flex');
            } else {
                document.getElementById('contentEditor').classList.remove('hidden');
                document.getElementById('contentDisplay').classList.add('hidden');
                document.getElementById('contentSaveContainer').classList.remove('hidden');
                document.getElementById('contentSaveContainer').classList.add('flex');
            }
        }
        
        async function saveContent() {
            const newText = document.getElementById('contentEditor').value;
            const btn = document.querySelector('#contentSaveContainer button:last-child');
            btn.innerText = 'Saving...';
            try {
                await fetch(`/api/content/${currentContentPage}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: newText})
                });
                document.getElementById('contentDisplay').innerText = newText;
                toggleEditMode();
            } catch(e) {
                alert('Failed to save content');
            }
            btn.innerText = 'Save Changes';
        }
        async function handleAuth(action) {
            const email = document.getElementById('authEmail').value.trim();
            const password = document.getElementById('authPassword').value;
            const errorDiv = document.getElementById('authError');
            
            if(!email || !password) {
                errorDiv.innerText = "Please fill out all fields.";
                errorDiv.classList.remove('hidden');
                return;
            }
            
            errorDiv.classList.add('hidden');
            
            try {
                const res = await fetch(`/api/${action}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                
                if(data.error) {
                    errorDiv.innerText = data.error;
                    errorDiv.classList.remove('hidden');
                } else {
                    document.getElementById('loginModal').classList.add('hidden');
                    document.getElementById('authHeaderBlock').classList.add('hidden');
                    document.getElementById('userHeaderBlock').classList.remove('hidden');
                    document.getElementById('userHeaderBlock').classList.add('flex');
                    document.getElementById('userNameDisplay').innerText = data.user;
                }
            } catch(e) {
                errorDiv.innerText = "Network error occurred.";
                errorDiv.classList.remove('hidden');
            }
        }
        
        function logout() {
            document.getElementById('userHeaderBlock').classList.add('hidden');
            document.getElementById('userHeaderBlock').classList.remove('flex');
            document.getElementById('authHeaderBlock').classList.remove('hidden');
            document.getElementById('authHeaderBlock').classList.add('flex');
            document.getElementById('authEmail').value = '';
            document.getElementById('authPassword').value = '';
        }

        // Splash Screen 3D Animation Logic
        window.addEventListener('load', () => {
            setTimeout(() => {
                const splash = document.getElementById('splashScreen');
                const splashLogo = document.getElementById('splashLogo');
                const navLogo = document.getElementById('navLogo');
                
                if(!splash || !splashLogo || !navLogo) return;

                const targetRect = navLogo.getBoundingClientRect();
                const splashRect = splashLogo.getBoundingClientRect();
                
                // Stop the intro animation and apply precise transform calculation
                splashLogo.style.animation = 'none';
                
                const targetCenterX = targetRect.left + (targetRect.width / 2);
                const targetCenterY = targetRect.top + (targetRect.height / 2);
                const splashCenterX = splashRect.left + (splashRect.width / 2);
                const splashCenterY = splashRect.top + (splashRect.height / 2);
                
                const moveX = targetCenterX - splashCenterX;
                const moveY = targetCenterY - splashCenterY;
                const scale = targetRect.width / splashRect.width;

                splashLogo.style.transform = `translate(${moveX}px, ${moveY}px) scale(${scale})`;
                
                // Fade out background slightly earlier
                splash.style.opacity = '0';
                splash.style.pointerEvents = 'none';

                // Wait for the swoop animation to finish
                setTimeout(() => {
                    navLogo.classList.remove('opacity-0');
                    splashLogo.style.opacity = '0';
                    setTimeout(() => {
                        splash.remove();
                    }, 200);
                }, 1100);
            }, 1200); // Wait 1.2s before flying
        });
    
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        if(dropZone) {
            dropZone.onclick = () => fileInput.click();
            dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500', 'bg-blue-50'); };
            dropZone.ondragleave = () => { dropZone.classList.remove('border-blue-500', 'bg-blue-50'); };
            dropZone.ondrop = (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-blue-500', 'bg-blue-50');
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            };
        }

        if(fileInput) {
            fileInput.onchange = () => {
                if (fileInput.files.length) handleFile(fileInput.files[0]);
            };
        }

        let currentImageBase64 = null;
        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageBase64 = e.target.result;
                const preview = document.getElementById('origImgPreview');
                if(preview) {
                    preview.src = currentImageBase64;
                    preview.classList.remove('hidden');
                    document.getElementById('origPlaceholder').classList.add('hidden');
                    document.getElementById('btnExtract').disabled = false;
                }
            };
            reader.readAsDataURL(file);
        }

        async function runAnalysis() {
            if (!currentImageBase64) return;
            goToStep(2);
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image: currentImageBase64, ruleset: 'lm_2011' })
                });
                
                const data = await response.json();
                goToStep(3);
                
                if(data.error) {
                    alert(data.error);
                    return;
                }
                
                // Update verdicts
                const resVerdict = document.getElementById('resVerdict');
                if (data.preliminary_status === "SEVERE_VIOLATION") {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-red-600 text-white';
                } else if (data.preliminary_status === "COMPLIANT") {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-600 text-white';
                } else {
                    resVerdict.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-500 text-white';
                }
                resVerdict.innerText = data.preliminary_status;
                
                // Build findings container
                const container = document.getElementById('findingsContainer');
                container.innerHTML = '';
                
                data.findings.forEach(f => {
                    const statusColor = f.status === 'SEVERE_VIOLATION' ? 'text-red-700 bg-red-50 border-red-200' : 
                                      f.status === 'COMPLIANT' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' :
                                      'text-amber-700 bg-amber-50 border-amber-200';
                    
                    container.innerHTML += 
                        <div class="border border-slate-200 rounded-xl p-4 hover:border-blue-300 transition-colors">
                            <h3 class="font-bold text-slate-800 mb-2">Rule: \</h3>
                            <p class="text-sm text-slate-500 mb-2">Detected from OCR:</p>
                            <div class="bg-slate-50 rounded-lg p-3 border border-slate-200 mb-4 font-mono text-sm">
                                                            </div>
                            <div class="flex items-start gap-3 p-3 rounded-lg text-sm font-bold border ">
                                <p>\: \</p>
                            </div>
                        </div>
                    ;
                });
                
            } catch (err) {
                console.error(err);
                alert("Error connecting to backend");
                goToStep(1);
            }
        }

    </script>
</body>
</html>
"""

DEFAULT_CONTENT = {
    "privacy": "Metri Guard processes packaging images solely for automated compliance analysis under the Legal Metrology Act, 2009. \n\n1. Data Processing\nImages uploaded by users are processed in memory and immediately discarded after the compliance audit is generated, unless explicitly opted-in for AI model training. \n\n2. Data Sharing\nWe do not share any vendor or package data with third parties. Findings generated by this tool are for decision support only and are strictly confidential between the authorized user and the system.",
    "terms": "Metri Guard provides AI-driven decision-support tools for packaging compliance.\n\n1. Disclaimer of Liability\nOur automated findings do not constitute binding legal advice. All results, notices, and penalty references must be manually verified by an authorized Legal Metrology Officer before any enforcement action is taken.\n\n2. Usage\nBy using this tool, you agree to upload only relevant packaging materials and to not use the system for unauthorized surveillance.",
    "contact": "Department of Legal Metrology Enforcement\n\nFor technical support or regulatory queries regarding the Metri Guard portal:\n\n📧 Email: support@metriguard.gov.in\n📞 Toll-Free Helpline: 1800-11-4000\n🏢 Nodal Office: Legal Metrology Bhavan, New Delhi\n\nAdmin / Nodal Officer:\nName: Mr. Rajeev Sharma\nContact: +91 98765 43210"
}

def get_content():
    content_file = os.path.join(os.path.dirname(__file__), 'content.json')
    if not os.path.exists(content_file):
        with open(content_file, 'w') as f:
            json.dump(DEFAULT_CONTENT, f)
    with open(content_file, 'r') as f:
        return json.load(f)

@app.route('/api/content/<page>', methods=['GET'])
def read_content(page):
    content = get_content()
    return jsonify({"text": content.get(page, "")})

@app.route('/api/content/<page>', methods=['POST'])
def write_content(page):
    data = request.get_json() or {}
    content = get_content()
    content[page] = data.get("text", "")
    content_file = os.path.join(os.path.dirname(__file__), 'content.json')
    with open(content_file, 'w') as f:
        json.dump(content, f)
    return jsonify({"success": True})

import bcrypt

def get_users_db_path():
    return os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    db_path = get_users_db_path()
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {}

def save_users(users_data):
    with open(get_users_db_path(), 'w') as f:
        json.dump(users_data, f)

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
        
    users = load_users()
    if email in users:
        return jsonify({"error": "User already exists"}), 400
        
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    display_name = email.split('@')[0].capitalize()
    users[email] = {"password": hashed, "name": display_name}
    save_users(users)
    
    return jsonify({"success": True, "name": display_name})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    users = load_users()
    if email not in users:
        return jsonify({"error": "Invalid email or password"}), 401
        
    stored_hash = users[email]["password"].encode('utf-8')
    if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
        return jsonify({"error": "Invalid email or password"}), 401
        
    return jsonify({"success": True, "name": users[email]["name"]})

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

    # Build combined OCR text for debug display
    raw_ocr_lines = [t["text"] for t in tokens]
    ocr_engine_used = "RapidOCR" if tokens else "NONE (OCR failed or no text detected)"

    response = {
        "overall_verdict": audit["overall_verdict"],
        "action_headline": audit["action_headline"],
        "action_description": audit["action_description"],
        "panel_type": audit.get("panel_type", "FRONT_PDP"),
        "panel_description": audit.get("panel_description", "Principal Display Panel"),
        "pdp_area_cm2": audit["pdp_area_cm2"],
        "min_font_size_mm": audit["min_font_size_mm"],
        "severe_violations_count": audit["severe_violations_count"],
        "minor_infractions_count": audit["minor_infractions_count"],
        "quality": quality,
        "masked_image": masked_b64,
        "ocr_tokens": [{"text": t["text"], "bbox": t["bbox"], "confidence": t["confidence"]} for t in tokens],
        "ocr_raw_text": "\n".join(raw_ocr_lines),
        "ocr_engine": ocr_engine_used,
        "rule_checks": rule_checks_list
    }

    return jsonify(response)


@app.route('/api/training-status', methods=['GET'])
def training_status():
    """Returns current YOLO11 training progress stats."""
    import os, glob

    runs_base = r"runs/detect"
    # We changed the training name to massive_280k_training
    full_run = os.path.join(runs_base, "metriguard_yolo_runs_full", "massive_280k_training")
    prev_run = os.path.join(runs_base, "metriguard_yolo_runs", "sih26034_yolo11")

    # Pick the most recent run folder
    if os.path.exists(full_run):
        run_dir = full_run
    elif os.path.exists(prev_run):
        run_dir = prev_run
    else:
        return {"status": "Not Started", "epochs": 0, "metrics": {}}

    csv_path = os.path.join(run_dir, "results.csv")
    metrics = {}
    epochs_done = 0
    if os.path.exists(csv_path):
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            epochs_done = len(df)
            if epochs_done > 0:
                last_row = df.iloc[-1]
                # Map YOLOv8/11 metric columns (strip whitespace)
                col_names = [c.strip() for c in df.columns]
                metrics = {
                    "mAP50": round(float(last_row.iloc[6]) * 100, 1) if len(last_row) > 6 else 0.0,
                    "mAP50-95": round(float(last_row.iloc[7]) * 100, 1) if len(last_row) > 7 else 0.0,
                }
        except Exception:
            pass
    
    # Check for weights
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        status_text = "Completed" if epochs_done >= 50 else "Training..."
    else:
        status_text = "Initializing..." if epochs_done == 0 else "Training..."

    # Count extracted dataset images
    full_ds = r"dataset_yolo_full"
    train_count = 0
    val_count = 0
    if os.path.exists(os.path.join(full_ds, "images", "train")):
        train_count = len(glob.glob(os.path.join(full_ds, "images", "train", "*.jpg")))
    if os.path.exists(os.path.join(full_ds, "images", "val")):
        val_count = len(glob.glob(os.path.join(full_ds, "images", "val", "*.jpg")))

    total_actual = train_count + val_count
    return jsonify({
        "full_81k_training": {
            "run_dir": full_run,
            "epochs_completed": epochs_done,
            "best_map50": metrics.get("mAP50", 0.0),
            "weights_ready": os.path.exists(best_pt),
            "best_weights_path": best_pt
        },
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

    best_weights = r"best.pt"
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



# UI Vocabulary standard: Requires manual review, Evidence crop, Detected from OCR
