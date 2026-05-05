#!/usr/bin/env python3
# Script to update medxplain_ui.html with new MedExplain branding and features

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MedExplain – Medical Image Assistant</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--primary:#1B8B8B;--primary-dark:#167474;--bg-white:#ffffff;--bg-light:#f8fafc;--bg-sidebar:#F5F5F5;--border:#E5E7EB;--border-dark:#E0E0E0;--text-primary:#111827;--text-secondary:#6b7280;--text-muted:#9ca3af;--doctor-gradient-start:#ffffff;--doctor-gradient-end:#e9d5ff;--badge-blue:#eff6ff;--badge-blue-border:#3b82f6;--badge-blue-text:#1d4ed8;--success:#10b981;--error:#dc2626;--warning:#f59e0b}
body{font-family:Inter,system-ui,-apple-system,sans-serif;background:var(--bg-white);color:var(--text-primary);height:100vh;overflow:hidden;line-height:1.5}
#loading-screen{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:var(--bg-light);z-index:1000;transition:opacity .3s}
#loading-screen.hidden{opacity:0;pointer-events:none}
.spinner{width:48px;height:48px;border:4px solid var(--border);border-top-color:var(--primary);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:16px}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{color:var(--text-secondary);font-size:1rem;margin-bottom:8px}
.loading-subtext{color:var(--text-muted);font-size:.85rem}
.error-message{color:var(--error);font-size:.85rem;margin-top:16px;max-width:400px;text-align:center;padding:0 20px;display:none}
.error-message.visible{display:block}
.researcher-badge{display:inline-flex;align-items:center;gap:6px;background:var(--badge-blue);border:1px solid var(--badge-blue-border);border-radius:20px;padding:4px 12px;font-size:.72rem;font-weight:600;color:var(--badge-blue-text)}
.app{display:flex;height:100vh;overflow:hidden}
.app.doctor-mode{background:linear-gradient(to top,var(--doctor-gradient-end),var(--doctor-gradient-start))}
.sidebar{width:220px;min-width:220px;background:var(--bg-sidebar);border-right:1px solid var(--border-dark);display:flex;flex-direction:column;padding:16px 12px;gap:4px;overflow-y:auto}
.sb-logo{display:flex;align-items:center;gap:8px;padding:8px 4px 12px;font-weight:700;font-size:1rem;color:var(--primary);border-bottom:1px solid var(--border-dark);margin-bottom:8px}
.sb-logo svg{width:20px;height:20px}
.btn-new{display:flex;align-items:center;justify-content:center;gap:6px;background:var(--primary);color:#fff;border:none;border-radius:8px;padding:10px 12px;font-size:.85rem;font-weight:600;cursor:pointer;width:100%;margin-bottom:12px;transition:background .15s}
.btn-new:hover{background:var(--primary-dark)}
.hist-label{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted);padding:4px;margin-top:8px}
.hist-item{padding:10px;border-radius:7px;cursor:pointer;transition:background .12s;border:1px solid transparent;margin-bottom:4px}
.hist-item:hover{background:#EBEBEB}
.hist-item.active{background:#e0f2fe;border-color:var(--primary)}
.hist-title{font-size:.8rem;font-weight:500;color:var(--text-primary);line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist-date{font-size:.7rem;color:var(--text-muted);margin-top:2px}
.hist-empty{font-size:.8rem;color:var(--text-muted);font-style:italic;padding:8px}
.sb-spacer{flex:1}
.sb-footer{padding:10px 4px 4px;border-top:1px solid var(--border-dark);font-size:.75rem;color:var(--text-muted)}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}
.navbar{display:flex;align-items:center;justify-content:space-between;padding:0 20px;height:56px;background:var(--bg-white);border-bottom:1px solid var(--border);flex-shrink:0}
.nav-left,.nav-right{display:flex;align-items:center;gap:12px}
.nav-select,.nav-input{border:1px solid var(--border);border-radius:7px;padding:6px 10px;font-size:.8rem;color:var(--text-primary);background:var(--bg-white);cursor:pointer;outline:none;font-family:inherit}
.nav-input{cursor:text}
.nav-link{font-size:.78rem;color:var(--primary);text-decoration:none;cursor:pointer;font-weight:500}
.content{flex:1;display:flex;flex-direction:column;align-items:center;padding:20px;overflow-y:auto;gap:16px}
.greeting-section{display:flex;flex-direction:column;align-items:center;gap:12px;margin-bottom:8px}
.greeting-icon{width:64px;height:64px;background:linear-gradient(135deg,var(--primary),var(--primary-dark));border-radius:16px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(27,139,139,.25)}
.greeting-icon svg{width:36px;height:36px;color:#fff}
.greeting-text{font-size:1.1rem;font-weight:500;color:var(--text-primary);text-align:center}
.mode-toggle{display:flex;gap:8px;background:var(--bg-sidebar);padding:4px;border-radius:10px}
.mode-btn{border:none;background:transparent;padding:8px 16px;font-size:.85rem;font-weight:500;color:var(--text-secondary);cursor:pointer;border-radius:7px;transition:all .15s}
.mode-btn:hover{color:var(--text-primary)}
.mode-btn.active{background:var(--bg-white);color:var(--text-primary);box-shadow:0 1px 3px rgba(0,0,0,.1)}
.doctor-note{font-size:.78rem;color:var(--text-secondary);margin-bottom:12px;padding:10px 14px;background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;max-width:600px;width:100%}
.doctor-fields{width:100%;max-width:600px;display:flex;flex-direction:column;gap:8px}
.doctor-fields label{font-size:.78rem;font-weight:600;color:var(--text-secondary)}
.doctor-fields textarea{width:100%;border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-family:inherit;font-size:.85rem;outline:none;resize:vertical;min-height:80px}
.upload-zone{width:100%;max-width:600px;border:2px dashed var(--border);border-radius:12px;padding:40px 20px;text-align:center;cursor:pointer;transition:all .15s;background:var(--bg-light)}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--primary);background:rgba(27,139,139,.05)}
.upload-zone.hidden,.hidden{display:none}
.upload-zone svg{width:48px;height:48px;color:var(--text-muted);margin-bottom:12px}
#file-input{display:none}
.image-preview{position:relative;width:100%;max-width:600px;border-radius:12px;overflow:hidden;box-shadow:0 4px 6px -1px rgba(0,0,0,.1)}
.image-preview img{width:100%;display:block}
.remove-image{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.6);color:#fff;border:none;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:1.2rem}
.chat-area{width:100%;max-width:700px;display:flex;flex-direction:column;gap:12px}
.bubble-user{align-self:flex-end;background:var(--primary);color:#fff;padding:10px 14px;border-radius:12px 12px 4px 12px;max-width:80%;font-size:.9rem}
.bubble-bot{align-self:flex-start;background:var(--bg-white);border:1px solid var(--border);padding:14px;border-radius:12px 12px 12px 4px;max-width:90%;font-size:.9rem;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.bubble-loading{align-self:flex-start;background:var(--bg-light);padding:12px 16px;border-radius:12px;display:flex;align-items:center;gap:6px}
.loading-dot{width:8px;height:8px;border-radius:50%;background:var(--primary);animation:bounce .6s infinite}
@keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
.result-section{margin-top:12px}
.result-section h4{font-size:.75rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
.answer-text{font-size:.95rem;line-height:1.6}
.gradcam-image{margin-top:12px;border-radius:8px;overflow:hidden;border:1px solid var(--border)}
.gradcam-image img{width:100%;display:block}
.classification-table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:8px}
.classification-table th,.classification-table td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
.classification-table th{font-weight:600;color:var(--text-secondary);background:var(--bg-light)}
.differential-list{list-style:none;margin-top:8px}
.differential-list li{padding:6px 0;font-size:.85rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between}
.differential-rank{font-weight:600;color:var(--primary);margin-right:8px}
.input-bar{width:100%;max-width:700px;display:flex;gap:8px;padding:12px;background:var(--bg-white);border:1px solid var(--border);border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.input-bar input{flex:1;border:none;outline:none;font-size:.95rem;padding:4px;font-family:inherit}
.btn-advanced{background:var(--bg-light);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:.8rem;color:var(--text-secondary);cursor:pointer;white-space:nowrap}
.btn-send{background:var(--primary);color:#fff;border:none;border-radius:8px;padding:8px 16px;font-size:.85rem;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px}
.btn-send:hover{background:var(--primary-dark)}
.btn-send:disabled{opacity:.6;cursor:not-allowed}
.advanced-panel{width:100%;max-width:700px;background:var(--bg-white);border:1px solid var(--border);border-radius:12px;padding:16px;box-shadow:0 4px 12px rgba(0,0,0,.08)}
.panel-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border)}
.panel-title{font-size:.95rem;font-weight:600}
.panel-close{background:none;border:none;font-size:1.2rem;color:var(--text-muted);cursor:pointer}
.patient-id-section{display:flex;align-items:center;gap:12px;margin-bottom:16px;padding:12px;background:var(--bg-light);border-radius:8px;border:1px solid var(--border)}
.patient-id-section label{font-size:.8rem;font-weight:600;color:var(--text-secondary);white-space:nowrap}
.patient-id-section input{flex:1;min-width:120px;border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:.85rem;font-family:inherit;outline:none}
.feature-list{display:flex;flex-direction:column;gap:10px;margin-bottom:16px}
.feature-item{display:flex;align-items:flex-start;gap:10px;padding:10px;border-radius:8px;cursor:pointer;transition:background .12s}
.feature-item:hover{background:var(--bg-light)}
.feature-item input{margin-top:2px}
.feature-name{font-size:.85rem;font-weight:500}
.feature-desc{font-size:.75rem;color:var(--text-muted);margin-top:2px}
.vitals-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.vital-input{display:flex;flex-direction:column;gap:4px}
.vital-input label{font-size:.75rem;color:var(--text-secondary);font-weight:500}
.vital-input input{border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:.8rem;outline:none}
.slider-group{display:flex;flex-direction:column;gap:12px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.slider-item{display:flex;flex-direction:column;gap:4px}
.slider-header{display:flex;justify-content:space-between;align-items:center}
.slider-item label{font-size:.8rem;font-weight:500}
.slider-value{font-size:.75rem;color:var(--text-muted)}
.slider-item input[type="range"]{width:100%}
.footer-bar{width:100%;max-width:700px;text-align:center;font-size:.7rem;color:var(--text-muted);padding:12px;border-top:1px solid var(--border);margin-top:auto}
.error-toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--error);color:#fff;padding:12px 20px;border-radius:8px;font-size:.85rem;box-shadow:0 4px 12px rgba(220,38,38,.3);z-index:1000}
@media(max-width:768px){.sidebar{display:none}.content{padding:12px}}
</style>
</head>
<body>
<div id="loading-screen"><div class="spinner"></div><div class="loading-text">MedExplain</div><div class="loading-subtext">Loading Medical Image Assistant...</div><div id="loading-error" class="error-message"></div></div>
<div id="app" class="app">
  <aside class="sidebar">
    <div class="sb-logo"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg>MedExplain</div>
    <button class="btn-new" id="btn-new">+ New Report</button>
    <div class="hist-label">History</div>
    <div id="history-list"><div class="hist-empty">No reports yet</div></div>
    <div class="sb-spacer"></div>
    <div class="sb-footer">MedExplain v1.0</div>
  </aside>
  <main class="main">
    <nav class="navbar">
      <div class="nav-left">
        <select id="modality-select" class="nav-select"><option>Auto Detect</option><option>X-ray</option><option>CT</option><option>MRI</option><option>Ultrasound</option><option>Pathology</option><option>Skin Photo</option></select>
        <select id="model-select" class="nav-select"><option>Ensemble (Classification + VQA)</option><option>TorchXRayVision (Classification only)</option><option>BLIP-VQA (Visual Q&amp;A only)</option></select>
      </div>
      <div class="nav-right"><a href="#" class="nav-link">Help</a></div>
    </nav>
    <div class="content">
      <div class="greeting-section">
        <div class="greeting-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg></div>
        <div class="greeting-text">I am MedExplain, your 24/7 medical image assistant.</div>
      </div>
      <div class="mode-toggle">
        <button class="mode-btn active" data-mode="Standard">Standard</button>
        <button class="mode-btn" data-mode="Doctor Assistant">Doctor Assistant</button>
      </div>
      <div id="doctor-fields" class="doctor-fields" style="display:none;">
        <div class="doctor-note">In Doctor Assistant mode, images are optional. Enter your clinical question below.</div>
        <label>Clinical Question</label>
        <textarea id="clinical-question" placeholder="e.g. What is the most likely diagnosis based on the clinical presentation?"></textarea>
      </div>
      <div id="upload-zone" class="upload-zone">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <div style="font-size:.9rem;color:var(--text-secondary);margin-bottom:8px">Drop a medical image or click to upload</div>
        <div style="font-size:.75rem;color:var(--text-muted)">Supports: JPG, PNG, DICOM</div>
        <input type="file" id="file-input" accept=".jpg,.jpeg,.png,.dcm,image/*">
      </div>
      <div id="image-preview" class="image-preview hidden">
        <img id="preview-img" src="" alt="Preview">
        <button class="remove-image" id="remove-image">×</button>
      </div>
      <div id="chat-area" class="chat-area hidden"></div>
      <div class="input-bar">
        <input type="text" id="question-input" placeholder="Ask a question about the image...">
        <button class="btn-advanced" id="btn-advanced">⚙️ Advanced</button>
        <button class="btn-send" id="btn-send">Send</button>
      </div>
      <div id="advanced-panel" class="advanced-panel hidden">
        <div class="panel-header"><span class="panel-title">Advanced Features</span><button class="panel-close" id="panel-close">×</button></div>
        <div class="patient-id-section">
          <label>Patient ID</label>
          <input type="text" id="patient-id" placeholder="PAT-XXXXXXXX">
          <div class="researcher-badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>Research Mode</div>
        </div>
        <div class="feature-list">
          <label class="feature-item"><input type="checkbox" id="feat-report"><div><div class="feature-name">📋 Report-Aware Context</div><div class="feature-desc">Load prior reports for longitudinal context.</div></div></label>
          <label class="feature-item"><input type="checkbox" id="feat-context"><div><div class="feature-name">🩺 Context-Aware (Vitals/Labs)</div><div class="feature-desc">Include vitals and labs in VQA context.</div></div></label>
          <label class="feature-item"><input type="checkbox" id="feat-longit"><div><div class="feature-name">📅 Longitudinal Comparison</div><div class="feature-desc">Compare current image with prior using SSIM.</div></div></label>
          <label class="feature-item"><input type="checkbox" id="feat-report1"><div><div class="feature-name">📄 One-Click Report</div><div class="feature-desc">Generate structured clinical report.</div></div></label>
          <label class="feature-item"><input type="checkbox" id="feat-diff"><div><div class="feature-name">🔬 Differential Diagnosis</div><div class="feature-desc">Show top-3 differential diagnoses.</div></div></label>
        </div>
        <div class="vitals-grid" id="vitals-grid" style="display:none;">
          <div class="vital-input"><label>BP</label><input type="text" id="vital-bp" placeholder="120/80"></div>
          <div class="vital-input"><label>HR</label><input type="text" id="vital-hr" placeholder="72"></div>
          <div class="vital-input"><label>Temp (°C)</label><input type="text" id="vital-temp" placeholder="37.0"></div>
          <div class="vital-input"><label>SpO2 (%)</label><input type="text" id="vital-spo2" placeholder="98"></div>
          <div class="vital-input"><label>WBC</label><input type="text" id="vital-wbc" placeholder="7.5"></div>
          <div class="vital-input"><label>CRP</label><input type="text" id="vital-crp" placeholder="5"></div>
        </div>
        <div class="slider-group">
          <div class="slider-item"><div class="slider-header"><label>Grad-CAM Strength</label><span id="gradcam-val" class="slider-value">0.45</span></div><input type="range" id="gradcam-slider" min="0" max="1" step="0.05" value="0.45"></div>
          <div class="slider-item"><div class="slider-header"><label>Temperature</label><span id="temp-val" class="slider-value">1.0</span></div><input type="range" id="temp-slider" min="0.1" max="2" step="0.1" value="1.0"></div>
          <div class="slider-item"><div class="slider-header"><label>Max Tokens</label><span id="tokens-val" class="slider-value">64</span></div><input type="range" id="tokens-slider" min="16" max="128" step="8" value="64"></div>
        </div>
      </div>
      <div class="footer-bar">⚠️ MedExplain can make mistakes. Consider checking important information with a qualified clinician. Not for standalone clinical diagnosis.</div>
    </div>
  </main>
</div>
<script>
(function(){"use strict";const API="http://localhost:8000";let currentMode="Standard",currentImage=null,currentImageData="",chat=[],loading=false;const els={loadingScreen:document.getElementById("loading-screen"),loadingError:document.getElementById("loading-error"),app:document.getElementById("app"),btnNew:document.getElementById("btn-new"),historyList:document.getElementById("history-list"),patientId:document.getElementById("patient-id"),modalitySelect:document.getElementById("modality-select"),modelSelect:document.getElementById("model-select"),modeBtns:document.querySelectorAll(".mode-btn"),doctorFields:document.getElementById("doctor-fields"),clinicalQuestion:document.getElementById("clinical-question"),uploadZone:document.getElementById("upload-zone"),fileInput:document.getElementById("file-input"),imagePreview:document.getElementById("image-preview"),previewImg:document.getElementById("preview-img"),removeImage:document.getElementById("remove-image"),chatArea:document.getElementById("chat-area"),questionInput:document.getElementById("question-input"),btnAdvanced:document.getElementById("btn-advanced"),btnSend:document.getElementById("btn-send"),advancedPanel:document.getElementById("advanced-panel"),panelClose:document.getElementById("panel-close"),featContext:document.getElementById("feat-context"),vitalsGrid:document.getElementById("vitals-grid"),gradcamSlider:document.getElementById("gradcam-slider"),gradcamVal:document.getElementById("gradcam-val"),tempSlider:document.getElementById("temp-slider"),tempVal:document.getElementById("temp-val"),tokensSlider:document.getElementById("tokens-slider"),tokensVal:document.getElementById("tokens-val")};function generatePatientId(){const chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";let id="PAT-";for(let i=0;i<8;i++)id+=chars.charAt(Math.floor(Math.random()*chars.length));return id}function showError(message){const toast=document.createElement("div");toast.className="error-toast";toast.textContent=message;document.body.appendChild(toast);setTimeout(()=>toast.remove(),5000)}function updateHistory(){const history=JSON.parse(localStorage.getItem("medexplain_history")||"[]");if(history.length===0){els.historyList.innerHTML='<div class="hist-empty">No reports yet</div>';return}els.historyList.innerHTML=history.map((item,i)=>`<div class="hist-item" data-index="${i}"><div class="hist-title">${item.title||"Analysis"}</div><div class="hist-date">${item.date||""}</div></div>`).join("");els.historyList.querySelectorAll(".hist-item").forEach(item=>{item.addEventListener("click",()=>loadHistoryItem(parseInt(item.dataset.index)))})}function saveToHistory(data){let history=JSON.parse(localStorage.getItem("medexplain_history")||"[]");history.unshift({title:data.classification?.label||"VQA Analysis",date:new Date().toLocaleString(),image:currentImageData,result:data});if(history.length>20)history=history.slice(0,20);localStorage.setItem("medexplain_history",JSON.stringify(history));updateHistory()}function loadHistoryItem(index){const history=JSON.parse(localStorage.getItem("medexplain_history")||"[]");const item=history[index];if(item&&item.image){currentImageData=item.image;els.previewImg.src=item.image;els.uploadZone.classList.add("hidden");els.imagePreview.classList.remove("hidden")}if(item&&item.result)displayResult(item.result)}function displayResult(data){els.chatArea.classList.remove("hidden");let html='<div class="bubble-bot">';if(data.answer||data.vqa_answer)html+=`<div class="answer-text">${data.answer||data.vqa_answer}</div>`;if(data.classification)html+=`<div class="result-section"><h4>Classification</h4><table class="classification-table"><tr><th>Label</th><td>${data.classification.label}</td></tr><tr><th>Confidence</th><td>${(data.classification.confidence*100).toFixed(1)}%</td></tr></table></div>`;if(data.gradcam_b64||data.gradcam_image)html+=`<div class="result-section"><h4>Grad-CAM</h4><div class="gradcam-image"><img src="${data.gradcam_b64||data.gradcam_image}" alt="Grad-CAM"></div></div>`;if(data.differential&&data.differential.length>0){html+='<div class="result-section"><h4>Differential Diagnosis</h4><ul class="differential-list">';data.differential.forEach((d,i)=>{html+=`<li><span><span class="differential-rank">${i+1}</span>${d.label}</span><span>${(d.prob*100).toFixed(1)}%</span></li>`});html+="</ul></div>"}html+="</div>";els.chatArea.innerHTML+=html;els.chatArea.scrollTop=els.chatArea.scrollHeight}function handleFile(file){if(!file)return;const reader=new FileReader();reader.onload=(e)=>{currentImageData=e.target.result;els.previewImg.src=currentImageData;els.uploadZone.classList.add("hidden");els.imagePreview.classList.remove("hidden")};reader.readAsDataURL(file);currentImage=file}async function sendRequest(){const question=els.questionInput.value.trim();const clinicalQuestion=els.clinicalQuestion.value.trim();if(!currentImage&&currentMode==="Standard"){showError("Please upload an image first");return}if(currentMode==="Doctor Assistant"&&!clinicalQuestion&&!question){showError("Please enter a clinical question");return}els.btnSend.disabled=true;els.chatArea.classList.remove("hidden");if(question)els.chatArea.innerHTML+=`<div class="bubble-user">${question}</div>`;els.chatArea.innerHTML+='<div class="bubble-loading"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div></div>';els.chatArea.scrollTop=els.chatArea.scrollHeight;const formData=new FormData();if(currentImage)formData.append("image",currentImage);formData.append("question",question);formData.append("mode",currentMode);formData.append("model_choice",els.modelSelect.value);formData.append("modality",els.modalitySelect.value);formData.append("patient_id",els.patientId.value);formData.append("clinical_question",clinicalQuestion||question);formData.append("reformulate","true");formData.append("gradcam_strength",els.gradcamSlider.value);formData.append("temperature",els.tempSlider.value);formData.append("max_tokens",els.tokensSlider.value);formData.append("feat_report",document.getElementById("feat-report").checked);formData.append("feat_context",els.featContext.checked);formData.append("feat_longit",document.getElementById("feat-longit").checked);formData.append("feat_oneclik",document.getElementById("feat-report1").checked);formData.append("feat_diff",document.getElementById("feat-diff").checked);if(els.featContext.checked){formData.append("bp",document.getElementById("vital-bp").value);formData.append("hr",document.getElementById("vital-hr").value);formData.append("temp",document.getElementById("vital-temp").value);formData.append("spo2",document.getElementById("vital-spo2").value);formData.append("wbc",document.getElementById("vital-wbc").value);formData.append("crp",document.getElementById("vital-crp").value)}try{console.log("[MedExplain] Sending request to",`${API}/api/analyze`);const response=await fetch(`${API}/api/analyze`,{method:"POST",body:formData});if(!response.ok)throw new Error(`Server error: ${response.status}`);const data=await response.json();console.log("[MedExplain] Response received:",data);const loadingBubble=els.chatArea.querySelector(".bubble-loading");if(loadingBubble)loadingBubble.remove();displayResult(data);saveToHistory(data)}catch(err){console.error("[MedExplain] Request failed:",err);const loadingBubble=els.chatArea.querySelector(".bubble-loading");if(loadingBubble)loadingBubble.remove();els.chatArea.innerHTML+=`<div class="bubble-bot" style="color:var(--error)">Error: ${err.message}</div>`;showError("Failed to get response from server. Is the backend running?")}finally{els.btnSend.disabled=false;els.questionInput.value=""}}els.modeBtns.forEach(btn=>{btn.addEventListener("click",()=>{els.modeBtns.forEach(b=>b.classList.remove("active"));btn.classList.add("active");currentMode=btn.dataset.mode;if(currentMode==="Doctor Assistant"){els.app.classList.add("doctor-mode");els.doctorFields.style.display="flex"}else{els.app.classList.remove("doctor-mode");els.doctorFields.style.display="none"}})});els.uploadZone.addEventListener("click",()=>els.fileInput.click());els.fileInput.addEventListener("change",(e)=>handleFile(e.target.files[0]));els.uploadZone.addEventListener("dragover",(e)=>{e.preventDefault();els.uploadZone.classList.add("dragover")});els.uploadZone.addEventListener("dragleave",()=>{els.uploadZone.classList.remove("dragover")});els.uploadZone.addEventListener("drop",(e)=>{e.preventDefault();els.uploadZone.classList.remove("dragover");handleFile(e.dataTransfer.files[0])});els.removeImage.addEventListener("click",()=>{currentImage=null;currentImageData="";els.previewImg.src="";els.imagePreview.classList.add("hidden");els.uploadZone.classList.remove("hidden")});els.btnSend.addEventListener("click",sendRequest);els.questionInput.addEventListener("keypress",(e)=>{if(e.key==="Enter")sendRequest()});els.btnAdvanced.addEventListener("click",()=>{els.advancedPanel.classList.toggle("hidden")});els.panelClose.addEventListener("click",()=>{els.advancedPanel.classList.add("hidden")});els.featContext.addEventListener("change",()=>{els.vitalsGrid.style.display=els.featContext.checked?"grid":"none"});els.gradcamSlider.addEventListener("input",()=>{els.gradcamVal.textContent=els.gradcamSlider.value});els.tempSlider.addEventListener("input",()=>{els.tempVal.textContent=els.tempSlider.value});els.tokensSlider.addEventListener("input",()=>{els.tokensVal.textContent=els.tokensSlider.value});els.btnNew.addEventListener("click",()=>{currentImage=null;currentImageData="";els.previewImg.src="";els.imagePreview.classList.add("hidden");els.uploadZone.classList.remove("hidden");els.chatArea.innerHTML="";els.chatArea.classList.add("hidden");els.questionInput.value="";els.clinicalQuestion.value="";els.patientId.value=generatePatientId()});function init(){console.log("[MedExplain] Initializing...");els.patientId.value=generatePatientId();updateHistory();fetch(`${API}/ping`).then(()=>{console.log("[MedExplain] Backend connected");els.loadingScreen.classList.add("hidden")}).catch(()=>{console.error("[MedExplain] Backend not available");els.loadingError.textContent="Backend not available at "+API;els.loadingError.classList.add("visible");setTimeout(()=>els.loadingScreen.classList.add("hidden"),2000)})}if(document.readyState==="loading"){document.addEventListener("DOMContentLoaded",init)}else{init()}
})();
</script>
</body>
</html>'''

with open('medxplain_ui.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print('medxplain_ui.html updated successfully')
