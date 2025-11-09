// =================== CONFIG ===================
const API_BASE = "";                 // same-origin
const PREDICT  = "/predict";         // FastAPI endpoint
const FIXED_RADIUS = 10;             // single blob size
// ==============================================

// Global error guard — show in UI
window.addEventListener("error", (e) => {
  setStatus("Error: " + (e.message || e.error?.toString() || "Unknown"));
  console.error("Window error:", e);
});
window.addEventListener("unhandledrejection", (e) => {
  setStatus("Promise error: " + (e.reason?.message || e.reason?.toString() || "Unknown"));
  console.error("Unhandled rejection:", e);
});

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  imgInput: $("img_input"),
  confSlider: $("conf_slider"),
  confVal: $("conf_val"),
  predictBtn: $("predict_btn"),
  exifStatus: $("exif_status"),
  useExifBtn: $("use_exif_btn"),
  imgPreview: $("img_preview"),
  overlay: $("overlay"),
  detCounts: $("det_counts"),
  lat: $("lat_input"),
  lng: $("lng_input"),
  total: $("count_input"),
  label: $("label_input"),
  pickMapBtn: $("pick_map_btn"),
  markBtn: $("mark_btn"),
  clearBtn: $("clear_btn"),
  deleteModeBtn: $("delete_mode_btn"),
  resetLayout: $("resetLayout"),
  leftPane: $("leftPane"),
  rightPane: $("rightPane"),
  gutter: $("gutter"),
  // map search
  mapSearchInput: $("map_search_input"),
  mapSearchBtn: $("map_search_btn"),
  // crowd sourcing
  crowdInput: $("crowd_input"),
  crowdSubmit: $("crowd_submit"),
  crowdStatus: $("crowd_status"),
};

function setStatus(text) {
  if (els.status) els.status.textContent = text;
  console.log("[STATUS]", text);
}
function setCrowdStatus(text){ if(els.crowdStatus) els.crowdStatus.textContent = text; }

let currentFile = null;
let dets = [];
let counts = {};
let exifLL = null;
let picking = false;

// ---------- Resizable Split ----------
(function initSplit(){
  const vertical = window.matchMedia("(max-width: 1024px)").matches;
  if (!vertical){
    let down=false, startX=0, startLeftW=0;
    els.gutter.addEventListener("mousedown", (e)=>{
      down=true; startX = e.clientX; startLeftW = els.leftPane.getBoundingClientRect().width;
      document.body.style.userSelect="none";
    });
    window.addEventListener("mousemove", (e)=>{
      if(!down) return;
      const dx = e.clientX - startX;
      const newW = Math.max(300, startLeftW + dx);
      els.leftPane.style.width = newW + "px";
      window.dispatchEvent(new Event("resize"));
    });
    window.addEventListener("mouseup", ()=>{ down=false; document.body.style.userSelect=""; });
  } else {
    let down=false, startY=0, startTopH=0;
    els.gutter.addEventListener("mousedown", (e)=>{
      down=true; startY=e.clientY; startTopH=els.leftPane.getBoundingClientRect().height;
      document.body.style.userSelect="none";
    });
    window.addEventListener("mousemove", (e)=>{
      if(!down) return;
      const dy = e.clientY - startY;
      const newH = Math.max(220, startTopH + dy);
      els.leftPane.style.height = newH + "px";
      window.dispatchEvent(new Event("resize"));
    });
    window.addEventListener("mouseup", ()=>{ down=false; document.body.style.userSelect=""; });
  }
  els.resetLayout.addEventListener("click", (e)=>{
    e.preventDefault();
    els.leftPane.removeAttribute("style");
    els.rightPane.removeAttribute("style");
    window.dispatchEvent(new Event("resize"));
  });
})();

// ---------- Map ----------
const map = L.map("map", { worldCopyJump: true }).setView([20,0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap"
}).addTo(map);

const blobLayer = L.layerGroup().addTo(map); // holds blob markers
let deleteMode = false;

let pickMarker = null;
function setPickMarker(lat, lng){
  if (pickMarker) pickMarker.setLatLng([lat,lng]);
  else {
    pickMarker = L.marker([lat,lng], {draggable:true}).addTo(map);
    pickMarker.on("dragend", ()=>{
      const p = pickMarker.getLatLng();
      els.lat.value = p.lat.toFixed(6);
      els.lng.value = p.lng.toFixed(6);
    });
  }
  map.setView([lat,lng], Math.max(12, map.getZoom()));
}
map.on("click", (e)=>{
  if (!picking) return;
  const {lat,lng} = e.latlng;
  els.lat.value = lat.toFixed(6);
  els.lng.value = lng.toFixed(6);
  setPickMarker(lat,lng);
  picking = false;
  els.pickMapBtn.textContent = "Pick on Map";
});

// Persisted blobs
function getStored(){ try{return JSON.parse(localStorage.getItem("trash_blobs")||"[]")}catch(_){return []} }
function setStored(a){ localStorage.setItem("trash_blobs", JSON.stringify(a)); }

function ensureBlobIds(){
  const arr = getStored();
  let changed = false;
  for (const b of arr){
    if (!b.id){
      b.id = (crypto.randomUUID?.() || ("id_"+Date.now()+"_"+Math.random().toString(36).slice(2,8)));
      changed = true;
    }
  }
  if (changed) setStored(arr);
}
function removeBlobById(id){
  const arr = getStored();
  const i = arr.findIndex(b => b.id === id);
  if (i >= 0){
    arr.splice(i,1);
    setStored(arr);
  }
  syncMarkersFromStorage(); // redraw with new min/max colors
}

function getMinMaxIncluding(candidateCount = null) {
  let counts = getStored().map(b => +b.count || 0);
  if (Number.isFinite(candidateCount)) counts = [...counts, +candidateCount];
  if (counts.length === 0) return [0, 1];
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  return min === max ? [min, min + 1] : [min, max];
}

// thresholds at 17%, 33%, 50%, 67%, 83%
function colorThresholds(min, max) {
  const r = max - min;
  return [
    min + r * 0.17,
    min + r * 0.33,
    min + r * 0.50,
    min + r * 0.67,
    min + r * 0.83
  ];
}

// 6-level pollution scale
function colorForCount(count, min, max) {
  const [t1, t2, t3, t4, t5] = colorThresholds(min, max);
  if (count <= t1) return { stroke: "#15803d", fill: "#4ade80" }; // very low (bright green)
  if (count <= t2) return { stroke: "#16a34a", fill: "#22c55e" }; // low (green)
  if (count <= t3) return { stroke: "#65a30d", fill: "#84cc16" }; // med-low (yellow-green)
  if (count <= t4) return { stroke: "#ca8a04", fill: "#eab308" }; // medium (yellow)
  if (count <= t5) return { stroke: "#ea580c", fill: "#fb923c" }; // med-high (orange)
  return              { stroke: "#b91c1c", fill: "#ef4444" };     // high (red)
}

// draw a blob using a FIXED size, with color chosen from global min/max
function addBlobScaled(lat, lng, count, label, persist, min, max, id) {
  const { stroke, fill } = colorForCount(+count || 0, min, max);
  const marker = L.circleMarker([lat, lng], {
    radius: FIXED_RADIUS,
    color: stroke,
    fillColor: fill,
    fillOpacity: 0.6,
    weight: 1
  }).addTo(blobLayer);

  const title = label ? `<b>${label}</b><br>` : "";
  marker.bindPopup(`${title}Count: <b>${count}</b><br>${lat.toFixed(5)}, ${lng.toFixed(5)}`);

  // click-to-delete when delete mode is ON
  marker.on("click", () => {
    if (!deleteMode) return;
    if (confirm("Delete this marker?")) removeBlobById(id);
  });

  if (persist) {
    const arr = getStored();
    arr.push({ lat, lng, count, label: label || "", id });
    setStored(arr);
  }
}

// Public API used elsewhere (unchanged signature)
function addBlob(lat, lng, count, label, persist = true) {
  const id = (crypto.randomUUID?.() || ("id_"+Date.now()+"_"+Math.random().toString(36).slice(2,8)));
  const [min, max] = getMinMaxIncluding(count);
  addBlobScaled(lat, lng, count, label, persist, min, max, id);
}

// Redraw every stored blob with a consistent global scale
function syncMarkersFromStorage() {
  blobLayer.clearLayers();
  const arr = getStored();
  const counts = arr.map(b => +b.count || 0);
  const min = counts.length ? Math.min(...counts) : 0;
  const max = counts.length ? Math.max(...counts) : 1;
  for (const b of arr) addBlobScaled(b.lat, b.lng, b.count, b.label, false, min, max, b.id);
}

// Initial render
ensureBlobIds();
syncMarkersFromStorage();

// ---------- Image preview + EXIF ----------
els.imgInput.addEventListener("change", async (e)=>{
  try{
    const f = e.target.files[0];
    currentFile = f || null; dets=[]; counts={}; exifLL=null;
    els.detCounts.textContent = "";
    setStatus("Ready");

    if (!f){ els.imgPreview.removeAttribute("src"); drawOverlay(); els.exifStatus.textContent="EXIF: –"; return; }

    const url = URL.createObjectURL(f);
    els.imgPreview.src = url;
    els.imgPreview.onload = ()=> drawOverlay();

    // EXIF (guarded)
    els.exifStatus.textContent = "EXIF: reading…";
    try{
      const exif = await exifr.parse(f, {gps:true});
      if (exif && typeof exif.latitude==="number" && typeof exif.longitude==="number"){
        exifLL = {lat:exif.latitude, lng:exif.longitude};
        els.exifStatus.innerHTML = `EXIF: <b>${exif.latitude.toFixed(6)}, ${exif.longitude.toFixed(6)}</b>`;
      } else {
        els.exifStatus.textContent = "EXIF: gps not found";
      }
    }catch(err){
      console.warn("EXIF read failed:", err);
      els.exifStatus.textContent = "EXIF: read failed";
    }
  }catch(err){
    setStatus("Image load error");
    console.error(err);
  }
});

els.useExifBtn.addEventListener("click", ()=>{
  if (!exifLL){ alert("No EXIF GPS in this image."); return; }
  els.lat.value = exifLL.lat.toFixed(6);
  els.lng.value = exifLL.lng.toFixed(6);
  setPickMarker(exifLL.lat, exifLL.lng);
});

// ---------- Detection ----------
els.confSlider.addEventListener("input", ()=> els.confVal.textContent = els.confSlider.value );

els.predictBtn.addEventListener("click", async ()=>{
  if (!currentFile){ alert("Upload an image first."); return; }
  els.predictBtn.disabled = true; const old = els.predictBtn.textContent; els.predictBtn.textContent = "Predicting…";
  setStatus("Predicting…");
  try{
    const form = new FormData();
    form.append("image", currentFile);
    form.append("conf", parseFloat(els.confSlider.value)); // YOLO conf / FRCNN score

    const res = await fetch(API_BASE + PREDICT, { method:"POST", body: form });
    const text = await res.text();
    let j;
    try {
      j = JSON.parse(text);
    } catch (e) {
      setStatus("Server returned non-JSON (see console)");
      console.error("Non-JSON response:", text);
      alert("Server returned non-JSON:\n" + text.slice(0,600));
      return;
    }
    if (j.error){
      setStatus("Prediction error");
      alert("Prediction error: " + j.error);
      return;
    }

    dets = Array.isArray(j.detections) ? j.detections : [];
    counts = (j.counts && typeof j.counts==="object") ? j.counts : {};
    els.total.value = Object.values(counts).reduce((a,b)=>a+(+b||0), 0);

    renderCounts(counts);
    drawOverlay(dets);
    setStatus("Done");
  }catch(err){
    setStatus("Predict failed (see console)");
    console.error("Predict failed:", err);
    alert("Predict failed: " + err);
  }finally{
    els.predictBtn.disabled = false; els.predictBtn.textContent = old;
  }
});

function renderCounts(counts){
  try{
    const lines = Object.entries(counts).map(([k,v])=> `${k.padEnd(20," ")} : ${v}`);
    els.detCounts.innerHTML = `<pre>${lines.join("\n") || "(no detections)"}</pre>`;
  }catch(err){
    console.error("renderCounts error:", err);
    els.detCounts.textContent = "(error rendering counts)";
  }
}

// draw boxes over image on a canvas
function drawOverlay(list=[]){
  try{
    const img = els.imgPreview;
    const cvs = els.overlay;
    if (!img || !img.complete || !img.naturalWidth){
      cvs.width = cvs.height = 0;
      return;
    }
    const rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0){
      setTimeout(()=>drawOverlay(list), 50);
      return;
    }
    cvs.width = rect.width;
    cvs.height = rect.height;

    const ctx = cvs.getContext("2d");
    ctx.clearRect(0,0,cvs.width,cvs.height);

    const sx = rect.width  / img.naturalWidth;
    const sy = rect.height / img.naturalHeight;

    for (const d of list){
      if (!Array.isArray(d.box) || d.box.length !== 4) continue;
      const [x1,y1,x2,y2] = d.box.map(Number);
      const X = x1*sx, Y = y1*sy, W = (x2-x1)*sx, H = (y2-y1)*sy;
      ctx.lineWidth = 2; ctx.strokeStyle = "#22c55e"; ctx.fillStyle = "rgba(34,197,94,.15)";
      ctx.strokeRect(X,Y,W,H); ctx.fillRect(X,Y,W,H);
      const label = `${d.class_name ?? d.class_id} ${Number(d.conf||0).toFixed(2)}`;
      ctx.fillStyle="#22c55e"; ctx.font="12px ui-sans-serif"; ctx.fillText(label, X+4, Y+14);
    }
  }catch(err){
    console.error("drawOverlay error:", err);
    setStatus("Overlay draw error");
  }
}

// ---------- Map mark ----------
els.pickMapBtn.addEventListener("click", ()=>{
  picking = !picking;
  els.pickMapBtn.textContent = picking ? "Click on Map…" : "Pick on Map";
});
els.markBtn.addEventListener("click", ()=>{
  const lat = parseFloat(els.lat.value), lng = parseFloat(els.lng.value);
  if (!isFinite(lat) || !isFinite(lng)){ alert("Set a valid latitude & longitude."); return; }
  const cnt = Math.max(0, parseInt(els.total.value||"0",10));
  const label = (els.label.value||"").trim();
  addBlob(lat,lng,cnt,label,true);
  map.setView([lat,lng], Math.max(12, map.getZoom()));
});
els.clearBtn.addEventListener("click", ()=>{
  if (!confirm("Clear all locally stored markers?")) return;
  localStorage.removeItem("trash_blobs");
  syncMarkersFromStorage();
});
if (els.deleteModeBtn){
  els.deleteModeBtn.addEventListener("click", ()=>{
    deleteMode = !deleteMode;
    els.deleteModeBtn.textContent = deleteMode ? "Delete mode: ON (click a blob)" : "Delete mode";
    els.deleteModeBtn.classList.toggle("active", deleteMode);
  });
}

// keep map and overlay correct when pane resizes
window.addEventListener("resize", ()=>{
  map.invalidateSize();
  drawOverlay(dets);
});

// ---------- Map search (geocode) ----------
async function geocodeAndZoom(query){
  if(!query || !query.trim()) return;
  try{
    const url = "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + encodeURIComponent(query.trim());
    const res = await fetch(url, { headers: { "Accept": "application/json" } });
    const data = await res.json();
    if(!Array.isArray(data) || data.length === 0){
      alert("No results for: " + query);
      return;
    }
    const { lat, lon, display_name } = data[0];
    const latNum = parseFloat(lat), lonNum = parseFloat(lon);
    if(Number.isFinite(latNum) && Number.isFinite(lonNum)){
      setPickMarker(latNum, lonNum);
      map.setView([latNum, lonNum], 12);
      els.lat.value = latNum.toFixed(6);
      els.lng.value = lonNum.toFixed(6);
      setStatus("Moved to: " + (display_name || query));
    }
  }catch(err){
    console.error("Geocode error:", err);
    alert("Search failed. Check your internet connection.");
  }
}
if (els.mapSearchBtn && els.mapSearchInput){
  els.mapSearchBtn.addEventListener("click", ()=> geocodeAndZoom(els.mapSearchInput.value) );
  els.mapSearchInput.addEventListener("keydown", (e)=>{
    if(e.key === "Enter") geocodeAndZoom(els.mapSearchInput.value);
  });
}

// ---------- Crowd sourcing (store only; no blobs) ----------
if (els.crowdSubmit){
  els.crowdSubmit.addEventListener("click", async ()=>{
    try{
      const f = els.crowdInput?.files?.[0];
      if(!f){
        alert("Please choose an image to upload.");
        return;
      }

      // Location: inputs or EXIF
      let latVal = els.lat.value?.trim();
      let lngVal = els.lng.value?.trim();

      if ((!latVal || !lngVal) && exifLL){
        latVal = exifLL.lat.toString();
        lngVal = exifLL.lng.toString();
        els.lat.value = exifLL.lat.toFixed(6);
        els.lng.value = exifLL.lng.toFixed(6);
      }

      const latNum = parseFloat(latVal), lngNum = parseFloat(lngVal);
      if (!Number.isFinite(latNum) || !Number.isFinite(lngNum)){
        alert("Location is required. Use EXIF or set Latitude/Longitude.");
        return;
      }

      const form = new FormData();
      form.append("image", f);
      form.append("lat", latNum.toString());
      form.append("lng", lngNum.toString());

      els.crowdSubmit.disabled = true;
      setCrowdStatus("Uploading…");

      const res = await fetch("/contribute", { method: "POST", body: form });
      const text = await res.text();
      let j; try { j = JSON.parse(text); } catch { alert("Server returned non-JSON:\n"+text.slice(0,600)); return; }

      if (j.error || !res.ok){
        alert("Upload failed: " + (j.error || res.statusText));
        setCrowdStatus("Upload failed");
        return;
      }

      // store only; no map blobs
      setCrowdStatus(`Saved as ${j.id} (${j.lat.toFixed(6)}, ${j.lng.toFixed(6)})`);
      // optional UX niceties
      els.crowdInput.value = "";
    }catch(err){
      console.error("Contribute failed:", err);
      alert("Contribute failed: " + err);
      setCrowdStatus("Contribute failed");
    }finally{
      els.crowdSubmit.disabled = false;
    }
  });
}

// On load
setStatus("Ready");
