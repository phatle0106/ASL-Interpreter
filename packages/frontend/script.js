const WS_URL    = "ws://localhost:8008";
const THRESHOLD = 0.70;
const USE_CANVAS = true; // set false để test hiển thị trực tiếp <video>

const $messages    = document.getElementById('messages');
const $video       = document.getElementById('video');
const $canvas      = document.getElementById('canvas');
const $videoDot    = document.getElementById('video-dot');
const $badgeVideo  = document.getElementById('badge-video');
const $btnWS       = document.getElementById('btn-ws');
const $btnDemo     = document.getElementById('btn-demo');
const $hint        = document.getElementById('hint');
const $toggleCam   = document.getElementById('toggleCamBtn');
const $uploadDot   = document.getElementById('upload-dot');


//Nút setting----------------------------------------------------------------------------------------------------------------
// Nút điều khiển menu Setting
const $toggleSetting = document.getElementById('toggle-setting');
const $settingOptions = document.getElementById('setting-options');

// Các tùy chọn trong menu Setting
const $optionUploadVideo = document.getElementById('option-upload-video');
const $optionShowKeypoints = document.getElementById('option-show-keypoints');

// Phần Upload Video và Display Keypoints
const $videoUploadSection = document.getElementById('video-upload-section');
const $keypointsDisplay = document.getElementById('keypoints-display');

// Toggle hiển thị menu Setting
$toggleSetting.addEventListener('click', () => {
    $settingOptions.classList.toggle('hidden'); // Hiện/ẩn menu tùy chọn
});


$optionUploadVideo.addEventListener('click', () => {
    // Toggle hidden
    $videoUploadSection.classList.toggle('hidden');

    // Kiểm tra đang bật hay không để áp dụng màu dot
    const isOn = !$videoUploadSection.classList.contains('hidden');

    if (isOn) {
        $uploadDot.classList.add('on');   // Màu đỏ
    } 
    else {
        $uploadDot.classList.remove('on'); // Màu xanh
    }

    // Sau khi chọn xong, tự động ẩn menu
    $settingOptions.classList.add('hidden');
});

// Trạng thái: bật/tắt hiển thị keypoints
let isKeypointsOn = false;

// Xử lý khi chọn "Show Keypoints"
$optionShowKeypoints.addEventListener('click', () => {
    isKeypointsOn = !isKeypointsOn; // Đảo trạng thái bật/tắt

    if (isKeypointsOn) {
        $optionShowKeypoints.textContent = 'Show Keypoints (On)';
        canvasElement.style.display = 'block'; // Hiện canvas
    } 
    else {
        $optionShowKeypoints.textContent = 'Show Keypoints (Off)';
        canvasElement.style.display = 'none'; // Ẩn canvas
    }
});


//Nút điều khiển camera----------------------------------------------------------------------------------------------------------------------
let camStream = null, rafId = null, videoActive = false;

// set canvas size khi có metadata
$video.addEventListener('loadedmetadata', () => {
    if ($canvas) {
    $canvas.width  = $video.videoWidth  || 640;
    $canvas.height = $video.videoHeight || 480;
    }
});

let hands;   // Tạo biến lưu instance của Mediapipe
let ctx;     // Canvas Context để vẽ hình
let latestHandLandmarks = null;

async function initializeMediapipe() {
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 2,                   // Số lượng bàn tay tối đa nhận diện
        modelComplexity: 1,               // Độ phức tạp của model (Default: 1)
        minDetectionConfidence: 0.7,      // Độ tin cậy tối thiểu của việc phát hiện bàn tay
        minTrackingConfidence: 0.5,       // Độ tin cậy tối thiểu của việc tracking keypoint
    });

    // Khi có kết quả từ model Mediapipe
    hands.onResults((results) => {
        if (!$canvas) return;

        // Xóa canvas trước khi vẽ lại
        ctx.clearRect(0, 0, $canvas.width, $canvas.height);
        // Vẽ khung video + keypoints
        ctx.drawImage(results.image, 0, 0, $canvas.width, $canvas.height);

        // Vẽ keypoints (dùng drawing_utils từ Mediapipe)
        if (results.multiHandLandmarks && isKeypointsOn) {
            latestHandLandmarks = results.multiHandLandmarks;

            for (const landmarks of results.multiHandLandmarks) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 2,
                });
                drawLandmarks(ctx, landmarks, {
                    color: '#FF0000',
                    lineWidth: 1,
                    radius: 5,
                });
            }
        }
        else {
            latestHandLandmarks = null;
        }
    });
}

async function startCam() {
    // Khởi tạo Mediapipe nếu chưa có
    if (!hands) {
        await initializeMediapipe();
    }

    // Mở camera
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    $video.srcObject = camStream;
    ctx = $canvas.getContext('2d'); // Khởi tạo context cho canvas
    await $video.play();

    // Chạy Mediapipe nhận diện bàn tay
    const updateHands = async () => {
        if ($video.readyState >= 2 && $canvas && $canvas.width && $canvas.height) {
            await hands.send({ image: $video });
        }
        rafId = requestAnimationFrame(updateHands); // Lặp lại
    };
    rafId = requestAnimationFrame(updateHands);

    videoActive = true; 
    updateVideoBadge();
    $toggleCam.textContent = 'Turn off camera';
}

function stopCam() {
    if (camStream) camStream.getVideoTracks().forEach(t => t.stop());
    camStream = null; 
    $video.srcObject = null;
    
    // Chuyển màn hình video sang màu đen
    if ($canvas && USE_CANVAS) {
        const ctx = $canvas.getContext('2d');
        ctx.fillStyle = '#000'; // Màu đen
        ctx.fillRect(0, 0, $canvas.width, $canvas.height); // Tô toàn bộ canvas
    }

    if (rafId) cancelAnimationFrame(rafId), rafId = null;
    videoActive = false; 
    updateVideoBadge();
    $toggleCam.textContent = 'Turn on camera';
}

$toggleCam.addEventListener('click', async () => {
    try { 
        camStream ? stopCam() : await startCam(); 
    }
    catch (e) { 
        console.error(e); alert('Không mở được camera. Dùng HTTPS hoặc http://localhost và cho phép quyền.'); 
    }
});

function updateVideoBadge() {
    if (videoActive && isSendingFrames) {
        $badgeVideo.textContent = 'Playing';
        $badgeVideo.classList.add('ok');
    } 
    else if (videoActive) {
        $badgeVideo.textContent = 'Not playing';
        $badgeVideo.classList.remove('ok');
        $videoDot.style.background = '#22c55e'; 
    } 
    else {
        $badgeVideo.textContent = 'Not playing';
        $badgeVideo.classList.remove('ok');
        $videoDot.style.background = '#ef4444';
    }
}


//Nút bắt đầu gửi keypoint từ camera để cho thủ ngữ có thể nhận diện---------------------------------------------------------------------------------
let isSendingFrames = false;
let sendInterval = null;

const $btnSendFrame = document.getElementById('btn-send-frame');

// Hàm gửi khung hình lên backend qua HTTP
async function sendKeypointsToBackend(keypoints) {
    try {
        const response = await fetch('/upload-keypoints', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({keypoints}), 
            });

        if (response.ok) {
            console.log('keypoints đã được gửi thành công.');
        } 
        else {
            console.error('keypoints không thể gửi khung hình tới backend.');
        }
    } 
    catch (error) {
        console.error('Lỗi khi gửi keypoints:', error);
    }
}

// Hàm bắt đầu gửi keypoint
function startSendingKeyPoint() {
    if (!videoActive) {
        alert("Camera chưa mở! Vui lòng mở camera trước.");
        return;
    }

    sendInterval = setInterval(() => {
        // Kiểm tra nếu không có kết quả nhận diện bảng tay
        if (!latestHandLandmarks) {
            console.warn("Không tìm thấy keypoints nào để gửi.");
            return;
        }

        // Xử lý keypoints từ bàn tay được nhận diện
        const keypoints = latestHandLandmarks.map((landmarks, handIndex) =>
            landmarks.map((point) => ({
                x: point.x, // Tọa độ X (0-1)
                y: point.y, // Tọa độ Y (0-1)
                z: point.z, // Tọa độ Z (chiều sâu)
            }))
        );

        sendKeypointsToBackend(keypoints); // Gửi lên backend
    }, 300); // Gửi mỗi 300ms
}

// Hàm dừng gửi keypoint
function stopSendingKeyPoint() {
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = null;
    }
}

// Sự kiện cho nút "Bắt đầu gửi" hoặc "Ngừng gửi"
$btnSendFrame.addEventListener('click', () => {
    if (!isSendingFrames) {
        startSendingKeyPoint();
        isSendingFrames = true;
        $btnSendFrame.textContent = 'Stop send keypoint';
    } 
    else {
        stopSendingKeyPoint();
        isSendingFrames = false;
        $btnSendFrame.textContent = 'Send keypoint';
    }
    updateVideoBadge();
});


//Gửi video lên backend--------------------------------------------------------------------------------------------------------------------------
const $fileInput = document.getElementById('video-file');               // Đầu vào để chọn file
const $btnUploadVideo = document.getElementById('btn-upload-video');    // Nút upload video
const $uploadStatus = document.getElementById('upload-status');         // Thông báo trạng thái tải

// Hàm gửi video lên backend
async function uploadVideo(file) {
    try {
        $uploadStatus.textContent = 'Đang tải video lên...';            // Hiển thị trạng thái

        // Tạo một FormData object để gửi tệp dưới dạng dạng "multipart/form-data"
        const formData = new FormData();
        formData.append('video', file); // Thêm file vào form

        // Gửi tệp tới backend qua API POST
        const response = await fetch('/upload-video', {
        method: 'POST',
        body: formData,
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Kết quả từ backend:', result);
            $uploadStatus.textContent = 'Tải video lên thành công!';
        } 
        else {
            console.error('Lỗi khi tải video:', response.statusText);
            $uploadStatus.textContent = 'Tải video thất bại. Hãy thử lại.';
        }
    } 
    catch (error) {
        console.error('Lỗi khi gửi video:', error);
        $uploadStatus.textContent = 'Tải lên thất bại. Hãy thử lại.';
    }
}

// Gắn sự kiện vào nút khi được nhấn
$btnUploadVideo.addEventListener('click', () => {
    const file = $fileInput.files[0]; // Lấy file đầu vào
    if (!file) {
        alert('Vui lòng chọn một tệp video!');
        return;
    }

    // Chỉ gửi nếu file đúng định dạng
    if (!file.type.startsWith('video/')) {
        alert('Chỉ hỗ trợ tệp video!');
        return;
    }

    uploadVideo(file); // Gọi hàm tải video lên backend
});


//Chat render-----------------------------------------------------------------------------------------------------------------------------------
function pushSystem(text){
    if(!text) return;
    const div = document.createElement('div');
    div.className = 'msg system';
    div.textContent = text;
    $messages.appendChild(div);
    $messages.scrollTop = $messages.scrollHeight;
    if ($hint) $hint.style.display = 'none';
}

let _last='', _lastAt=0;
function onRecognized(payload){
    const text = typeof payload==='string' ? payload : (payload?.text ?? '');
    const conf = typeof payload==='object' ? (payload.confidence ?? payload.score ?? 1) : 1;
    if (!videoActive || !text || conf < THRESHOLD) return;
    const now = Date.now(); if (text===_last && now-_lastAt<800) return;
    _last=text; _lastAt=now;
    pushSystem(text.trim());
}
window.onRecognized = onRecognized;


//Websocket (backend)---------------------------------------------------------------------------------------------------------------------------------

// --- WebRTC signaling & streaming (replaces previous connectWS WebSocket logic) ---
// SIGNALING_URL: backend endpoint that accepts POST {sdp,type} and returns {sdp,type} answer
const SIGNALING_URL = window.SIGNALING_URL || (WS_URL.replace(/^ws(s)?:/, 'https:') + '/offer');

let pc = null;
let localStream = null;
let resultsDC = null;
let isCalling = false;

// Ensure $video, $toggleCam, onRecognized, updateVideoBadge exist in scope
async function startLocalCamera(constraints = { video: { width: 640, height: 480 }, audio: false }) {
  if (localStream) return localStream;
  try {
    localStream = await navigator.mediaDevices.getUserMedia(constraints);
    if ($video) {
      $video.srcObject = localStream;
      await $video.play().catch(()=>{});
    }
    if (typeof updateVideoBadge === 'function') updateVideoBadge();
    if ($toggleCam) $toggleCam.textContent = 'Turn off camera';
    return localStream;
  } catch (err) {
    console.error("getUserMedia error:", err);
    throw err;
  }
}

function stopLocalCamera() {
  if (localStream) {
    localStream.getTracks().forEach(t => { try { t.stop(); } catch(e){} });
    localStream = null;
  }
  if ($video) { $video.srcObject = null; }
  if (typeof updateVideoBadge === 'function') updateVideoBadge();
  if ($toggleCam) $toggleCam.textContent = 'Turn on camera';
}

// Create RTCPeerConnection and DataChannel for results
async function createPeerConnection() {
  if (pc) return pc;
  pc = new RTCPeerConnection({
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" }
      // Add TURN server here if needed
    ]
  });

  pc.oniceconnectionstatechange = () => {
    console.log("pc iceState:", pc.iceConnectionState);
    if (pc.iceConnectionState === "disconnected" || pc.iceConnectionState === "failed") {
      stopCall();
    }
  };

  // Create datachannel to receive recognition results
  resultsDC = pc.createDataChannel("results");
  resultsDC.onopen = () => console.log("results datachannel open");
  resultsDC.onclose = () => console.log("results datachannel closed");
  resultsDC.onerror = (e) => console.warn("results DC error", e);
  resultsDC.onmessage = (evt) => {
    let data = evt.data;
    try { data = JSON.parse(data); } catch(e){}
    if (typeof onRecognized === 'function') onRecognized(data);
    else console.log("Recognized:", data);
  };

  pc.ontrack = (evt) => {
    console.log("Remote track received (unused):", evt.streams);
    // In this flow we send only video to backend; backend returns results on DataChannel.
  };

  return pc;
}

// Start WebRTC call: send local video to backend and set remote description
async function startCall() {
  if (isCalling) return;
  try {
    await startLocalCamera();
    await createPeerConnection();

    // add local tracks to pc
    if (localStream) {
      for (const t of localStream.getTracks()) {
        try { pc.addTrack(t, localStream); } catch(e){ console.warn("addTrack failed", e); }
      }
    }

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // send offer to signaling endpoint
    const resp = await fetch(SIGNALING_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Signaling failed: ${resp.status} ${text}`);
    }

    const answer = await resp.json();
    await pc.setRemoteDescription(answer);

    isCalling = true;
    if (typeof updateVideoBadge === 'function') updateVideoBadge();
    console.log("WebRTC call started");
  } catch (err) {
    console.error("startCall error:", err);
    stopCall();
    alert("Không thể bắt đầu WebRTC call: " + (err.message || err));
  }
}

function stopCall() {
  if (resultsDC) try { resultsDC.close(); } catch(e) {}
  resultsDC = null;

  if (pc) {
    try { pc.getSenders().forEach(s => { try { s.track && s.track.stop(); } catch(e){} }); } catch(e){}
    try { pc.close(); } catch(e) {}
    pc = null;
  }

  stopLocalCamera();
  isCalling = false;
  if (typeof updateVideoBadge === 'function') updateVideoBadge();
}

// Wire btn-ws to start/stop call (replace original WS button behavior)
const btnWs = document.getElementById('btn-ws');
if (btnWs) {
  btnWs.addEventListener('click', async () => {
    if (!isCalling) {
      await startCall();
      btnWs.textContent = 'Stop stream';
    } else {
      stopCall();
      btnWs.textContent = 'Start stream';
    }
  });
}

// Clean up on unload
window.addEventListener('beforeunload', () => {
  stopCall();
});
// --- end WebRTC replacement ---



//Demo--------------------------------------------------------------------------------------------------------------------------------------------------------------
document.getElementById('btn-demo').addEventListener('click', ()=>{
    if (!videoActive) { videoActive = true; updateVideoBadge(); }
    const demo = ['Demo start','Nhận diện: Xin chào','Chuyển văn bản → chat','Kết thúc'];
    let i=0; const id=setInterval(()=>{ if(i<demo.length) onRecognized({text:demo[i++],confidence:0.95}); else clearInterval(id); },800);
});

// Seed
pushSystem('Panel is ready. Ready to get word from AI');